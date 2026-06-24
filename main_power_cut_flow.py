import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

import database

load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("MainPowerCutFlow")

app = FastAPI(title="GPS Outage Workflow - Case 2: Main Power Cut Service")

# Initialize Azure OpenAI Client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ==============================================================================
# PYDANTIC DATA STRUCTS
# ==============================================================================

class GpsData(BaseModel):
    gpstime: Optional[str] = None
    main_powervoltage: Optional[float] = None
    ismainpoerconnected: Optional[str] = None
    gpsStatus: Optional[int] = None
    driver_name: Optional[str] = None
    driver_phone: Optional[str] = None
    current_location: Optional[str] = None
    vehicle_state: Optional[str] = None

class RoutedRequest(BaseModel):
    root_cause: str
    phone_number: str
    vehicle_no: str
    last_location: str
    timestamp: str
    gps_data: Optional[GpsData] = None

class WhatsAppWebhookMessage(BaseModel):
    phone_number: str
    message_text: str


async def start_main_power_flow(payload: dict):
    """
    Entry point called by main.py routing logic.
    Initializes main power cut flow.
    """
    return await handle_main_power_cut(RoutedRequest(**payload))


# ==============================================================================
# UTILITY COMM LAYER
# ==============================================================================

def send_whatsapp_meta(to_number: str, text_body: str):
    """Dispatches asynchronous notification alerts over Meta API WhatsApp channel."""
    url = f"https://graph.facebook.com/v18.0/{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text_body}
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=8)
        if res.status_code != 200:
            logger.error(f"WhatsApp dispatch failed: Code {res.status_code} - {res.text}")
    except Exception as e:
        logger.critical(f"Transport failure dispatching message context to {to_number}: {e}")


# ==============================================================================
# NATURAL LANGUAGE PROCESSING ENTITY EXTRACTION LAYER
# ==============================================================================

def llm_parse_text(user_input: str, system_prompt: str) -> str:
    """Invokes downstream generative parsing model to isolate operational keys."""
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        logger.error(f"LLM Parsing structural breakdown exception raised: {e}")
        return "OTHER"


def extract_entities_from_message(user_input: str, current_context: str, current_state: str) -> dict:
    """
    Extracts ALL entities and intent from customer message BEFORE any state transition.
    This ensures we never ignore information provided by the customer.
    """
    prompt = f"""
You are analyzing a customer's WhatsApp message in a GPS troubleshooting conversation.

Current State: {current_state}
Current Context: {current_context}

Customer Message: "{user_input}"

Extract ALL information and intent from this message. Return a JSON object:
{{
  "driver_name": string or null,
  "driver_phone": string or null (10-digit phone number only),
  "contact_driver_requested": true/false,
  "customer_will_handle": true/false,
  "answered_current_question": true/false,
  "needs_guidance": true/false,
  "asked_question": true/false,
  "confused": true/false,
  "issue_resolved": true/false/null,
  "work_completed": true/false/null
}}

Extraction Rules:
- driver_phone: Extract any 10-digit Indian phone number (with or without country code)
- driver_name: Extract any person name mentioned
- contact_driver_requested: true if they say "driver se baat karo", "driver ko bolo", etc.
- customer_will_handle: true if they say "main check kar leta hu", "haan check karwa deta hu", etc.
- answered_current_question: true if they answered the question we asked
- needs_guidance: true if asking HOW to do something (kaise, how, procedure)
- asked_question: true if asking ANY question (?, kya, kaise, kab, kahan, kaun)
- confused: true if unclear about what we're asking
- issue_resolved: true if GPS is working now, false if still problem, null if not mentioned
- work_completed: true if power connection work is done, false if not, null if not mentioned

Important: 
- Extract phone even if embedded in other text like "918290323758 isse baat kar lo"
- Extract name even if with number like "Rahul 9876543210"
- Customer can answer AND provide data in same message

Return ONLY valid JSON, no other text.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a data extraction expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        import json
        extracted = json.loads(result)
        logger.info(f"[Entity Extraction] {extracted}")
        return extracted
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        # Default to safe behavior
        return {
            "driver_name": None,
            "driver_phone": None,
            "contact_driver_requested": False,
            "customer_will_handle": False,
            "answered_current_question": True,
            "needs_guidance": False,
            "asked_question": False,
            "confused": False,
            "issue_resolved": None,
            "work_completed": None
        }


def merge_extracted_entities(context: dict, extracted: dict) -> dict:
    """
    Merges extracted entities into session context.
    Always updates with new information when provided.
    """
    # Update driver details if provided
    if extracted.get("driver_phone"):
        context["driver_phone"] = extracted["driver_phone"]
        logger.info(f"[Entity Merge] Updated driver_phone: {extracted['driver_phone']}")
    
    if extracted.get("driver_name"):
        context["driver_name"] = extracted["driver_name"]
        logger.info(f"[Entity Merge] Updated driver_name: {extracted['driver_name']}")
    
    # Update handler choice if indicated
    if extracted.get("contact_driver_requested"):
        context["power_check_handler"] = "DRIVER"
        logger.info(f"[Entity Merge] Set handler to DRIVER")
    
    if extracted.get("customer_will_handle"):
        context["power_check_handler"] = "CUSTOMER"
        logger.info(f"[Entity Merge] Set handler to CUSTOMER")
    
    return context


def generate_contextual_response(user_input: str, current_question: str, context: dict, state: str) -> str:
    """
    Generates a helpful response to customer's question or confusion.
    This provides guidance without changing the state.
    Uses predefined templates for consistency and accuracy.
    """
    vehicle_no = context.get("vehicle_no", "vehicle")
    last_location = context.get("last_location", "N/A")
    
    user_lower = user_input.lower()
    
    # SCENARIO 2: Customer asks HOW to check wire
    if any(word in user_lower for word in ["kaise", "how", "check karu", "kaise check"]):
        if state == "MAIN_POWER_INITIAL_RESPONSE":
            return (
                "GPS device ki wiring aur power connection check kar lijiye.\n"
                "Agar koi wire loose ya disconnected ho to use properly connect karwa dijiye.\n\n"
                "Aap check karwa lenge ya driver se baat karni hogi?"
            )
        else:
            return (
                "GPS device ke connections check karein - red/yellow wire main power hoti hai. "
                "Dekhe ki properly connected aur tight hai. Loose ho toh secure kar dein."
            )
    
    # SCENARIO 3: Customer asks what is the problem
    if any(word in user_lower for word in ["problem kya", "kya problem", "issue kya", "what is problem", "what problem"]):
        return (
            "Hamare system ke anusaar GPS device ka main power connection disconnected dikh raha hai.\n"
            "Isi wajah se GPS update receive nahi ho raha ho sakta.\n\n"
            "Aap check karwa sakte hain ya driver se baat karni hogi?"
        )
    
    # SCENARIO 9: Customer asks where is vehicle
    if any(word in user_lower for word in ["kaha hai", "kahan hai", "location", "vehicle kaha", "gaadi kaha"]):
        location_response = f"Vehicle ki last known location {last_location} hai.\n"
        if state == "MAIN_POWER_INITIAL_RESPONSE":
            location_response += (
                "Saath hi GPS ka main power connection disconnected dikh raha hai.\n\n"
                "Aap check karwa sakte hain ya driver se baat karni hogi?"
            )
        return location_response
    
    # Customer asks about timing
    if any(word in user_lower for word in ["kitna time", "kab tak", "how long", "kitne din"]):
        return (
            "Usually 15-30 minute mein check ho jata hai, vehicle ki location pe depend karta hai.\n"
            "Power connection check hone ke baad hum GPS status verify kar lenge."
        )
    
    # Customer asks who is driver
    if any(word in user_lower for word in ["driver kaun", "driver kon", "who is driver"]):
        driver_name = context.get("driver_name", "N/A")
        driver_phone = context.get("driver_phone", "N/A")
        return f"Hamare record ke anusaar driver {driver_name} ({driver_phone}) hain."
    
    # SCENARIO 8: Customer asks what will driver check
    if any(word in user_lower for word in ["kya check", "what check", "kya dekh", "wo kya"]):
        return (
            "Driver ko GPS device ki wiring aur power connection check karna hoga.\n"
            "Main power wire properly connected hai ya nahi, ye verify karna hai."
        )
    
    # Generic confusion - provide general clarification
    if any(word in user_lower for word in ["samajh nahi", "confused", "matlab", "means what"]):
        return (
            "Main pooch raha hu ki power connection check karne ke liye - "
            "aap khud check karwa sakte hain ya driver se baat karni hogi?\n"
            "Agar aap check karwa sakte hain toh 'haan' kahein, "
            "nahi toh 'driver se baat karo' kahein."
        )
    
    # Fallback - use LLM for other questions
    try:
        prompt = f"""
You are a GPS support assistant. Answer this customer question briefly in Hindi/Hinglish (2-3 sentences max).

Current Context: We're asking if customer or driver will check power connection for vehicle {vehicle_no}.
Last Known Location: {last_location}
Issue: GPS main power connection disconnected

Customer Question: "{user_input}"

Provide a helpful answer in Hindi/Hinglish, then remind them of the current step.
"""
        
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a helpful GPS support assistant. Answer briefly in Hindi/Hinglish."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating contextual response: {e}")
        return (
            "Main samajh gaya. Kripya batayein - aap power connection check karwa sakte hain "
            "ya driver se baat karni hogi?"
        )

@app.post("/api/flow/main-power-cut")
async def handle_main_power_cut(payload: RoutedRequest):
    """
    Entry point for Main Power Cut flow.
    Sends initial message asking if customer or driver will check power connection.
    """
    gps_data = payload.gps_data.model_dump() if payload.gps_data else {}
    gpstime = gps_data.get("gpstime", "N/A")
    last_location = gps_data.get("current_location") or payload.last_location
    vehicle_state = gps_data.get("vehicle_state", "N/A")

    # Initial message asking if customer or driver will handle the power check
    alert_msg = (
        f"Vehicle {payload.vehicle_no} ka GPS update nahi aa raha hai kyunki system mein "
        f"GPS ka main power connection disconnected dikh raha hai.\n\n"
        f"Kripya wiring aur power connection check karwa dijiye.\n\n"
        f"*Aap power connection check karwa sakte hain ya driver se baat karni hogi?*"
    )

    # Initialize the database tracking state for Main Power flow
    database.save_session(
        phone_number=payload.phone_number,
        current_state="MAIN_POWER_INITIAL_RESPONSE",
        collected_json={
            "vehicle_no": payload.vehicle_no,
            "root_cause": "MAIN_POWER",
            "driver_name": gps_data.get("driver_name"),
            "driver_phone": gps_data.get("driver_phone"),
            "last_location": last_location,
            "gpstime": gpstime,
            "vehicle_state": vehicle_state,
            "main_power_driver_contacted": False,
            "power_check_handler": None,  # 'CUSTOMER' or 'DRIVER'
        },
        chat_history=[{"role": "bot", "text": alert_msg}]
    )

    send_whatsapp_meta(payload.phone_number, alert_msg)
    logger.info(f"Main Power Cut flow initialized for {payload.vehicle_no}")
    return {"status": "flow_initialized", "case": "MAIN_POWER"}


@app.post("/api/flow/main-power-cut/webhook")
async def handle_whatsapp_replies(webhook_data: WhatsAppWebhookMessage):
    """
    Main state engine for Main Power Cut flow.
    Handles all incoming WhatsApp messages and routes based on current state.
    """
    user_phone = webhook_data.phone_number
    incoming_text = webhook_data.message_text.strip()

    # Fetch active database context profile
    session = database.get_session(user_phone)
    if not session:
        return {"status": "ignored", "reason": "No active conversation session sequence detected."}

    state = session.get("current_state")
    context = session.get("collected_json", {})
    history = session.get("chat_history", [])
    vehicle_no = context.get("vehicle_no", "Unknown")

    logger.info(f"[Main Power] Processing phone: {user_phone} | State: {state}")

    # Append customer message to history
    history.append({"role": "user", "content": incoming_text})

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: MAIN_POWER_INITIAL_RESPONSE
    # Customer says: will check myself OR talk to driver
    # ──────────────────────────────────────────────────────────────────────────
    if state == "MAIN_POWER_INITIAL_RESPONSE":
        current_question = "Aap power connection check karwa sakte hain ya driver se baat karni hogi?"
        
        # STEP 1: Extract ALL entities from message FIRST
        extracted = extract_entities_from_message(
            incoming_text,
            f"We asked: {current_question}. We need to know if customer or driver will check the power connection.",
            state
        )
        
        # STEP 2: Merge extracted entities into context BEFORE any decisions
        context = merge_extracted_entities(context, extracted)
        
        # STEP 3: Handle questions/guidance if needed
        if extracted.get("asked_question") or extracted.get("needs_guidance") or extracted.get("confused"):
            guidance_response = generate_contextual_response(incoming_text, current_question, context, state)
            history.append({"role": "assistant", "content": guidance_response})
            database.save_session(user_phone, state, context, history)  # Save with updated context
            send_whatsapp_meta(user_phone, guidance_response)
            return {"status": "guidance_provided"}
        
        # STEP 4: Check if they answered
        if not extracted.get("answered_current_question"):
            clarification = "Kripya batayein - aap khud power connection check karwa sakte hain ya driver se baat karni hogi?"
            database.save_session(user_phone, state, context, history)  # Save context even if not proceeding
            send_whatsapp_meta(user_phone, clarification)
            return {"status": "awaiting_answer"}
        
        # STEP 5: Now determine state transition based on UPDATED context
        # Check if we already have new driver phone directly provided
        if extracted.get("driver_phone") and extracted.get("contact_driver_requested"):
            # Customer provided new driver number directly: "918290323758 isse baat kar lo"
            driver_phone = context.get("driver_phone")
            driver_name = context.get("driver_name", "Driver")
            
            # Send message directly to new driver
            driver_msg = (
                f"Namaste,\n\n"
                f"Vehicle *{vehicle_no}* ke GPS ka main power connection disconnected dikh raha hai.\n\n"
                f"Kripya GPS wiring aur power connection check karke theek karwa dijiye aur update dein."
            )
            send_whatsapp_meta(driver_phone, driver_msg)
            context["main_power_driver_contacted"] = True
            
            mgr_msg = f"Driver ko message bhej diya gaya hai. Hum power check complete hone ka wait karenge."
            history.append({"role": "assistant", "content": mgr_msg})
            database.save_session(user_phone, "MAIN_POWER_WAITING_FOR_CHECK", context, history)
            send_whatsapp_meta(user_phone, mgr_msg)
            return {"status": "processed"}
        
        # Otherwise, use extracted handler choice
        handler = context.get("power_check_handler")
        
        if handler == "CUSTOMER":
            next_msg = (
                "Theek hai. Power connection check hone ke baad hum GPS status verify karenge."
            )
            history.append({"role": "assistant", "content": next_msg})
            database.save_session(user_phone, "MAIN_POWER_WAITING_FOR_CHECK", context, history)
            send_whatsapp_meta(user_phone, next_msg)

        elif handler == "DRIVER":
            d_name = context.get("driver_name", "N/A")
            d_phone = context.get("driver_phone", "N/A")
            
            next_msg = (
                f"Hamare record ke anusaar driver *{d_name}* ({d_phone}) hain.\n\n"
                f"Kya isi driver se baat karein ya koi aur contact number hai?"
            )
            history.append({"role": "assistant", "content": next_msg})
            database.save_session(user_phone, "MAIN_POWER_DRIVER_CONFIRMATION", context, history)
            send_whatsapp_meta(user_phone, next_msg)
        else:
            fallback = "Kripya spasht batayein - aap khud power connection check karwa sakte hain ya driver se baat karni hogi?"
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, fallback)

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: MAIN_POWER_DRIVER_CONFIRMATION
    # Customer confirms same driver OR provides new driver details
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "MAIN_POWER_DRIVER_CONFIRMATION":
        current_question = f"Driver {context.get('driver_name', 'N/A')} ({context.get('driver_phone', 'N/A')}) - Kya isi driver se baat karein ya koi aur contact number hai?"
        
        # STEP 1: Extract ALL entities from message FIRST
        extracted = extract_entities_from_message(
            incoming_text,
            f"We asked: {current_question}. We need confirmation to use existing driver or new driver contact.",
            state
        )
        
        # STEP 2: Merge extracted entities into context BEFORE any decisions
        context = merge_extracted_entities(context, extracted)
        
        # STEP 3: Handle questions/guidance if needed
        if extracted.get("asked_question") or extracted.get("needs_guidance") or extracted.get("confused"):
            guidance_response = generate_contextual_response(incoming_text, current_question, context, state)
            history.append({"role": "assistant", "content": guidance_response})
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, guidance_response)
            return {"status": "guidance_provided"}
        
        # STEP 4: Determine action based on extracted data
        driver_phone = context.get("driver_phone")
        driver_name = context.get("driver_name", "Driver")
        
        # Check if new phone number was provided
        if extracted.get("driver_phone"):
            # New driver number provided - send message directly
            driver_msg = (
                f"Namaste,\n\n"
                f"Vehicle *{vehicle_no}* ke GPS ka main power connection disconnected dikh raha hai.\n\n"
                f"Kripya GPS wiring aur power connection check karke theek karwa dijiye aur update dein."
            )
            send_whatsapp_meta(driver_phone, driver_msg)
            context["main_power_driver_contacted"] = True
            
            mgr_msg = f"Driver ko message bhej diya gaya hai ({driver_name} - {driver_phone}). Power check complete hone ka wait karenge."
            history.append({"role": "assistant", "content": mgr_msg})
            database.save_session(user_phone, "MAIN_POWER_WAITING_FOR_CHECK", context, history)
            send_whatsapp_meta(user_phone, mgr_msg)
            return {"status": "processed"}
        
        # Otherwise check if they confirmed existing driver
        confirmation_prompt = (
            "Analyze the response in Hindi, English, or Hinglish.\n"
            "Return 'YES' if customer confirms using the existing/current driver (haan, sahi hai, yes, correct, isi se baat karo, etc.)\n"
            "Return 'OTHER' if unclear.\n"
        )
        confirmation = llm_parse_text(incoming_text, confirmation_prompt)

        if "YES" in confirmation:
            # Use existing driver details and send message
            if driver_phone and driver_phone != "N/A":
                driver_msg = (
                    f"Namaste,\n\n"
                    f"Vehicle *{vehicle_no}* ke GPS ka main power connection disconnected dikh raha hai.\n\n"
                    f"Kripya GPS wiring aur power connection check karke theek karwa dijiye aur update dein."
                )
                send_whatsapp_meta(driver_phone, driver_msg)
                context["main_power_driver_contacted"] = True
                
                mgr_msg = f"Driver {driver_name} ko message bhej diya gaya hai. Hum power check complete hone ka wait karenge."
                history.append({"role": "assistant", "content": mgr_msg})
                database.save_session(user_phone, "MAIN_POWER_WAITING_FOR_CHECK", context, history)
                send_whatsapp_meta(user_phone, mgr_msg)
            else:
                error_msg = "Driver phone number database mein nahi mila. Kripya driver ka naam aur number provide karein."
                database.save_session(user_phone, state, context, history)
                send_whatsapp_meta(user_phone, error_msg)
        else:
            fallback = "Kripya batayein - kya existing driver se baat karein ya nayi driver details provide karein (Naam, Phone Number)?"
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, fallback)

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: MAIN_POWER_WAITING_FOR_CHECK
    # Wait for update that power connection has been fixed
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "MAIN_POWER_WAITING_FOR_CHECK":
        current_question = "Power connection check ho raha hai. Jab complete ho jaye, kripya update dijiye."
        
        # STEP 1: Extract ALL entities from message FIRST
        extracted = extract_entities_from_message(
            incoming_text,
            f"We are waiting for customer to confirm power connection has been checked and fixed. {current_question}",
            state
        )
        
        # STEP 2: Merge any new entities (though unlikely in this state)
        context = merge_extracted_entities(context, extracted)
        
        # STEP 3: Handle questions/guidance if needed
        if extracted.get("asked_question") or extracted.get("needs_guidance") or extracted.get("confused"):
            guidance_response = generate_contextual_response(incoming_text, current_question, context, state)
            history.append({"role": "assistant", "content": guidance_response})
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, guidance_response)
            return {"status": "guidance_provided"}
        
        # STEP 4: Check if work is complete from extracted data
        if extracted.get("work_completed") is True:
            next_msg = (
                "Dhanyavaad. Kya GPS update aana shuru ho gaya hai ya abhi bhi problem hai?"
            )
            history.append({"role": "assistant", "content": next_msg})
            database.save_session(user_phone, "MAIN_POWER_POST_CHECK", context, history)
            send_whatsapp_meta(user_phone, next_msg)
        elif extracted.get("work_completed") is False:
            waiting_msg = "Theek hai. Jab power connection check ho jaye, kripya update dijiye."
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, waiting_msg)
        else:
            # Unclear - stay in state and ask for update
            waiting_msg = "Kripya batayein - kya power connection check ho gaya hai?"
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, waiting_msg)

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: MAIN_POWER_POST_CHECK
    # Check if GPS is now working after power fix
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "MAIN_POWER_POST_CHECK":
        current_question = "Kya GPS update aana shuru ho gaya hai ya abhi bhi problem hai?"
        
        # STEP 1: Extract ALL entities from message FIRST
        extracted = extract_entities_from_message(
            incoming_text,
            f"We asked: {current_question}. We need to know if GPS is working now after power fix.",
            state
        )
        
        # STEP 2: Merge any new entities
        context = merge_extracted_entities(context, extracted)
        
        # STEP 3: Handle questions/guidance if needed
        if extracted.get("asked_question") or extracted.get("needs_guidance") or extracted.get("confused"):
            guidance_response = generate_contextual_response(incoming_text, current_question, context, state)
            history.append({"role": "assistant", "content": guidance_response})
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, guidance_response)
            return {"status": "guidance_provided"}
        
        # STEP 4: Check resolution status from extracted data
        if extracted.get("issue_resolved") is True:
            success_msg = (
                "Dhanyavaad. GPS update normal dikh raha hai. Issue resolve ho gaya hai. 🎉"
            )
            history.append({"role": "assistant", "content": success_msg})
            database.save_session(user_phone, "COMPLETED", context, history)
            send_whatsapp_meta(user_phone, success_msg)
            # Clean up session after successful resolution
            database.delete_session(user_phone)
        elif extracted.get("issue_resolved") is False:
            # Issue not resolved - move to GPS_INACTIVE_PROBE
            escalation_msg = (
                "Samajh gaya. Power connection theek hone ke baad bhi GPS inactive hai.\n\n"
                "Hum is issue ko further investigate karenge. "
                "Hamare technical team aapse jald hi contact karegi."
            )
            history.append({"role": "assistant", "content": escalation_msg})
            database.save_session(user_phone, "GPS_INACTIVE_PROBE", context, history)
            send_whatsapp_meta(user_phone, escalation_msg)
            logger.info(f"[Main Power] Issue not resolved for {vehicle_no}. Moving to GPS_INACTIVE_PROBE state.")
        else:
            # Unclear - ask again
            clarification = "Kripya spasht batayein - kya GPS ab kaam kar raha hai ya abhi bhi problem hai?"
            database.save_session(user_phone, state, context, history)
            send_whatsapp_meta(user_phone, clarification)

    return {"status": "processed"}