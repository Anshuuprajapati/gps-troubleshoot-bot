import os
import json
import random
import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime, date
import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

import database
from date_utils import normalize_date

load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("OtherIssueFlow")

app = FastAPI(title="GPS Outage Workflow - Case 3: Other Issues Handler")

# Initialize Azure OpenAI Client
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# ==============================================================================
# PYDANTIC SCHEMAS
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
    standing_hours: Optional[float] = None  # Populated upstream or from telematics


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


# ==============================================================================
# HELPERS (REUSED FROM CENTRAL CORE)
# ==============================================================================

def send_whatsapp_meta(to_number: str, text_body: str):
    url = f"https://graph.facebook.com/v18.0/{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    headers = {
        "Authorization": f"Bearer {os.getenv('WHATSAPP_TOKEN')}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text_body},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            logger.error(f"[META API ERROR] {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"[META SEND EXCEPTION] {e}")


# ==============================================================================
# SYSTEM DEFINITION & SINGLE-ENGINE PROMPT
# ==============================================================================

_DYNAMIC_BRAIN_SYSTEM = """
You are an automated human-like conversational support agent dealing with vehicle tracking downtime.
Your job is to analyze the conversation history and the latest message to determine intent, sub-intent, extract data, and intelligently handle scheduling rejections.

## MASTER INTENT CLASSIFICATION:
- WORKSHOP: Vehicle is at a service point, body shop, workshop, or undergoing garage maintenance.
- ACCIDENT: Vehicle met with a highway collision, structural damage, or breakdown.
- VEHICLE_RUNNING: Vehicle is driving, moving, or on an active shipment trip.
- VEHICLE_STANDING: Vehicle is parked at an open plot, yard, home, or factory securely.
- GPS_DAMAGED: Physical tracker wires were cut, broken, burned, or destroyed.
- GPS_REMOVED: The device was physically unplugged, detached, stolen, or stored separately.
- OTHER: Unclear issue or side-chatter.

## CRITICAL OPERATIONAL LOGIC:
1. **Entity Extraction**: Extract `vehicle_location`, `driver_name`, `driver_phone`, `workshop_name`, `resume_date` (when the vehicle runs again), and `service_date` (when an engineer fixes it).
2. **Scheduling Loop Tracking**:
   - If a slot suggestion is rejected, look at the last `scheduling_step` value.
   - If the user explicitly rejects the suggested time, increment the `scheduling_step` or mark `step_rejected: true` so the code can advance to the next strategy level (+4 days, +5 days, +7 days, then Next Trip details).
   - If they agree to a location/date or supply one themselves, capture it immediately.
3. **GPS_REMOVED Rule**: If the tracker is self-removed, look out for whether they want a service engineer visit (`wants_service_visit`). If they don't, we will just close the loop on `resume_date`.

## RESPONSE SCHEMA (STRICT - Return ONLY valid JSON):
{
  "intent": "WORKSHOP" | "ACCIDENT" | "VEHICLE_RUNNING" | "VEHICLE_STANDING" | "GPS_DAMAGED" | "GPS_REMOVED" | "OTHER" | null,
  "entities": {
    "vehicle_location": string or null,
    "driver_phone": string or null,
    "driver_name": string or null,
    "workshop_name": string or null,
    "resume_date": string or null,
    "service_date": string or null,
    "next_trip_date": string or null,
    "next_trip_location": string or null
  },
  "wants_service_visit": boolean | null,
  "is_in_workshop_currently": boolean | null,
  "slot_rejected": boolean,
  "side_question_reply": string or null,
  "conversational_reply": string
}
"""

def merge_extracted_data(existing: dict, new_data: dict) -> dict:
    merged = dict(existing)
    for key, value in new_data.items():
        if value is not None and str(value).strip().lower() != "null":
            merged[key] = value
    return merged


def is_phone_refusal_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(
        r"\b(nahi dena|na dena|baad me|baad mein|no number|number nahi hai|privacy|security|nahi chahiye)\b", 
        cleaned
    ))


# ==============================================================================
# ENTRY ROUTE FOR CENTRAL HUB
# ==============================================================================

async def start_other_issue_flow(payload: dict):
    phone_number = payload.get("phone_number")
    vehicle_no = payload.get("vehicle_no")
    last_location = payload.get("last_location")
    timestamp = payload.get("timestamp")
    gps_data = payload.get("gps_data", {}) or {}

    gpstime = gps_data.get("gpstime")
    standing_hrs = gps_data.get("standing_hours") or 0.0

    initial_alert_msg = (
        f"Namaste Sir,\n\n"
        f"Vehicle {vehicle_no} se GPS data receive nahi ho raha hai.\n\n"
        f"📍 Last Known Location: {last_location}\n"
        f"🕐 Last Update: {timestamp or gpstime}\n\n"
        f"Kripya batayein ki aapki vehicle ki current status kya hai:\n\n"
        f"1️⃣ Workshop / Service Center\n"
        f"2️⃣ Accident\n"
        f"3️⃣ Vehicle Running but GPS Not Updating\n\n"
        f"Kripya short me batayein taaki hum status update kar sakein."
    )

    database.save_session(
        phone_number,
        "INITIAL_ALERT",
        {
            "root_cause": "OTHER_ISSUE",
            "vehicle_no": vehicle_no,
            "last_location": last_location,
            "timestamp": timestamp,
            "standing_hours": standing_hrs,
            "intent": None,
            "vehicle_location": last_location or None,
            "service_date": None,
            "resume_date": None,
            "next_trip_date": None,
            "next_trip_location": None,
            "driver_phone": gps_data.get("driver_phone"),
            "driver_name": gps_data.get("driver_name"),
            "workshop_name": None,
            "is_in_workshop_currently": None,
            "wants_service_visit": None,
            "scheduling_step": 0, 
            "ticket_id": None
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )

    send_whatsapp_meta(phone_number, initial_alert_msg)
    return {"status": "flow_initialized", "case": "OTHER_ISSUE"}


# ==============================================================================
# WEBHOOK HANDLER ENGINE
# ==============================================================================

async def handle_whatsapp_replies(msg: WhatsAppWebhookMessage) -> dict:
    phone = msg.phone_number
    message = msg.message_text.strip()
    
    session = database.get_session(phone)
    if not session:
        return {"status": "no_active_session"}

    current_state = session.get("current_state", "INITIAL_ALERT")
    collected = session.get("collected_json", {})
    chat_hist = session.get("chat_history", [])

    # Keep a running short memory
    llm_messages = []
    for entry in chat_hist[-8:]:
        role = "assistant" if entry.get("role") == "bot" else "user"
        content = entry.get("text") or entry.get("content") or ""
        llm_messages.append({"role": role, "content": content})
    llm_messages.append({"role": "user", "content": message})

    # Execute AI Brain Parsing
    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _DYNAMIC_BRAIN_SYSTEM},
                *llm_messages
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=500
        )
        brain = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"[LLM Brain Exception] {e}")
        return {"status": "error_processing_llm"}

    # Intent Assignment
    if brain.get("intent") and not collected.get("intent"):
        collected["intent"] = brain["intent"]
    
    current_intent = collected.get("intent")
    
    # Merge Extracted Entities Safely
    ext_entities = brain.get("entities", {}) or {}
    collected = merge_extracted_data(collected, ext_entities)

    if brain.get("is_in_workshop_currently") is not None:
        collected["is_in_workshop_currently"] = brain["is_in_workshop_currently"]
    if brain.get("wants_service_visit") is not None:
        collected["wants_service_visit"] = brain["wants_service_visit"]

    # Normalize Dates Deterministically
    if ext_entities.get("service_date"):
        collected["service_date"] = normalize_date(ext_entities["service_date"])
    if ext_entities.get("resume_date"):
        collected["resume_date"] = normalize_date(ext_entities["resume_date"])
    if ext_entities.get("next_trip_date"):
        collected["next_trip_date"] = normalize_date(ext_entities["next_trip_date"])

    if is_phone_refusal_response(message):
        collected["driver_phone"] = "NOT_PROVIDED"

    chat_hist.append({"role": "user", "text": message})
    prefix_reply = f"{brain.get('side_question_reply')}\n\n" if brain.get('side_question_reply') else ""

    # Check Standing Duration Condition Threshold Upstream
    standing_hours = collected.get("standing_hours", 0.0)

    # ==========================================================================
    # WORKFLOW BRANCH ROUTING
    # ==========================================================================

    # 1. WORKSHOP FLOW
    if current_intent == "WORKSHOP":
        if not collected.get("workshop_name"):
            reply_msg = f"{prefix_reply}Workshop/Service center ka naam kya hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_workshop_name"}
        
        if not collected.get("resume_date"):
            reply_msg = f"{prefix_reply}Vehicle kab tak ready hoke road par chalne lagegi?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_resume_date"}

        # Termination without forcing service tickets
        reply_msg = "✅ Update note kar liya gaya hai. Vehicle ready hone par tracking normal ho jayegi. Dhanyavaad!"
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        database.delete_session(phone)
        return {"status": "workshop_case_closed"}

    # 2. ACCIDENT FLOW
    elif current_intent == "ACCIDENT":
        if collected.get("is_in_workshop_currently") is None:
            reply_msg = f"{prefix_reply}Kya gaadi abhi kisi workshop ya garage me khadi hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "probing_accident_workshop"}
        
        if not collected.get("resume_date"):
            reply_msg = f"{prefix_reply}Gaadi kab tak running condition me aa jayegi?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_resume_date"}

        # Termination with no ticket dispatch
        reply_msg = "✅ Details register kar li gayi hain. Emergency update backend me push ho gaya hai. Take care!"
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        database.delete_session(phone)
        return {"status": "accident_case_closed"}

    # 3. VEHICLE STANDING LONG SLEEP CHECK (> 48 Hours)
    elif current_intent == "VEHICLE_STANDING" and standing_hours > 48:
        if not collected.get("resume_date"):
            reply_msg = f"{prefix_reply}Gaadi 48 ghante se jyada se stationary hai. Agli trip kis date ko nikalne wali hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_resume_date_standing"}
        
        reply_msg = "✅ Information update ho gayi hai. Gaadi next run par aate hi data transmission automatic update ho jayega."
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        database.delete_session(phone)
        return {"status": "standing_long_duration_closed"}

    # 4. GPS SELF REMOVED DETECTOR
    elif current_intent == "GPS_REMOVED":
        if not collected.get("resume_date"):
            reply_msg = f"{prefix_reply}GPS device vehicle me wapas kab tak connect/reinstall ho jayega?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_gps_reinstall_date"}
        
        if collected.get("wants_service_visit") is None:
            reply_msg = f"{prefix_reply}Kya aapko reinstall karne ke liye hamare physical service engineer ki zaroorat hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "probing_service_requirement"}
        
        if collected.get("wants_service_visit") is False:
            reply_msg = "✅ Sahi hai, aap jab device plug karenge tracking start ho jayegi. Case update log kar diya hai."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            database.delete_session(phone)
            return {"status": "gps_removed_self_fix_closed"}
        
        # If user explicitly wants service, auto-converge cleanly down to Service Slot-Filling
        pass

    # 5. CORE FIELD SERVICE SLOT-FILLING ENGINE
    # Triggers for: VEHICLE_RUNNING, GPS_DAMAGED, VEHICLE_STANDING (<48h), and GPS_REMOVED (when visit is True)
    if current_intent in ["VEHICLE_RUNNING", "GPS_DAMAGED", "VEHICLE_STANDING", "GPS_REMOVED"]:
        
        # Step A: Validate Current City Presence
        if not collected.get("vehicle_location"):
            reply_msg = f"{prefix_reply}Gaadi abhi kis city/location par chal rahi hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_location"}

        # Step B: Informative Point Recommendation Mapped to extracted vehicle location
        loc = collected.get("vehicle_location")
        
        # Step C: Dynamic Temporal Suggestion Negotiator Loop
        step = collected.get("scheduling_step", 0)
        if brain.get("slot_rejected") is True:
            step += 1
            collected["scheduling_step"] = step

        if not collected.get("service_date"):
            # Step Tier 0: Current Time Logic Strategy Selector
            if step == 0:
                current_hour = datetime.now().hour
                if current_hour < 12:
                    reply_msg = f"{prefix_reply}Aapke current route par *{loc} service point* upalabdh hai. Kya hum aaj *shaam* tak inspection schedule kar dein?"
                else:
                    reply_msg = f"{prefix_reply}Aapke area ke hisab se *{loc} service counter* check ho sakta hai. Kya hum isko *kal* ke liye fix karein?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_date_step_0"}
            
            # Step Tier 1: 4 Days Deferred
            elif step == 1:
                reply_msg = "Koi baat nahi sir, kya phir 4 din baad ka appointment set kar dein?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_date_step_1"}
            
            # Step Tier 2: 5 or 7 Days Deferred
            elif step == 2:
                reply_msg = "Aapke scheduling ke hisab se, kya 5 se 7 dino ke baad inspection karwana sahi rahega?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_date_step_2"}
            
            # Step Tier 3: Fallback straight into Next Trip Tracking Capture
            else:
                if not collected.get("next_trip_date") or not collected.get("next_trip_location"):
                    reply_msg = "Theek hai, kripya batayein ki aapki gaadi ki agli trip kab hai aur wo kis location par hogi?"
                    chat_hist.append({"role": "bot", "text": reply_msg})
                    database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                    send_whatsapp_meta(phone, reply_msg)
                    return {"status": "collecting_next_trip_details"}
                else:
                    # Map finalized next trip vectors straight down to operational properties
                    collected["service_date"] = collected["next_trip_date"]
                    collected["vehicle_location"] = collected["next_trip_location"]

        # Step D: Extract active driver phone parameters if missing
        if not collected.get("driver_phone"):
            reply_msg = f"{prefix_reply}Driver ka active mobile number share kijiye taaki technician coordinate kar sake."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_driver_phone"}

        # ======================================================================
        # STEP E: STANDARDIZED SECURE TICKET GENERATION
        # ======================================================================
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        collected["ticket_id"] = ticket_id

        # Structural Payload Builder
        ticket_payload = {
            "vehicle_location": collected.get("vehicle_location"),
            "service_date": collected.get("service_date") or date.today().isoformat(),
            "driver_phone": collected.get("driver_phone") if collected.get("driver_phone") != "NOT_PROVIDED" else phone,
            "engineer_id": "ENG-642",
            "engineer_name": "Ramesh Kumar",
            "engineer_phone": "9876543210",
            "assignment_status": "ASSIGNED"
        }
        
        # Persist to central ledger
        database.save_ticket(ticket_id, phone, ticket_payload)

        # Build output structure matching exact design pattern layout
        reply_msg = (
            f"✅ Service request create kar di gayi hai!\n\n"
            f"📋 *Ticket Details:*\n\n"
            f"🎫 *Ticket ID:* {ticket_id}\n"
            f"📍 *Location:* {ticket_payload['vehicle_location']}\n"
            f"📅 *Service Date:* {ticket_payload['service_date']}\n"
            f"📞 *Contact:* {ticket_payload['driver_phone']}\n\n"
            f"👤 Engineer assignment jald ho jayega.\n"
            f"Engineer aapse jald sampark karega.\n\n"
            f"Koi sawal ho toh Ticket ID {ticket_id} ke saath humse sampark karein.\n\n"
            f"Dhanyavaad!"
        )

        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "TICKET_RAISED", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        
        # Complete session destruction
        database.delete_session(phone)
        return {"status": "ticket_created_successfully", "ticket_id": ticket_id}

    # 6. DYNAMIC OVERRIDE FOR AMBIGUOUS OR CONVERSATIONAL FALLBACK CHAT
    conversational_fallback = brain.get("conversational_reply") or "Kripya vehicle ki sthiti short me spasht karein."
    chat_hist.append({"role": "bot", "text": conversational_fallback})
    database.save_session(phone, current_state, collected, chat_hist)
    send_whatsapp_meta(phone, conversational_fallback)
    return {"status": "fallback_interaction_prompted"}