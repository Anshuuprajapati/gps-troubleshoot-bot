import os
import json
import random
import re
import logging
from typing import Optional, Dict, Any
from datetime import datetime, date, timedelta
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
1. **Entity Extraction**: Extract `current_location` (for VEHICLE_RUNNING from-location), `destination_location` (for VEHICLE_RUNNING to-location), `vehicle_location` (for other intents), `service_city` (preferred service city), `driver_name`, `driver_phone`, `workshop_name`, `resume_date` (when the vehicle runs again), `service_date` (when an engineer fixes it), and `contact_person` (who will coordinate with engineer).

2. **VEHICLE_RUNNING Special Fields**:
   - `current_location`: Where the vehicle is driving from
   - `destination_location`: Where the vehicle is going to
   - `service_city_confirmed`: true if user agrees to Delhi, false if they reject Delhi, null if not asked yet
   - `service_city`: The actual city where they want service (only if Delhi rejected)

3. **Scheduling Loop Tracking**:
   - If a slot suggestion is rejected, look at the last `scheduling_step` value.
   - If the user explicitly rejects the suggested time, increment the `scheduling_step` or mark `step_rejected: true` so the code can advance to the next strategy level (+4 days, +5 days, +7 days, then Next Trip details).
   - If they agree to a location/date or supply one themselves, capture it immediately.

4. **GPS_REMOVED Rule**: If the tracker is self-removed, look out for whether they want a service engineer visit (`wants_service_visit`). If they don't, we will just close the loop on `resume_date`.

## RESPONSE SCHEMA (STRICT - Return ONLY valid JSON):
{
  "intent": "WORKSHOP" | "ACCIDENT" | "VEHICLE_RUNNING" | "VEHICLE_STANDING" | "GPS_DAMAGED" | "GPS_REMOVED" | "OTHER" | null,
  "entities": {
    "current_location": string or null,
    "destination_location": string or null,
    "vehicle_location": string or null,
    "service_city": string or null,
    "driver_phone": string or null,
    "driver_name": string or null,
    "workshop_name": string or null,
    "resume_date": string or null,
    "service_date": string or null,
    "next_trip_date": string or null,
    "next_trip_location": string or null,
    "contact_person": string or null
  },
  "service_city_confirmed": boolean | null,
  "wants_service_visit": boolean | null,
  "is_in_workshop_currently": boolean | null,
  "slot_rejected": boolean,
  "side_question_reply": string or null,
  "conversational_reply": string
}

5. **Driver Verification Rule (VEHICLE_RUNNING)**: If `verifying_driver` is true and the user agrees (e.g., "Haa", "yes", "correct"), set the extracted `contact_person` to the value of `driver_name` and `driver_phone` to the driver's phone. If they provide a completely new name or phone number instead, extract those into `contact_person` and update `driver_phone` accordingly, and mark `contact_person_rejected: true`.

6. **Date Alternative Selection (VEHICLE_RUNNING)**: If the user responds to the 3 scheduling options:
   - If they select option "1" or say "after 2 days", calculate the target date as today + 2 days, and map it into `entities.service_date`.
   - If they select option "2" or say "after 4 days", calculate the target date as today + 4 days, and map it into `entities.service_date`.
   - If they pick option "3" or specify an explicit date/relative timeline (e.g., "5 din baad", "05-07-2026"), extract that information into `entities.service_date`.
   - Ensure `slot_rejected` is marked as false once they choose one of these options so the flow can advance to Step F.
"""

async def create_service_ticket_flow(phone: str, collected: dict, chat_hist: list):
    """Create service ticket and send confirmation"""
    try:
        # Generate ticket ID
        ticket_id = f"TKT-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}"
        
        # Determine service city
        if collected.get("service_city"):
            service_city = collected["service_city"]
        elif collected.get("service_city_confirmed") == True:
            # Use destination if they confirmed it, otherwise fallback
            service_city = collected.get("destination_location") or "Delhi"
        else:
            service_city = "Delhi"  # Default fallback
        
        # Determine service date
        if collected.get("service_date"):
            service_date = collected["service_date"]
        else:
            # Default to tomorrow if user agreed to "kal"
            tomorrow = date.today() + timedelta(days=1)
            service_date = tomorrow.strftime("%d-%m-%Y")
        
        # Create ticket data
        ticket_data = {
            "vehicle_no": collected.get("vehicle_no"),
            "intent": collected.get("intent"),
            "current_location": collected.get("current_location"),
            "destination_location": collected.get("destination_location"),
            "service_city": service_city,
            "service_date": service_date,
            # FALLBACK: If contact_person is empty, use driver_name
            "contact_person": collected.get("contact_person") or collected.get("driver_name") or "Driver",
            "driver_phone": collected.get("driver_phone") if collected.get("driver_phone") != "NOT_PROVIDED" else phone,
            "driver_name": collected.get("driver_name"),
            "status": "OPEN"
        }   
        
        # Save ticket to database
        database.save_ticket(ticket_id, phone, ticket_data)
        collected["ticket_id"] = ticket_id
        
        # Send confirmation message
        reply_msg = (
            f"🎫 *Service Ticket Created Successfully!*\n\n"
            f"• *Ticket ID:* {ticket_id}\n"
            f"• *Vehicle:* {collected.get('vehicle_no')}\n"
            f"• *Issue:* GPS not updating while running\n"
            f"• *Current Location:* {collected.get('current_location')}\n"
            f"• *Destination:* {collected.get('destination_location')}\n"
            f"• *Service City:* {service_city}\n"
            f"• *Service Date:* {service_date}\n"
            f"• *Contact Person:* {collected.get('contact_person', 'Manager')}\n\n"
            f"Hamare engineer aapke contact person se coordinate karenge service ke liye. Dhanyavaad!"
        )
        
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "TICKET_CREATED", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        
        # Clean up session after successful ticket creation
        database.delete_session(phone)
        
        return {"status": "ticket_created", "ticket_id": ticket_id}
        
    except Exception as e:
        logger.error(f"Error creating service ticket: {e}")
        error_msg = "Ticket create karne mein problem aayi. Please try again."
        send_whatsapp_meta(phone, error_msg)
        return {"status": "error", "message": str(e)}


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


def is_affirmative_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(r"\b(haan|ha|yes|y|ok|okay|theek|thik|sahi|bilkul|confirm|correct)\b", cleaned))


def detect_contact_choice(text: str) -> Optional[str]:
    cleaned = text.strip().lower()
    if re.search(r"\b(khud|self|main|hum|owner|mai|me)\b", cleaned):
        return "self"
    if re.search(r"\b(driver|driver se|driver ko|driver ka|unse)\b", cleaned):
        return "driver"
    if re.search(r"\b(kisi aur|koi aur|other|dusra|dusre|contact person|contact person se|manager|supervisor)\b", cleaned):
        return "other"
    return None


def extract_phone_number(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{10,13})\b", text)
    return match.group(1) if match else None


def build_troubleshooting_prompt(current_intent: Optional[str], collected: dict, standing_hours: float) -> str:
    if current_intent == "WORKSHOP":
        return "Workshop/Service center ka naam kya hai?"

    if current_intent == "ACCIDENT":
        return "Kya gaadi abhi kisi workshop ya garage me khadi hai?"

    if current_intent == "VEHICLE_RUNNING":
        if not collected.get("current_location"):
            return "Aapki gaadi abhi kis location par hai? (Current location batayein)"
        if not collected.get("destination_location"):
            return "Kahan ja rahe hain? (Destination batayein)"
        if collected.get("service_city_confirmed") is None and not collected.get("service_city"):
            suggested_city = collected.get("destination_location") or "Delhi"
            return f"Kya hum {suggested_city} mein service book kar dein?"
        if collected.get("service_city_confirmed") == False and not collected.get("service_city"):
            return "Kaun se city mein service chahiye? (Preferred city batayein)"
        if not collected.get("service_date"):
            return "Kya kal service book kar dein?"
        if not collected.get("contact_person") and not collected.get("contact_person_rejected"):
            d_name = collected.get("driver_name")
            d_phone = collected.get("driver_phone")
            if d_name or d_phone:
                d_name_str = d_name or "Not Available"
                d_phone_str = d_phone or "Not Available"
                return (
                    "Humare paas driver ki details available hain:\n\n"
                    f"👤 *Driver Name:* {d_name_str}\n"
                    f"📞 *Driver Contact:* {d_phone_str}\n\n"
                    "Kya hum unse hi coordinate karein? (Haan batayein ya unka alternative number/naam share karein)"
                )
            return "Service coordination ke liye contact person ka naam aur mobile number kya hai?"
        return "Kripya apni next update share kijiye."

    if current_intent == "GPS_REMOVED":
        if not collected.get("resume_date"):
            return "GPS device vehicle me wapas kab tak connect/reinstall ho jayega?"
        if collected.get("wants_service_visit") is None:
            return "Kya aapko reinstall karne ke liye hamare physical service engineer ki zaroorat hai?"
        if collected.get("wants_service_visit") is False:
            return "✅ Sahi hai, aap jab device plug karenge tracking start ho jayegi. Case update log kar diya hai."

    if current_intent == "VEHICLE_STANDING" and standing_hours > 48:
        if not collected.get("resume_date"):
            return "Gaadi 48 ghante se jyada se stationary hai. Agli trip kis date ko nikalne wali hai?"

    if not collected.get("vehicle_location"):
        return "Gaadi abhi kis city/location par chal rahi hai?"

    return "Kripya vehicle ki sthiti short me batayein."


def reassign_troubleshooting_contact(
    original_phone: str,
    target_phone: str,
    collected: dict,
    chat_hist: list,
    current_state: str,
    current_intent: Optional[str],
    standing_hours: float,
    status_note: Optional[str] = None,
):
    collected["original_customer_phone"] = collected.get("original_customer_phone") or original_phone
    collected["active_contact_phone"] = target_phone
    collected["contact_mode"] = "driver" if target_phone == collected.get("driver_phone") else "other"

    intro_message = collected.get("initial_alert_msg")
    if not intro_message:
        for entry in chat_hist:
            if entry.get("role") == "bot" and entry.get("text"):
                intro_message = entry.get("text")
                break

    if original_phone != target_phone:
        original_session = dict(collected)
        original_session["status_only"] = True
        database.save_session(original_phone, "STATUS_ONLY", original_session, list(chat_hist))
        if status_note:
            send_whatsapp_meta(original_phone, status_note)

    new_history = list(chat_hist)
    if original_phone != target_phone and intro_message:
        send_whatsapp_meta(target_phone, intro_message)
    prompt = build_troubleshooting_prompt(current_intent, collected, standing_hours)
    new_history.append({"role": "bot", "text": prompt})
    database.save_session(target_phone, current_state, collected, new_history)
    send_whatsapp_meta(target_phone, prompt)


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
            "current_location": None,  # For VEHICLE_RUNNING: where driving from
            "destination_location": None,  # For VEHICLE_RUNNING: where going to
            "service_city": None,  # Preferred service city
            "service_city_confirmed": None,  # True/False for Delhi preference
            "service_date": None,
            "resume_date": None,
            "next_trip_date": None,
            "next_trip_location": None,
            "driver_phone": gps_data.get("driver_phone"),
            "driver_name": gps_data.get("driver_name"),
            "initial_alert_msg": initial_alert_msg,
            "original_customer_phone": phone_number,
            "active_contact_phone": phone_number,
            "contact_mode": None,
            "status_only": False,
            "workshop_name": None,
            "contact_person": None,
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

    if current_state == "STATUS_ONLY":
        status_contact = collected.get("contact_person") or collected.get("driver_name") or "selected contact"
        status_reply = f"Update: hum {status_contact} ke sath troubleshooting continue kar rahe hain. Aapko sirf status updates milte rahenge."
        chat_hist.append({"role": "user", "text": message})
        chat_hist.append({"role": "bot", "text": status_reply})
        database.save_session(phone, current_state, collected, chat_hist)
        send_whatsapp_meta(phone, status_reply)
        return {"status": "status_only_update"}

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
    if brain.get("service_city_confirmed") is not None:
        collected["service_city_confirmed"] = brain["service_city_confirmed"]

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

    standing_hours = collected.get("standing_hours", 0.0)

    if current_state == "INITIAL_ALERT" and current_intent in {"VEHICLE_RUNNING", "GPS_DAMAGED", "VEHICLE_STANDING", "GPS_REMOVED"}:
        reply_msg = f"{prefix_reply}Kya aap khud GPS issue check karenge, ya hum driver ya kisi aur contact person se baat karein?"
        chat_hist.append({"role": "bot", "text": reply_msg})
        collected["troubleshooting_contact_requested"] = True
        database.save_session(phone, "AWAITING_TROUBLESHOOTING_CONTACT", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "awaiting_troubleshooting_contact"}

    if current_state == "AWAITING_TROUBLESHOOTING_CONTACT":
        contact_choice = detect_contact_choice(message)
        phone_in_message = extract_phone_number(message)

        if phone_in_message and contact_choice is None:
            collected["contact_mode"] = "other"
            new_name = brain.get("entities", {}).get("contact_person") or brain.get("entities", {}).get("driver_name")
            if new_name:
                collected["contact_person"] = new_name
            collected["driver_phone"] = phone_in_message
            collected["active_contact_phone"] = phone_in_message
            reassign_troubleshooting_contact(
                phone,
                phone_in_message,
                collected,
                chat_hist,
                "COLLECTING_DETAILS",
                current_intent,
                standing_hours,
                "Update: aapne alternate contact share kar diya hai. Hum troubleshooting naye contact ke saath continue kar rahe hain.",
            )
            return {"status": "other_contact_handoff_completed"}

        if contact_choice == "self":
            collected["contact_mode"] = "self"
            collected["active_contact_phone"] = phone
            reply_msg = f"{prefix_reply}Theek hai, hum aapke saath hi troubleshooting continue karte hain."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "self_handoff_completed"}

        if contact_choice == "driver":
            collected["contact_mode"] = "driver"
            reply_msg = (
                f"{prefix_reply}Humare paas driver ki details available hain:\n\n"
                f"👤 *Driver Name:* {collected.get('driver_name') or 'Not Available'}\n"
                f"📞 *Driver Contact:* {collected.get('driver_phone') or 'Not Available'}\n\n"
                "Kya aap in details ko confirm karte hain, ya driver ka naya naam/number share karenge?"
            )
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "AWAITING_DRIVER_CONFIRMATION", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "awaiting_driver_confirmation"}

        if contact_choice == "other":
            collected["contact_mode"] = "other"
            reply_msg = f"{prefix_reply}Theek hai. Kripya contact person's name (optional) aur mobile number bhejiye."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "AWAITING_OTHER_CONTACT_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "awaiting_other_contact_details"}

        reply_msg = f"{prefix_reply}Kya aap khud GPS issue check karenge, ya hum driver ya kisi aur contact person se baat karein?"
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "AWAITING_TROUBLESHOOTING_CONTACT", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "awaiting_troubleshooting_contact"}

    if current_state == "AWAITING_DRIVER_CONFIRMATION":
        driver_phone = extract_phone_number(message)
        new_contact_name = brain.get("entities", {}).get("contact_person") or brain.get("entities", {}).get("driver_name")
        if is_affirmative_response(message):
            target_phone = collected.get("driver_phone") or phone
            collected["contact_mode"] = "driver"
            collected["active_contact_phone"] = target_phone
            collected["contact_person"] = collected.get("driver_name") or collected.get("contact_person") or "Driver"
            reassign_troubleshooting_contact(
                phone,
                target_phone,
                collected,
                chat_hist,
                "COLLECTING_DETAILS",
                current_intent,
                standing_hours,
                "Update: customer ne driver confirmation de di hai. Hum troubleshooting driver ke saath continue kar rahe hain.",
            )
            return {"status": "driver_handoff_completed"}

        if driver_phone:
            collected["contact_mode"] = "driver"
            collected["driver_phone"] = driver_phone
            if new_contact_name:
                collected["driver_name"] = new_contact_name
            collected["contact_person"] = new_contact_name or collected.get("driver_name") or "Driver"
            reassign_troubleshooting_contact(
                phone,
                driver_phone,
                collected,
                chat_hist,
                "COLLECTING_DETAILS",
                current_intent,
                standing_hours,
                "Update: driver details update ho gayi hain. Hum troubleshooting naye driver ke saath continue kar rahe hain.",
            )
            return {"status": "driver_handoff_updated"}

        reply_msg = f"{prefix_reply}Please driver ko confirm karein ya naya naam/number bhej dein."
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "AWAITING_DRIVER_CONFIRMATION", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "awaiting_driver_confirmation"}

    if current_state == "AWAITING_OTHER_CONTACT_DETAILS":
        new_phone = extract_phone_number(message) or brain.get("entities", {}).get("driver_phone")
        if not new_phone:
            reply_msg = f"{prefix_reply}Kripya contact person's mobile number bhejiye."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "AWAITING_OTHER_CONTACT_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "awaiting_other_contact_details"}

        new_name = brain.get("entities", {}).get("contact_person") or brain.get("entities", {}).get("driver_name")
        if new_name:
            collected["contact_person"] = new_name
        collected["driver_phone"] = new_phone
        collected["contact_mode"] = "other"
        collected["active_contact_phone"] = new_phone
        reassign_troubleshooting_contact(
            phone,
            new_phone,
            collected,
            chat_hist,
            "COLLECTING_DETAILS",
            current_intent,
            standing_hours,
            "Update: alternate contact add ho gaya hai. Ab troubleshooting naye contact ke saath continue ho rahi hai.",
        )
        return {"status": "other_contact_handoff_completed"}

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

    # 5. VEHICLE RUNNING SPECIFIC FLOW
    if current_intent == "VEHICLE_RUNNING":
        # Step A: Ask current location (from where)
        if not collected.get("current_location"):
            reply_msg = f"{prefix_reply}Aapki gaadi abhi kis location par hai? (Current location batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_current_location"}

        # Step B: Ask destination (where going)
        if not collected.get("destination_location"):
            reply_msg = f"{prefix_reply}Kahan ja rahe hain? (Destination batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_destination"}

        # Step C: Suggest service city (Delhi by default)
        if collected.get("service_city_confirmed") is None and not collected.get("service_city"):
            suggested_city = collected.get("destination_location") or "Delhi"
            
            reply_msg = f"{prefix_reply}Kya hum {suggested_city} mein service book kar dein?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "asking_service_city_preference"}

        # Step D: If Delhi rejected, ask preferred service city
        if collected.get("service_city_confirmed") == False and not collected.get("service_city"):
            reply_msg = f"{prefix_reply}Kaun se city mein service chahiye? (Preferred city batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_preferred_service_city"}

        # Step E: Ask service date
        if not collected.get("service_date"):
            step = collected.get("scheduling_step", 0)
            
            # If the user rejected a previous slot, increment the step tracker
            if brain.get("slot_rejected") is True:
                step += 1
                collected["scheduling_step"] = step
                # Clear slot_rejected flag so it doesn't trigger on consecutive turns blindly
                brain["slot_rejected"] = False 

            # Tier 0: Initial question (Tomorrow)
            if step == 0:
                reply_msg = f"{prefix_reply}Kya kal service book kar dein?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "asking_service_date_tomorrow"}
            
            # Tier 1: User said "Nahi" to tomorrow -> Show the 3 structured options
            else:
                reply_msg = (
                    f"{prefix_reply}Please choose one option from below:\n\n"
                    f"1️⃣ Book service after 2 days\n"
                    f"2️⃣ Book service after 4 days\n"
                    f"3️⃣ Enter a specific date or tell after how many days..."
                )
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_alternative_service_date"}  # <--- THIS RETURN IS CRITICAL

        # Step F: Collect contact person if needed
        if not collected.get("contact_person") and not collected.get("contact_person_rejected"):
            d_name = collected.get("driver_name")
            d_phone = collected.get("driver_phone")

            # Scenario A: We have driver details in the session data
            if d_name or d_phone:
                d_name_str = d_name or "Not Available"
                d_phone_str = d_phone or "Not Available"
                
                reply_msg = (
                    f"{prefix_reply}Humare paas driver ki details available hain:\n\n"
                    f"👤 *Driver Name:* {d_name_str}\n"
                    f"📞 *Driver Contact:* {d_phone_str}\n\n"
                    f"Kya hum unse hi coordinate karein? (Haan batayein ya unka alternative number/naam share karein)"
                )
                chat_hist.append({"role": "bot", "text": reply_msg})
                
                # Set a state tracker so the LLM brain knows we are verifying the driver
                collected["verifying_driver"] = True 
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "verifying_existing_driver"}
                
            # Scenario B: No driver details found at all upstream
            else:
                reply_msg = f"{prefix_reply}Service coordination ke liye contact person ka naam aur mobile number kya hai?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "collecting_contact_person"}
        # All details collected - create ticket
        return await create_service_ticket_flow(phone, collected, chat_hist)

    # 6. CORE FIELD SERVICE SLOT-FILLING ENGINE FOR OTHER INTENTS
    # Triggers for: GPS_DAMAGED, VEHICLE_STANDING (<48h), and GPS_REMOVED (when visit is True)
    elif current_intent in ["GPS_DAMAGED", "VEHICLE_STANDING", "GPS_REMOVED"]:
        
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
                
                # APPEND DRIVER CONFIRMATION IF DETAILS EXIST
                if collected.get("driver_name") or collected.get("driver_phone"):
                    d_name = collected.get("driver_name") or "Not Available"
                    d_phone = collected.get("driver_phone") or "Not Available"
                    reply_msg += (
                        f"\n\nHumare paas driver ki details available hain:\n\n"
                        f"* Driver Name: {d_name}\n"
                        f"* Driver Contact: {d_phone}\n\n"
                        f"Ya koi aur number dena hai ya ise ko save kar le ??"
                    )
                
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
                    collected["service_date"] = collected["next_trip_date"]
                    collected["vehicle_location"] = collected["next_trip_location"]

        # Step D: Extract active driver phone parameters if missing or user wants to update
        # If user said "save this", LLM will keep existing phone. If they provide a new one, it updates.
        if not collected.get("driver_phone") or collected.get("driver_phone") == "NOT_PROVIDED":
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