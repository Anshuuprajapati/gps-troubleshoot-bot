import os
import json
import random
import re
import requests
from typing import Optional
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

import database
from date_utils import normalize_date
from gps_analysis_api import analyze_gps_device   # ← NEW

load_dotenv()

app = FastAPI(title="Meta WhatsApp + Azure OpenAI Troubleshooting Hub")

azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

database.init_db()

# ==========================================
# 1. Pydantic Models for JSON Validation
# ==========================================

class GpsData(BaseModel):
    gpstime: Optional[str] = None
    main_powervoltage: Optional[float] = None
    ismainpoerconnected: Optional[str] = None   # "1" = connected, "0" = disconnected
    gpsStatus: Optional[int] = None             # 0 = no fix, 1 = fix
    driver_name: Optional[str] = None
    driver_phone: Optional[str] = None
    current_location: Optional[str] = None
    vehicle_state: Optional[str] = None


class OutageRequest(BaseModel):
    phone_number: str
    vehicle_no: str
    last_location: str
    timestamp: str
    gps_data: Optional[GpsData] = None

class TicketDetails(BaseModel):
    vehicle_location: Optional[str] = None
    service_date: Optional[str] = None
    driver_phone: Optional[str] = None

# ==========================================
# 2. CORE HELPERS
# ==========================================

def generate_ticket_id() -> str:
    """Generates a unique ticket ID."""
    return f"TKT-{random.randint(1000, 9999)}"


ENGINEERS = [
    {
        "engineer_id": "ENG-001",
        "name": "Rajesh Kumar",
        "phone": "9876543210",
        "city": "Jaipur",
        "status": "available"
    },
    {
        "engineer_id": "ENG-002",
        "name": "Amit Sharma",
        "phone": "9876543211",
        "city": "Delhi",
        "status": "available"
    },
]


def normalize_engineer_city(city: str) -> str:
    if not city:
        return ""
    return " ".join(word.capitalize() for word in re.split(r"\s+", city.strip().lower()))


def find_nearest_engineer(service_location: str) -> dict | None:
    if not service_location:
        return None
    normalized_location = normalize_engineer_city(service_location)
    same_city = [e for e in ENGINEERS if e["status"] == "available" and normalize_engineer_city(e["city"]) == normalized_location]
    if same_city:
        return same_city[0]
    available = [e for e in ENGINEERS if e["status"] == "available"]
    return available[0] if available else None


def assign_engineer_to_ticket(ticket_data: dict) -> dict:
    print(f"[ENGINEER_ASSIGNMENT] Assigning engineer for location {ticket_data.get('vehicle_location')}")
    engineer = find_nearest_engineer(ticket_data.get("vehicle_location"))
    if engineer:
        ticket_data["engineer_id"] = engineer["engineer_id"]
        ticket_data["engineer_name"] = engineer["name"]
        ticket_data["engineer_phone"] = engineer["phone"]
        ticket_data["assignment_status"] = "assigned"
        engineer["status"] = "assigned"
        print(f"[ENGINEER_FOUND] {engineer['engineer_id']} {engineer['name']} assigned")
    else:
        ticket_data["engineer_id"] = None
        ticket_data["engineer_name"] = None
        ticket_data["engineer_phone"] = None
        ticket_data["assignment_status"] = "pending"
        print("[ENGINEER_NOT_FOUND] No available engineer")
    return ticket_data


def build_ticket_message(ticket_id: str, data: dict) -> str:
    """Builds the final ticket creation confirmation message."""
    ticket_msg = (
        f"✅ Service request create kar di gayi hai!\n\n"
        f"📋 Ticket Details:\n\n"
        f"🎫 Ticket ID: {ticket_id}\n"
        f"📍 Location: {data.get('vehicle_location', 'N/A')}\n"
        f"📅 Service Date: {data.get('service_date', 'N/A')}\n"
        f"📞 Contact: {data.get('driver_phone', 'N/A')}\n\n"
    )
    if data.get("engineer_name"):
        ticket_msg += (
            f"👨‍🔧 Assigned Engineer: {data.get('engineer_name')}\n"
            f"📱 Engineer Contact: {data.get('engineer_phone')}\n\n"
            f"Engineer aapse jald sampark karega.\n\n"
            f"Koi sawal ho toh Ticket ID {ticket_id} ke saath humse sampark karein.\n\n"
            f"Dhanyavaad!"
        )
    else:
        ticket_msg += (
            f"👤 Engineer assignment jald ho jayega.\n\n"
            f"Engineer aapse jald sampark karega.\n\n"
            f"Koi sawal ho toh Ticket ID {ticket_id} ke saath humse sampark karein.\n\n"
            f"Dhanyavaad!"
        )
    return ticket_msg


def is_case_closed_intent(intent: str) -> bool:
    """Returns True for intents that close the case without a ticket."""
    return intent in ["WORKSHOP", "ACCIDENT", "BATTERY_DISCONNECT", "GPS_REMOVED",
                      "BATTERY_ISSUE", "MAIN_POWER_CUT"]          # ← NEW root causes


def is_ticket_required_intent(intent: str) -> bool:
    """Returns True for intents that require a service ticket."""
    return intent in ["GPS_DAMAGED", "VEHICLE_RUNNING_GPS_NOT_UPDATING", "VEHICLE_STANDING", "OTHER"]


def normalize_whatsapp_number(raw: str) -> str:
    """
    Normalise a phone number to E.164 format (digits only, with country code).
    - Strips spaces, dashes, parentheses.
    - If the number already starts with '91' and is 12 digits → keep as-is.
    - If it's a 10-digit Indian number → prepend '91'.
    - Otherwise → return stripped digits as-is.
    """
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return "91" + digits
    if len(digits) == 12 and digits.startswith("91"):
        return digits
    return digits


def forward_alert_to_driver(driver_number: str, original_session: dict) -> None:
    """
    Initialise a fresh session for the driver and send them the same
    initial alert that was sent to the original customer, so the GPS
    resolution flow continues with the driver directly.

    If the driver already has an active session we skip re-initialisation
    to avoid overwriting an in-progress conversation.
    """
    existing = database.get_session(driver_number)

    # Don't overwrite an already-active driver session
    if existing["current_state"] not in [None, "INITIAL_ALERT"]:
        print(f"[DRIVER FWD] Driver {driver_number} already has active session "
              f"({existing['current_state']}). Skipping.")
        return

    # Build the initial alert from the original session's first bot message
    original_history = original_session.get("chat_history", [])
    initial_alert_msg = next(
        (m["text"] for m in original_history if m["role"] == "bot"),
        None,
    )

    if not initial_alert_msg:
        print(f"[DRIVER FWD] Could not find original alert message in session history. Skipping.")
        return

    database.save_session(
        phone_number=driver_number,
        current_state="INITIAL_ALERT",
        collected_json={
            "intent": None,
            "vehicle_location": None,
            "service_date": None,
            "arrival_date": None,
            "driver_phone": None,
            "driver_name": None,
            "contact_person": None,
            "origin_city": None,
            "destination_city": None,
            "resume_date": None,
            "ticket_id": None,
            "forwarded_from": original_session.get("phone_number"),  # audit trail
            # Pre-analysis fields default to unknown for forwarded sessions
            "battery_issue": False,
            "main_power_issue": False,
            "root_cause": "OTHER_ISSUE",
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}],
    )

    send_whatsapp_meta(driver_number, initial_alert_msg)
    print(f"[DRIVER FWD] Alert forwarded to driver {driver_number}")


# ==========================================
# 3. INITIAL ALERT MESSAGE BUILDERS          ← NEW SECTION
# ==========================================

def build_battery_issue_alert(vehicle_no: str, gps_data: dict) -> str:
    """Initial message for BATTERY_ISSUE root cause."""
    gpstime       = gps_data.get("gpstime", "N/A")
    last_location = gps_data.get("current_location") or gps_data.get("last_location", "N/A")
    vehicle_state = gps_data.get("vehicle_state", "N/A")
    return (
        f"Hello,\n\n"
        f"We have analyzed the GPS status for vehicle {vehicle_no}.\n"
        f"Our system indicates a possible battery-related issue affecting the GPS device.\n\n"
        f"Last GPS Update: {gpstime}\n"
        f"Last Known Location: {last_location}\n"
        f"Current Vehicle Status: {vehicle_state}\n\n"
        f"Could you please confirm whether the vehicle battery was recently disconnected, "
        f"replaced, serviced, or if the battery is currently discharged?\n\n"
        f"Please let us know so we can assist you further."
    )


def build_main_power_cut_alert(vehicle_no: str, gps_data: dict) -> str:
    """Initial message for MAIN_POWER_CUT root cause."""
    gpstime       = gps_data.get("gpstime", "N/A")
    last_location = gps_data.get("current_location") or gps_data.get("last_location", "N/A")
    vehicle_state = gps_data.get("vehicle_state", "N/A")
    return (
        f"Hello,\n\n"
        f"We have analyzed the GPS status for vehicle {vehicle_no}.\n"
        f"Our system indicates that the GPS device may not be receiving main power from the vehicle.\n\n"
        f"Last GPS Update: {gpstime}\n"
        f"Last Known Location: {last_location}\n"
        f"Current Vehicle Status: {vehicle_state}\n\n"
        f"Could you please confirm whether there has been any recent electrical work, "
        f"wiring issue, fuse issue, or power disconnection in the vehicle?\n\n"
        f"Please let us know so we can assist you further."
    )


def build_other_issue_alert(vehicle_no: str, gps_data: dict) -> str:
    """Initial message for OTHER_ISSUE root cause."""
    gpstime       = gps_data.get("gpstime", "N/A")
    driver_name   = gps_data.get("driver_name", "N/A")
    driver_phone  = gps_data.get("driver_phone", "N/A")
    cur_location  = gps_data.get("current_location", "N/A")
    vehicle_state = gps_data.get("vehicle_state", "N/A")
    return (
        f"Hello,\n\n"
        f"We have analyzed the GPS status for vehicle {vehicle_no}.\n"
        f"No battery-related issue or main power issue has been detected from our side.\n\n"
        f"Vehicle Details:\n"
        f"• Driver Name: {driver_name}\n"
        f"• Driver Contact: {driver_phone}\n"
        f"• Current Location: {cur_location}\n"
        f"• Vehicle Status: {vehicle_state}\n"
        f"• Last GPS Update: {gpstime}\n\n"
        f"Could you please describe the exact GPS-related issue you are facing?\n\n"
        f"For example:\n"
        f"• GPS location not updating\n"
        f"• Vehicle status not showing\n"
        f"• GPS device damaged\n"
        f"• GPS device removed\n"
        f"• Any other tracking issue\n\n"
        f"Please provide details so we can assist you further."
    )


def build_initial_alert(vehicle_no: str, gps_data: dict, root_cause: str) -> str:
    """Route to the correct initial message based on root cause."""
    if root_cause == "BATTERY_ISSUE":
        return build_battery_issue_alert(vehicle_no, gps_data)
    if root_cause == "MAIN_POWER_CUT":
        return build_main_power_cut_alert(vehicle_no, gps_data)
    return build_other_issue_alert(vehicle_no, gps_data)


# ==========================================
# 4. LLM SYSTEM INSTRUCTION
# ==========================================

SYSTEM_INSTRUCTION = """
You are an automated AI-powered GPS Troubleshooting Bot for vehicle GPS downtime resolution.

## YOUR TASK
Process the customer's message(s) and return a single valid JSON object.

## RESPONSE SCHEMA (STRICT - return ONLY this JSON, no markdown, no extra text):
{
    "extracted_data": {
        "intent": string or null,
        "vehicle_location": string or null,
        "service_date": string or null,
        "arrival_date": string or null,
        "driver_phone": string or null,
        "driver_name": string or null,
        "contact_person": string or null,
        "origin_city": string or null,
        "destination_city": string or null,
        "resume_date": string or null
    },
    "next_state": "INITIAL_ALERT" | "CASE_CLOSED" | "COLLECTING_DETAILS" | "TICKET_RAISED",
    "reply_to_user": string,
    "debug_notes": string
}

## PRE-ANALYSIS ROOT CAUSE CONTEXT
The session will include a "root_cause" field set by the GPS analysis API BEFORE the first message:
  - "BATTERY_ISSUE"   — battery disconnect or battery discharge detected
  - "MAIN_POWER_CUT"  — main/external power supply cut detected
  - "OTHER_ISSUE"     — neither battery nor main power; user must describe the problem

## INTENT MAPPING FOR BATTERY_ISSUE ROOT CAUSE
When root_cause = "BATTERY_ISSUE" and user replies to the battery troubleshooting prompt:
- "1" or "haan" or "battery issue" or "disconnect" or "charge kar rahe" → intent = "BATTERY_ISSUE" → CASE_CLOSED
- "2" or "battery theek hai" or "koi aur" → override to "OTHER_ISSUE" flow, ask standard menu

## INTENT MAPPING FOR MAIN_POWER_CUT ROOT CAUSE
When root_cause = "MAIN_POWER_CUT" and user replies to the power troubleshooting prompt:
- "1" or "haan" or "power issue" or "fix kar rahe" → intent = "MAIN_POWER_CUT" → CASE_CLOSED
- "2" or "power theek hai" or "koi aur" → override to "OTHER_ISSUE" flow, ask standard menu

## INTENT MAPPING FOR OTHER_ISSUE ROOT CAUSE (standard 8-option menu)
When user replies to the standard GPS issue menu (option number or description), map to intent:
- "1" or "workshop" or "service center" → intent = "WORKSHOP"
- "2" or "accident" → intent = "ACCIDENT"
- "3" or "battery" or "battery disconnect" → intent = "BATTERY_DISCONNECT"
- "4" or "gps removed" or "nikal diya" → intent = "GPS_REMOVED"
- "5" or "gps damaged" or "toot gaya" → intent = "GPS_DAMAGED"
- "6" or "running" or "chal rahi" or "gadi chal rahi" → intent = "VEHICLE_RUNNING_GPS_NOT_UPDATING"
- "7" or "standing" or "khadi hai" → intent = "VEHICLE_STANDING"
- "8" or "other" or anything else unclear → intent = "OTHER"
Natural language like "gadi running me hai" or "GPS update nahi ho raha chal rahi hai" → VEHICLE_RUNNING_GPS_NOT_UPDATING.

## CASE CLOSED FLOW (intents: WORKSHOP, ACCIDENT, BATTERY_DISCONNECT, GPS_REMOVED, BATTERY_ISSUE, MAIN_POWER_CUT)
- If resume_date is NOT known: set next_state = "COLLECTING_DETAILS", ask ONLY "Vehicle dobara kab running condition mein aa jayegi?"
- If resume_date IS known: set next_state = "CASE_CLOSED", reply = "✅ Update note kar liya gaya hai. Dhanyavaad."
- SPECIAL: for BATTERY_ISSUE and MAIN_POWER_CUT — if user confirms the issue is being fixed,
  ask "Vehicle GPS dobara kab normal ho jayega?" as the resume question, then CASE_CLOSED.
- Do NOT create a ticket. Do NOT ask anything else.
- RESUME DATE EXTRACTION: "parso tak theek ho jayegi" → resume_date = "parso". "kal aa jayegi" → resume_date = "kal". "3 din mein" → resume_date = "3 din mein". ANY time reference given in response to "kab running hogi" question → THAT IS resume_date, NOT arrival_date or service_date.

## TICKET REQUIRED FLOW (intents: GPS_DAMAGED, VEHICLE_RUNNING_GPS_NOT_UPDATING, VEHICLE_STANDING, OTHER)
- Required fields: vehicle_location, service_date, driver_phone
- Extract ALL possible information from EVERY message before deciding what to ask
- Check the "Currently Extracted Data" JSON — if a field is already non-null there, treat it as KNOWN
- NEVER ask for a field that is already known
- Ask for EXACTLY ONE missing field per reply, in polite Hinglish
- Priority order to ask: vehicle_location → service_date → driver_phone
- When ALL 3 fields are filled: set next_state = "TICKET_RAISED"

## LOCATION UPDATE RULE
- If user CORRECTS a previously given location (e.g. "Nahi Mumbai nahi, Pune me hai", "Actually Nagpur me hai"), extract the NEW corrected city as vehicle_location. The latest location always wins.

## PHONE REFUSAL HANDLING
- If user refuses to give phone number ("nahi dena", "baad me", "no number"), set driver_phone = "NOT_PROVIDED" and set next_state = "TICKET_RAISED" with whatever data is available. Do NOT keep asking.

## SHORTHAND / TYPO UNDERSTANDING
- Understand abbreviations and typos: "gps dmg" = GPS damaged, "vhcl" = vehicle, "srvce" = service, "tmrw" = tomorrow, "cntct" = contact, "h" = hai, "m" = mein, "n" = nahi
- Treat emoji-only replies as unclear → ask for the missing field again politely

## ROUTE & LOCATION EXTRACTION (CRITICAL)
The vehicle service location is always where it CURRENTLY IS or where it IS GOING — the DESTINATION.
- "Pune se Delhi ja rahi hai" → origin_city=Pune, destination_city=Delhi, vehicle_location=Delhi
- "Delhi se Bangalore pahuchegi 25 ko" → origin_city=Delhi, destination_city=Bangalore, vehicle_location=Bangalore
- "Mumbai mein hai" → vehicle_location=Mumbai
- "X se Y ja rahi hai" pattern: vehicle_location = Y (destination), NEVER X (origin)
- If only one city mentioned with no route context → vehicle_location = that city

## DATE EXTRACTION (CRITICAL)
Extract ALL date mentions and map them to the correct field:
- "pahuchegi 25 ko" = arrival_date = "25" (vehicle will arrive on 25th)
- "26 ko service chahiye" = service_date = "26"
- "kal service" = service_date = "kal"
- "parso tak aa jayegi" = resume_date = "parso"
- When TWO dates appear: the earlier one is typically arrival_date, the later is service_date
- Output raw date tokens as-is ("25", "kal", "26 June") — Python post-processing will normalize them to full dates
- NEVER output vague strings like "25th of current month" — output just "25"

## PHONE NUMBER EXTRACTION (CRITICAL)
- Any 10-digit number in the message = driver_phone. Extract immediately.
- "8882374849 par contact kar lena" → driver_phone = "8882374849"
- "driver ka number 9876543210 hai" → driver_phone = "9876543210"
- Once extracted, NEVER ask for phone again

## FULL MESSAGE EXTRACTION EXAMPLE
Customer: "Gadi running me hai, Delhi se Bangalore pahuchegi 25 ko, 26 ko service chahiye, 8882374849 par contact kar lena."
Correct extraction:
  intent = "VEHICLE_RUNNING_GPS_NOT_UPDATING"
  origin_city = "Delhi"
  destination_city = "Bangalore"
  vehicle_location = "Bangalore"
  arrival_date = "25"
  service_date = "26"
  driver_phone = "8882374849"
Since all 3 required fields (vehicle_location, service_date, driver_phone) are present → next_state = "TICKET_RAISED"

## INFORMATION MERGE RULES
- NEVER overwrite already-extracted non-null fields with null
- ALWAYS merge new info with existing session data
- Information provided across multiple messages must all be preserved

## INTENT LOCKING
- Once intent is set in session, NEVER change it regardless of what user says next
- NEVER switch flows mid-conversation

## SIDE QUESTION HANDLING
- If user asks something off-topic, give a 1-line answer, then immediately ask for the next missing field

## LANGUAGE
- Respond in casual, respectful Hinglish (Hindi + English mix)
- Keep replies short (1-2 lines max), human-like, never robotic
- Use "aap" for respect

## GUARDRAILS
- Return raw JSON ONLY. No markdown. No code fences. No explanation outside JSON.
- "debug_notes": briefly explain what you extracted and why (used for server-side logging only)
"""


def build_execution_context(session: dict, user_input: str) -> str:
    """Builds the full context string passed to Azure OpenAI."""
    from datetime import date
    today_str = date.today().strftime("%d %B %Y")  # e.g. "19 June 2026"
    recent_history = session["chat_history"][-10:]  # last 10 messages for context window efficiency
    collected = session["collected_json"]
    return (
        f"TODAY'S DATE: {today_str}\n"
        f"Current Bot State: {session['current_state']}\n"
        f"Root Cause (from GPS analysis API): {collected.get('root_cause', 'OTHER_ISSUE')}\n"
        f"Battery Issue Detected: {collected.get('battery_issue', False)}\n"
        f"Main Power Issue Detected: {collected.get('main_power_issue', False)}\n"
        f"Intent Locked: {collected.get('intent', 'NOT SET YET')}\n"
        f"Currently Extracted Data: {json.dumps(collected)}\n\n"
        f"Recent Conversation:\n"
        + "\n".join([f"{m['role'].upper()}: {m['text']}" for m in recent_history])
        + f"\nUSER: {user_input}"
    )


# ==========================================
# 5. META WHATSAPP OUTBOUND UTILITY
# ==========================================

def send_whatsapp_meta(to_number: str, text_body: str):
    """Sends a message directly via Meta's Graph API Cloud Gateway."""
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
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"[META API ERROR] Status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[META SEND EXCEPTION] {e}")


# ==========================================
# 6. CORE MESSAGE PROCESSING LOGIC
# ==========================================

def merge_extracted_data(existing: dict, new_data: dict) -> dict:
    """
    Smart merge: never overwrite an existing non-null value with null.
    New non-null values always win (updates allowed).
    """
    merged = dict(existing)
    for key, value in new_data.items():
        if value is not None:
            merged[key] = value
        elif key not in merged:
            merged[key] = None
    return merged


def is_affirmative_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(r"\b(haan|han|yes|jee|ji|theek|thik|bilkul|sure|please do|kar do|kar du|ekdam)\b", cleaned))


def is_negative_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(r"\b(nahi|na|abhi nahi|kal possible nahi|nahi possible|not now|no)\b", cleaned))


def is_phone_refusal_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(
        r"\b(nahi dena|na dena|baad me|baad mein|no number|no phone|phone nahi|number nahi|provide phone nahi)\b",
        cleaned
    ))


def is_conversation_completed_response(text: str) -> bool:
    cleaned = text.strip().lower()
    return bool(re.search(r"\b(ok|thanks|thank you|noted|done|haan|yes|thik hai|theek hai|okey|okay)\b", cleaned))


def add_days(days: int) -> str:
    from datetime import date, timedelta
    return (date.today() + timedelta(days=days)).isoformat()


def enrich_phone_contact(extracted: dict, user_input: str) -> dict:
    if not extracted.get("driver_phone"):
        match = re.search(r"\b(\d{10})\b", user_input)
        if match:
            extracted["driver_phone"] = match.group(1)
    if extracted.get("driver_phone") and not extracted.get("contact_person"):
        if re.search(r"\bdriver\b", user_input, re.IGNORECASE):
            extracted["contact_person"] = "driver"
    return extracted


def user_mentions_location(user_input: str) -> bool:
    if not user_input:
        return False
    city_names = [
        "delhi", "jaipur", "ahmedabad", "mumbai", "pune", "bangalore", "hyderabad",
        "chennai", "kolkata", "lucknow", "nagpur", "bhopal", "indore", "noida",
        "gurgaon", "gurugram", "vadodara", "patna", "jaipur", "agra", "varanasi"
    ]
    if re.search(r"\b(se|mein|me|pahunch|pahuch|ja rahi|ja raha|ja rahe|jaane|gayi|gaya|jaye|jaye|aayi|aaya|hoga|hogi)\b", user_input, re.IGNORECASE):
        return True
    cities_regex = r"\b(" + "|".join(re.escape(city) for city in city_names) + r")\b"
    return bool(re.search(cities_regex, user_input, re.IGNORECASE))


def all_ticket_fields_present(collected: dict) -> bool:
    service_location = collected.get("vehicle_location")
    service_date = collected.get("service_date")
    driver_phone = collected.get("driver_phone")
    return bool(service_location and service_date and driver_phone and len(str(driver_phone)) > 0)


ROUTE_SERVICE_CENTER_ORDER = [
    "Delhi",
    "Jaipur",
    "Ahmedabad",
    "Mumbai",
    "Pune",
    "Bangalore",
    "Hyderabad",
    "Chennai",
    "Kolkata",
    "Lucknow",
    "Nagpur",
    "Bhopal",
]

ROUTE_SERVICE_CENTER_OVERRIDES = {
    ("Delhi", "Pune"): "Jaipur",
    ("Pune", "Delhi"): "Jaipur",
    ("Jaipur", "Mumbai"): "Jaipur",
    ("Mumbai", "Jaipur"): "Jaipur",
    ("Ahmedabad", "Delhi"): "Jaipur",
    ("Delhi", "Ahmedabad"): "Jaipur",
}


def normalize_city_name(city: str) -> str:
    if not city:
        return ""
    return " ".join(word.capitalize() for word in re.split(r"\s+", city.strip().lower()))


def choose_route_service_center(origin: str, destination: str) -> str | None:
    origin_norm = normalize_city_name(origin)
    destination_norm = normalize_city_name(destination)
    if not origin_norm or not destination_norm or origin_norm == destination_norm:
        return None

    override = ROUTE_SERVICE_CENTER_OVERRIDES.get((origin_norm, destination_norm))
    if override:
        return override

    try:
        origin_idx = ROUTE_SERVICE_CENTER_ORDER.index(origin_norm)
        dest_idx = ROUTE_SERVICE_CENTER_ORDER.index(destination_norm)
    except ValueError:
        return None

    if origin_idx < dest_idx:
        route_segment = ROUTE_SERVICE_CENTER_ORDER[origin_idx + 1:dest_idx]
    else:
        route_segment = list(reversed(ROUTE_SERVICE_CENTER_ORDER[dest_idx:origin_idx]))

    # Exclude current location and destination; suggest the next service center on route.
    route_segment = [city for city in route_segment if city not in {origin_norm, destination_norm}]
    return route_segment[0] if route_segment else None


def route_service_suggestion_applicable(collected: dict) -> bool:
    if not is_ticket_required_intent(collected.get("intent", "")):
        return False
    if collected.get("service_date"):
        return False
    origin = collected.get("origin_city")
    destination = collected.get("destination_city")
    if not origin or not destination or normalize_city_name(origin) == normalize_city_name(destination):
        return False
    suggestion = choose_route_service_center(origin, destination)
    return bool(suggestion)


def advance_service_booking_stage(collected: dict, user_input: str) -> dict:
    """Advance staged service booking flow for ticket-required intents."""
    if not is_ticket_required_intent(collected.get("intent", "")):
        return collected

    if not collected.get("vehicle_location"):
        return collected

    stage = collected.get("service_booking_stage")
    service_date_exists = bool(collected.get("service_date"))
    phone_exists = bool(collected.get("driver_phone")) and len(str(collected.get("driver_phone"))) > 0
    ticket_ready = all_ticket_fields_present(collected)

    if ticket_ready:
        collected["service_booking_stage"] = "COMPLETED"
        return collected

    if stage is None and route_service_suggestion_applicable(collected):
        suggested_center = choose_route_service_center(collected["origin_city"], collected["destination_city"])
        if suggested_center:
            collected["suggested_route_center"] = suggested_center
            collected["service_booking_stage"] = "ASK_ROUTE_SERVICE_CENTER"
            return collected

    if service_date_exists and stage is None:
        collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
        return collected

    if stage is None:
        collected["service_booking_stage"] = "ASK_TOMORROW"
        return collected

    if stage == "ASK_ROUTE_SERVICE_CENTER":
        if is_affirmative_response(user_input):
            suggested = collected.get("suggested_route_center")
            if suggested:
                collected["vehicle_location"] = suggested
            collected["service_booking_stage"] = "ASK_TOMORROW"
            return collected
        if is_negative_response(user_input):
            collected["service_booking_stage"] = "ASK_TOMORROW"
            return collected
        if collected.get("vehicle_location") and collected.get("vehicle_location") != collected.get("suggested_route_center"):
            collected["service_booking_stage"] = "ASK_TOMORROW"
            return collected

    if stage == "ASK_TOMORROW":
        if service_date_exists or is_affirmative_response(user_input):
            if not service_date_exists:
                collected["service_date"] = add_days(1)
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected
        if is_negative_response(user_input):
            collected["service_booking_stage"] = "ASK_4_DAYS"
            return collected

    if stage == "ASK_4_DAYS":
        if service_date_exists or is_affirmative_response(user_input):
            if not service_date_exists:
                collected["service_date"] = add_days(4)
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected
        if is_negative_response(user_input):
            collected["service_booking_stage"] = "ASK_7_DAYS"
            return collected

    if stage == "ASK_7_DAYS":
        if service_date_exists or is_affirmative_response(user_input):
            if not service_date_exists:
                collected["service_date"] = add_days(7)
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected
        if is_negative_response(user_input):
            collected["service_booking_stage"] = "ASK_NEXT_TRIP"
            return collected

    if stage == "ASK_NEXT_TRIP":
        if collected.get("destination_city"):
            collected["service_booking_stage"] = "ASK_ARRIVAL"
            return collected
        if collected.get("arrival_date"):
            if not collected.get("service_date"):
                collected["service_date"] = collected["arrival_date"]
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected
        if service_date_exists:
            collected["service_booking_stage"] = "ASK_DESTINATION"
            return collected

    if stage == "ASK_DESTINATION":
        if collected.get("destination_city"):
            collected["service_booking_stage"] = "ASK_ARRIVAL"
            return collected

    if stage == "ASK_ARRIVAL":
        if collected.get("arrival_date"):
            if not collected.get("service_date"):
                collected["service_date"] = collected["arrival_date"]
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected
        if collected.get("service_date"):
            collected["service_booking_stage"] = "ASK_SERVICE_CENTER_CONFIRMATION"
            return collected

    if stage == "ASK_SERVICE_CENTER_CONFIRMATION":
        if is_affirmative_response(user_input):
            collected["service_booking_stage"] = "COMPLETED" if phone_exists else "ASK_PHONE"
            return collected
        if is_negative_response(user_input):
            collected["service_booking_stage"] = "ASK_SERVICE_LOCATION_FALLBACK"
            return collected
        if service_date_exists and phone_exists:
            collected["service_booking_stage"] = "COMPLETED"
            return collected

    if stage == "ASK_SERVICE_LOCATION_FALLBACK":
        if collected.get("vehicle_location") and service_date_exists:
            collected["service_booking_stage"] = "ASK_PHONE" if not phone_exists else "COMPLETED"
            return collected

    if stage == "ASK_PHONE":
        if phone_exists:
            collected["service_booking_stage"] = "COMPLETED"
            return collected

    return collected


def post_process_extracted(data: dict) -> dict:
    """
    Deterministic post-processing applied AFTER LLM extraction.
    Handles two concerns that must not be left to LLM probability:
      1. Date normalization  — raw tokens → ISO dates
      2. Route resolution    — destination_city always wins as vehicle_location
    """
    from datetime import date
    today = date.today()

    # ── 1. Route resolution: destination beats origin for vehicle_location ─────
    dest = data.get("destination_city")
    if dest:
        data["vehicle_location"] = dest
        print(f"[ROUTE FIX] vehicle_location set to destination: {dest}")

    # ── 2. Date normalization ─────────────────────────────────────────────────
    for date_field in ("service_date", "arrival_date", "resume_date"):
        raw = data.get(date_field)
        if raw:
            normalized = normalize_date(raw, today)
            if normalized != raw:
                print(f"[DATE FIX] {date_field}: '{raw}' → '{normalized}'")
            data[date_field] = normalized

    return data


def infer_route_fields(extracted: dict, user_input: str) -> dict:
    """Infer missing route fields from the latest user message."""
    if extracted is None:
        extracted = {}

    text = user_input.lower()

    route_indicators = [r"\bse\b.*\bja(?:\s+rahi)?\b", r"\bse\b.*\bpahuch", r"\bse\b.*\bpahunch", r"\bse\b.*\bpahuche"]
    if not extracted.get("destination_city") and extracted.get("vehicle_location"):
        if any(re.search(pattern, text) for pattern in route_indicators):
            extracted["destination_city"] = extracted["vehicle_location"]
            print(f"[ROUTE INFER] destination_city inferred from vehicle_location: {extracted['vehicle_location']}")

    if not extracted.get("destination_city"):
        match = re.search(
            r"\b([A-Za-z]+)\s+se\s+([A-Za-z]+)\s+(?:ja(?:\s+rahi)?|pahuch(?:e|i|a|egi)?|pahunch(?:e|i|a)?)\b",
            text
        )
        if match:
            origin, dest = match.group(1).capitalize(), match.group(2).capitalize()
            extracted.setdefault("origin_city", origin)
            extracted["destination_city"] = dest
            print(f"[ROUTE INFER] origin_city={origin}, destination_city={dest}")
            if not extracted.get("vehicle_location"):
                extracted["vehicle_location"] = dest

    return extracted


def build_collection_reply(collected: dict) -> str:
    if not collected.get("vehicle_location"):
        return "Vehicle ka location kya hai? Kahan par hai aapki vehicle?"

    stage = collected.get("service_booking_stage")
    if stage == "ASK_ROUTE_SERVICE_CENTER":
        suggested = collected.get("suggested_route_center")
        if suggested:
            return f"Hamara {suggested} service center route mein aata hai. Kya main {suggested} mein service schedule kar du?"
        return "Kya main service center location suggest karun?"
    if stage == "ASK_TOMORROW":
        return "Kya main service kal book kar du?"
    if stage == "ASK_4_DAYS":
        return "4 din baad service book kar du?"
    if stage == "ASK_7_DAYS":
        return "7 din baad service book kar du?"
    if stage == "ASK_NEXT_TRIP":
        return "Vehicle agli trip par kab jayegi?"
    if stage == "ASK_DESTINATION":
        return "Kya main jaan sakta hu vehicle kahan ja rahi hai?"
    if stage == "ASK_ARRIVAL":
        return "Vehicle wahan kab tak pahunch jayegi?"
    if stage == "ASK_SERVICE_CENTER_CONFIRMATION":
        return "Kya main nearest service center par visit book kar du?"
    if stage == "ASK_SERVICE_LOCATION_FALLBACK":
        return "Aapki preferred service location kya hai?"
    if stage == "ASK_PHONE" or not collected.get("driver_phone"):
        return "Driver ka contact number bata dijiye."

    return "Kripya thoda aur detail dein."


def pre_process_user_input(user_input: str) -> str:
    """
    Pre-process user input BEFORE sending to LLM.
    Resolves weekday phrases and relative date references to concrete ISO dates
    so the LLM cannot miscalculate them.
    """
    import re
    from datetime import date, timedelta
    today = date.today()

    _WEEKDAY_MAP = {
        "monday": 0, "mon": 0, "somwar": 0, "somvaar": 0,
        "tuesday": 1, "tue": 1, "tues": 1, "mangalwar": 1,
        "wednesday": 2, "wed": 2, "budhwar": 2,
        "thursday": 3, "thu": 3, "thurs": 3, "guruwar": 3, "veervar": 3,
        "friday": 4, "fri": 4, "shukrawar": 4, "shukravar": 4,
        "saturday": 5, "sat": 5, "shaniwar": 5,
        "sunday": 6, "sun": 6, "raviwar": 6, "itwaar": 6,
    }

    def next_weekday(wd: int) -> date:
        days_ahead = (wd - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    pattern = re.compile(
        r"\b(next|agle|agli)\s+(" + "|".join(_WEEKDAY_MAP.keys()) + r")\b",
        re.IGNORECASE
    )

    def replace_weekday(m):
        day_word = m.group(2).lower()
        wd = _WEEKDAY_MAP.get(day_word)
        if wd is not None:
            resolved = next_weekday(wd).isoformat()
            print(f"[PRE-PROCESS] '{m.group(0)}' → '{resolved}'")
            return f"{resolved}"
        return m.group(0)

    processed = pattern.sub(replace_weekday, user_input)
    return processed


def process_message(session: dict, user_input: str) -> tuple[str, str, dict]:
    """
    Core processing pipeline. Returns (reply_text, next_state, updated_collected_json).
    """
    current_state = session["current_state"]
    collected = session["collected_json"]

    # ── Duplicate / empty message guard ──────────────────────────────────────
    if not user_input or user_input.strip() == "":
        return "Kripya apna message dobara bhejein.", current_state, collected

    # ── If the ticket flow is already completed, keep the conversation closed for acknowledgements.
    if collected.get("conversation_completed") and is_conversation_completed_response(user_input):
        return "Dhanyavaad 😊 Hamari team aapse sampark karegi.", current_state, collected

    # ── Pre-process: resolve weekday phrases before LLM sees them ─────────────
    user_input = pre_process_user_input(user_input)

    # ── Build context and call Azure OpenAI ──────────────────────────────────
    execution_context = build_execution_context(session, user_input)

    raw_content = ""
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": execution_context}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
        parsed_output = json.loads(raw_content)
    except json.JSONDecodeError as e:
        print(f"[AZURE JSON PARSE ERROR] {e} | Raw: {raw_content}")
        return "Ek technical issue aa gaya hai. Kripya dobara try karein.", current_state, collected
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "rate_limit" in error_str.lower() or "RateLimitError" in type(e).__name__:
            print(f"[AZURE RATE LIMIT] {e}")
            return "Kripya kuch der baad dobara try karein. (API rate limit exceeded)", current_state, collected
        print(f"[AZURE API ERROR] {e}")
        return "Service temporarily unavailable. Please try again.", current_state, collected

    # ── Debug log ─────────────────────────────────────────────────────────────
    print(f"\n[🤖 AZURE DEBUG] State: {current_state} → {parsed_output.get('next_state')}")
    print(f"[🤖 AZURE DEBUG] Notes: {parsed_output.get('debug_notes', '')}")
    print(f"[🤖 AZURE DEBUG] Extracted: {parsed_output.get('extracted_data', {})}")

    new_extracted = parsed_output.get("extracted_data", {})
    llm_next_state = parsed_output.get("next_state", current_state)
    llm_reply = parsed_output.get("reply_to_user", "Kripya dobara batayein.")

    # ── Deterministic post-processing (dates + route resolution) ─────────────
    new_extracted = infer_route_fields(new_extracted, user_input)
    new_extracted = post_process_extracted(new_extracted)

    # ── Preserve confirmed vehicle location unless user explicitly changes it ─
    if collected.get("vehicle_location") and not user_mentions_location(user_input):
        new_extracted["vehicle_location"] = collected["vehicle_location"]

    # ── Smart merge: never lose existing data ─────────────────────────────────
    updated_collected = merge_extracted_data(collected, new_extracted)
    updated_collected = infer_route_fields(updated_collected, user_input)
    updated_collected = enrich_phone_contact(updated_collected, user_input)

    # Guard against a spurious 'NOT_PROVIDED' extraction when the user did not refuse.
    if (
        updated_collected.get("driver_phone") == "NOT_PROVIDED"
        and not is_phone_refusal_response(user_input)
    ):
        print(f"[PHONE REFUSAL OVERRIDE] Ignoring NOT_PROVIDED from input: {user_input}")
        updated_collected["driver_phone"] = None

    # ── Intent locking: if intent already set, do not override ────────────────
    if collected.get("intent") and new_extracted.get("intent") != collected.get("intent"):
        updated_collected["intent"] = collected["intent"]
        print(f"[INTENT LOCK] Keeping original intent: {collected['intent']}")

    updated_collected = advance_service_booking_stage(updated_collected, user_input)
    locked_intent = updated_collected.get("intent")

    ticket_ready = is_ticket_required_intent(locked_intent or "") and all_ticket_fields_present(updated_collected)
    ticket_created = False
    driver_notification = False
    if ticket_ready:
        ticket_id = generate_ticket_id()
        updated_collected["ticket_id"] = ticket_id
        updated_collected = assign_engineer_to_ticket(updated_collected)
        updated_collected["conversation_completed"] = True
        llm_reply = build_ticket_message(ticket_id, updated_collected)
        llm_next_state = "TICKET_RAISED"
        ticket_created = True
        print(f"\n[🎟️ TICKET CREATED] {ticket_id} for {session.get('phone_number', 'UNKNOWN')}")
        print(f"[🎟️ TICKET DATA] {json.dumps(updated_collected)}")
        print(f"[TICKET_UPDATED] {ticket_id} saved with engineer assignment status {updated_collected.get('assignment_status')}")
        database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    # ── State machine override guard ──────────────────────────────────────────
    if not ticket_created and llm_next_state == "TICKET_RAISED":
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")

        phone_filled = phone and len(str(phone)) > 0
        if not all([loc, date_, phone_filled]):
            print(f"[TICKET GUARD] Fields incomplete. loc={loc}, date={date_}, phone={phone}")
            llm_next_state = "COLLECTING_DETAILS"
        else:
            ticket_id = generate_ticket_id()
            updated_collected["ticket_id"] = ticket_id
            updated_collected = assign_engineer_to_ticket(updated_collected)
            updated_collected["conversation_completed"] = True
            llm_reply = build_ticket_message(ticket_id, updated_collected)
            print(f"\n[🎟️ TICKET CREATED] {ticket_id} for {session.get('phone_number', 'UNKNOWN')}")
            print(f"[🎟️ TICKET DATA] {json.dumps(updated_collected)}")
            print(f"[TICKET_UPDATED] {ticket_id} saved with engineer assignment status {updated_collected.get('assignment_status')}")
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    elif llm_next_state == "CASE_CLOSED":
        if not updated_collected.get("resume_date") and is_case_closed_intent(locked_intent or ""):
            llm_next_state = "COLLECTING_DETAILS"
        elif is_ticket_required_intent(locked_intent or ""):
            print(f"[CASE_CLOSE BLOCK] Intent {locked_intent} requires ticket. Overriding CASE_CLOSED → COLLECTING_DETAILS")
            llm_next_state = "COLLECTING_DETAILS"

    elif llm_next_state == "COLLECTING_DETAILS" and is_ticket_required_intent(locked_intent or ""):
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")
        phone_filled = phone and len(str(phone)) > 0
        if all([loc, date_, phone_filled]):
            ticket_id = generate_ticket_id()
            updated_collected["ticket_id"] = ticket_id
            updated_collected = assign_engineer_to_ticket(updated_collected)
            updated_collected["conversation_completed"] = True
            llm_reply = build_ticket_message(ticket_id, updated_collected)
            llm_next_state = "TICKET_RAISED"
            print(f"\n[🎟️ AUTO TICKET] {ticket_id} — all fields filled, LLM missed TICKET_RAISED state.")
            print(f"[TICKET_UPDATED] {ticket_id} saved with engineer assignment status {updated_collected.get('assignment_status')}")
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    if llm_next_state == "COLLECTING_DETAILS" and is_ticket_required_intent(locked_intent or ""):
        llm_reply = build_collection_reply(updated_collected)

    # ── Driver phone forwarding ───────────────────────────────────────────────
    raw_driver_phone = updated_collected.get("driver_phone")
    already_forwarded = collected.get("driver_forwarded")

    if (
        raw_driver_phone
        and raw_driver_phone not in ("NOT_PROVIDED",)
        and not already_forwarded
        and raw_driver_phone != session.get("phone_number")
    ):
        driver_number = normalize_whatsapp_number(raw_driver_phone)
        original_normalized = normalize_whatsapp_number(session.get("phone_number", ""))
        if driver_number != original_normalized:
            forward_alert_to_driver(driver_number, session)
            updated_collected["driver_forwarded"] = True
            driver_notification = True
            print(f"[DRIVER FWD] Alert forwarded to driver {driver_number} for session {session.get('phone_number')}")
        else:
            print(f"[DRIVER FWD] Driver number same as sender after normalisation. Skipping forward.")

    if driver_notification and ticket_created:
        updated_collected["driver_notified"] = True
        llm_reply = build_ticket_message(updated_collected["ticket_id"], updated_collected)

    final_state = llm_next_state
    return llm_reply, final_state, updated_collected


# ==========================================
# 7. SYSTEM WEBHOOKS & API CHANNELS
# ==========================================

@app.get("/api/whatsapp-webhook", response_class=PlainTextResponse)
@app.get("/webhook/", response_class=PlainTextResponse)
@app.get("/webhook", response_class=PlainTextResponse)
def verify_meta_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """Handles the initial authentication verification handshake loop requested by Meta Console."""
    if hub_mode == "subscribe" and hub_verify_token == os.getenv("WHATSAPP_VERIFY_TOKEN"):
        return hub_challenge
    raise HTTPException(status_code=403, detail="Verification token mismatch runtime exception")


@app.post("/api/whatsapp-webhook")
@app.post("/webhook/")
@app.post("/webhook")
async def process_whatsapp_webhook(request: Request):
    """Listens for inbound real-time customer replies routed from Meta WhatsApp nodes."""
    payload = await request.json()

    try:
        entry = payload["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return {"status": "ignored_event"}

        message_data = value["messages"][0]
        clean_sender = message_data["from"]
        message_id = message_data.get("id", "")

        if database.is_duplicate_message(message_id):
            print(f"[DUPLICATE] Skipping already-processed message_id: {message_id}")
            return {"status": "duplicate_ignored"}

        database.mark_message_processed(message_id)

        user_input = message_data["text"]["body"].strip()
        print(f"\n[📥 INBOUND] From: {clean_sender} | Message: {user_input}")

    except (KeyError, IndexError) as e:
        print(f"[PAYLOAD PARSE ERROR] {e}")
        return {"status": "malformed_meta_payload"}

    session = database.get_session(clean_sender)
    session["phone_number"] = clean_sender

    if session["current_state"] in ["TICKET_RAISED", "CASE_CLOSED"] and not session["collected_json"].get("conversation_completed"):
        print(f"[SESSION] Conversation already closed (state={session['current_state']}). Ignoring further input.")
        return {"status": "conversation_closed"}

    session["chat_history"].append({"role": "user", "text": user_input})

    reply_text, next_state, updated_json = process_message(session, user_input)

    session["chat_history"].append({"role": "bot", "text": reply_text})

    database.save_session(
        phone_number=clean_sender,
        current_state=next_state,
        collected_json=updated_json,
        chat_history=session["chat_history"]
    )

    send_whatsapp_meta(clean_sender, reply_text)
    print(f"[📤 OUTBOUND] To: {clean_sender} | State: {next_state} | Reply: {reply_text[:80]}...")

    return {"status": "processed"}


@app.post("/api/trigger-outage")
async def trigger_outage(payload: OutageRequest):
    """
    API Endpoint to initialize tracking downtime alerts from backend scripts.

    Flow:
    1. Normalize gps_data from the request payload.
    2. Classify root cause via GPS analysis API: BATTERY_ISSUE | MAIN_POWER_CUT | OTHER_ISSUE.
    3. Build a tailored initial WhatsApp message using gps_data fields.
    4. Persist session with root_cause + gps_data in collected_json.
    5. Send the WhatsApp message.
    """
    print(f"[🚨 OUTAGE] Received for {payload.phone_number} | Vehicle: {payload.vehicle_no}")

    # ── Step 1: Normalize gps_data ────────────────────────────────────────────
    raw_gps = payload.gps_data.model_dump() if payload.gps_data else {}
    # Merge top-level fields as fallbacks so builders always have last_location / timestamp
    gps_data = {
        "last_location": payload.last_location,
        "timestamp":     payload.timestamp,
        **raw_gps,
    }

    # ── Step 2: Classify root cause from gps_data fields ────────────────────
    #
    # Priority order:
    #   1. ismainpoerconnected == "0"          → MAIN_POWER_CUT
    #   2. main_powervoltage present & < 11 V  → BATTERY_ISSUE
    #   3. Fallback to external API (if URL set), else OTHER_ISSUE
    #
    is_main_connected = str(gps_data.get("ismainpoerconnected", "1")).strip()
    voltage_raw       = gps_data.get("main_powervoltage")

    main_power_issue = (is_main_connected == "0")
    battery_issue    = (
        not main_power_issue
        and voltage_raw is not None
        and float(voltage_raw) < 11.0
    )

    if main_power_issue:
        root_cause = "MAIN_POWER_CUT"
    elif battery_issue:
        root_cause = "BATTERY_ISSUE"
    else:
        # Neither detected from gps_data — try external API as a final check
        analysis = analyze_gps_device(
            vehicle_no=payload.vehicle_no,
            phone_number=payload.phone_number,
        )
        root_cause       = analysis["root_cause"]
        battery_issue    = analysis["battery_issue"]
        main_power_issue = analysis["main_power_issue"]

    print(
        f"[🔍 PRE-ANALYSIS] root_cause={root_cause} | "
        f"battery_issue={battery_issue} | main_power_issue={main_power_issue} | "
        f"voltage={voltage_raw}V | main_connected={is_main_connected}"
    )

    # ── Step 3: Build tailored initial message ────────────────────────────────
    initial_alert_msg = build_initial_alert(
        vehicle_no=payload.vehicle_no,
        gps_data=gps_data,
        root_cause=root_cause,
    )

    # ── Step 4: Persist session ───────────────────────────────────────────────
    database.save_session(
        phone_number=payload.phone_number,
        current_state="INITIAL_ALERT",
        collected_json={
            # Pre-analysis fields
            "battery_issue":    battery_issue,
            "main_power_issue": main_power_issue,
            "root_cause":       root_cause,
            # GPS device data from request
            "gps_gpstime":      gps_data.get("gpstime"),
            "gps_location":     gps_data.get("current_location") or payload.last_location,
            "gps_vehicle_state": gps_data.get("vehicle_state"),
            # Pre-fill driver details if provided in gps_data
            "driver_name":      gps_data.get("driver_name"),
            "driver_phone":     gps_data.get("driver_phone"),
            # Standard collected fields
            "intent":           None,
            "vehicle_location": gps_data.get("current_location") or None,
            "service_date":     None,
            "arrival_date":     None,
            "contact_person":   None,
            "origin_city":      None,
            "destination_city": None,
            "resume_date":      None,
            "ticket_id":        None,
            "driver_forwarded": False,
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )

    # ── Step 5: Send WhatsApp message ─────────────────────────────────────────
    send_whatsapp_meta(payload.phone_number, initial_alert_msg)
    print(
        f"[🚨 OUTAGE ALERT SENT] {payload.phone_number} | "
        f"Vehicle: {payload.vehicle_no} | root_cause: {root_cause}"
    )

    return {
        "status":           "alert_fired",
        "root_cause":       root_cause,
        "battery_issue":    battery_issue,
        "main_power_issue": main_power_issue,
    }