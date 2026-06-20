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

class OutageRequest(BaseModel):
    phone_number: str
    vehicle_no: str
    last_location: str
    timestamp: str

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


def build_ticket_message(ticket_id: str, data: dict) -> str:
    """Builds the final ticket creation confirmation message."""
    return (
        f"✅ Service request create kar di gayi hai!\n\n"
        f"📋 Ticket Details:\n\n"
        f"🎫 Ticket ID: {ticket_id}\n"
        f"📍 Location: {data.get('vehicle_location', 'N/A')}\n"
        f"📅 Service Date: {data.get('service_date', 'N/A')}\n"
        f"📞 Contact: {data.get('driver_phone', 'N/A')}\n\n"
        f"👤 Engineer assignment jald ho jayega.\n\n"
        f"Engineer aapse jald sampark karega.\n\n"
        f"Koi sawal ho toh Ticket ID {ticket_id} ke saath humse sampark karein.\n\n"
        f"Dhanyavaad!"
    )


def is_case_closed_intent(intent: str) -> bool:
    """Returns True for intents that close the case without a ticket."""
    return intent in ["WORKSHOP", "ACCIDENT", "BATTERY_DISCONNECT", "GPS_REMOVED"]


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
        state="INITIAL_ALERT",
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
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}],
    )

    send_whatsapp_meta(driver_number, initial_alert_msg)
    print(f"[DRIVER FWD] Alert forwarded to driver {driver_number}")


# ==========================================
# 3. GROQ SYSTEM DEFINITION
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

## INTENT MAPPING
When user replies to initial alert (option number or description), map to intent:
- "1" or "workshop" or "service center" → intent = "WORKSHOP"
- "2" or "accident" → intent = "ACCIDENT"
- "3" or "battery" or "battery disconnect" → intent = "BATTERY_DISCONNECT"
- "4" or "gps removed" or "nikal diya" → intent = "GPS_REMOVED"
- "5" or "gps damaged" or "toot gaya" → intent = "GPS_DAMAGED"
- "6" or "running" or "chal rahi" or "gadi chal rahi" → intent = "VEHICLE_RUNNING_GPS_NOT_UPDATING"
- "7" or "standing" or "khadi hai" → intent = "VEHICLE_STANDING"
- "8" or "other" or anything else unclear → intent = "OTHER"
Natural language like "gadi running me hai" or "GPS update nahi ho raha chal rahi hai" → VEHICLE_RUNNING_GPS_NOT_UPDATING.

## CASE CLOSED FLOW (intents: WORKSHOP, ACCIDENT, BATTERY_DISCONNECT, GPS_REMOVED)
- If resume_date is NOT known: set next_state = "COLLECTING_DETAILS", ask ONLY "Vehicle dobara kab running condition mein aa jayegi?"
- If resume_date IS known: set next_state = "CASE_CLOSED", reply = "✅ Update note kar liya gaya hai. Dhanyavaad."
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
    return (
        f"TODAY'S DATE: {today_str}\n"
        f"Current Bot State: {session['current_state']}\n"
        f"Intent Locked: {session['collected_json'].get('intent', 'NOT SET YET')}\n"
        f"Currently Extracted Data: {json.dumps(session['collected_json'])}\n\n"
        f"Recent Conversation:\n"
        + "\n".join([f"{m['role'].upper()}: {m['text']}" for m in recent_history])
        + f"\nUSER: {user_input}"
    )


# ==========================================
# 4. META WHATSAPP OUTBOUND UTILITY
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
# 5. CORE MESSAGE PROCESSING LOGIC
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
    if not collected.get("service_date"):
        return "Service date kya hai? Kab service karwana hai?"
    if not collected.get("driver_phone"):
        return "Driver ka phone number kya hai? Contact number do."
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

    # ── Smart merge: never lose existing data ─────────────────────────────────
    updated_collected = merge_extracted_data(collected, new_extracted)
    updated_collected = infer_route_fields(updated_collected, user_input)

    # ── Intent locking: if intent already set, do not override ────────────────
    if collected.get("intent") and new_extracted.get("intent") != collected.get("intent"):
        updated_collected["intent"] = collected["intent"]
        print(f"[INTENT LOCK] Keeping original intent: {collected['intent']}")

    locked_intent = updated_collected.get("intent")

    # ── State machine override guard ──────────────────────────────────────────
    if llm_next_state == "TICKET_RAISED":
        # Final validation: ensure all 3 required fields are truly present
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")

        # "NOT_PROVIDED" counts as filled (user refused, we proceed anyway)
        phone_filled = phone and len(str(phone)) > 0
        if not all([loc, date_, phone_filled]):
            print(f"[TICKET GUARD] Fields incomplete. loc={loc}, date={date_}, phone={phone}")
            llm_next_state = "COLLECTING_DETAILS"
        else:
            # All fields present — generate ticket and override reply
            ticket_id = generate_ticket_id()
            llm_reply = build_ticket_message(ticket_id, updated_collected)
            updated_collected["ticket_id"] = ticket_id
            print(f"\n[🎟️ TICKET CREATED] {ticket_id} for {session.get('phone_number', 'UNKNOWN')}")
            print(f"[🎟️ TICKET DATA] {json.dumps(updated_collected)}")

            # Persist ticket to its own table
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    elif llm_next_state == "CASE_CLOSED":
        # Validate resume_date exists before closing
        if not updated_collected.get("resume_date") and is_case_closed_intent(locked_intent or ""):
            llm_next_state = "COLLECTING_DETAILS"
        # CRITICAL: if locked_intent is a TICKET flow, LLM is wrong to CASE_CLOSE —
        # user said "workshop" mid-flow but intent is already GPS_DAMAGED etc.
        elif is_ticket_required_intent(locked_intent or ""):
            print(f"[CASE_CLOSE BLOCK] Intent {locked_intent} requires ticket. Overriding CASE_CLOSED → COLLECTING_DETAILS")
            llm_next_state = "COLLECTING_DETAILS"

    # ── Auto-promote to TICKET_RAISED if all fields now filled (LLM missed it) ─
    elif llm_next_state == "COLLECTING_DETAILS" and is_ticket_required_intent(locked_intent or ""):
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")
        phone_filled = phone and len(str(phone)) > 0
        if all([loc, date_, phone_filled]):
            ticket_id = generate_ticket_id()
            llm_reply = build_ticket_message(ticket_id, updated_collected)
            updated_collected["ticket_id"] = ticket_id
            llm_next_state = "TICKET_RAISED"
            print(f"\n[🎟️ AUTO TICKET] {ticket_id} — all fields filled, LLM missed TICKET_RAISED state.")
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    # If we are still collecting required ticket fields, use deterministic prompts.
    if llm_next_state == "COLLECTING_DETAILS" and is_ticket_required_intent(locked_intent or ""):
        llm_reply = build_collection_reply(updated_collected)

    # ── Driver phone forwarding ───────────────────────────────────────────────
    # If a driver/contact phone number was just extracted and we haven't already
    # forwarded the alert to a driver in this session, do it now.
    raw_driver_phone = updated_collected.get("driver_phone")
    already_forwarded = collected.get("driver_forwarded")  # flag set on first forward

    if (
        raw_driver_phone
        and raw_driver_phone not in ("NOT_PROVIDED",)
        and not already_forwarded
        and raw_driver_phone != session.get("phone_number")  # don't forward to self
    ):
        driver_number = normalize_whatsapp_number(raw_driver_phone)
        # Extra guard: don't forward to the original sender even after normalisation
        original_normalized = normalize_whatsapp_number(session.get("phone_number", ""))
        if driver_number != original_normalized:
            forward_alert_to_driver(driver_number, session)
            updated_collected["driver_forwarded"] = True  # prevent double-forward

            # Reply only with driver forwarding confirmation
            forwarded_notice = (
                f"✅ Humne driver ko ({driver_number}) message kar diya hai. Woh jald respond karenge."
            )
            llm_reply = forwarded_notice
            print(f"[DRIVER FWD] Reply replaced with driver confirmation for {session.get('phone_number')}")
        else:
            print(f"[DRIVER FWD] Driver number same as sender after normalisation. Skipping forward.")

    final_state = llm_next_state
    return llm_reply, final_state, updated_collected


# ==========================================
# 6. SYSTEM WEBHOOKS & API CHANNELS
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

        # ── Duplicate message prevention ──────────────────────────────────────
        if database.is_duplicate_message(message_id):
            print(f"[DUPLICATE] Skipping already-processed message_id: {message_id}")
            return {"status": "duplicate_ignored"}

        database.mark_message_processed(message_id)

        user_input = message_data["text"]["body"].strip()
        print(f"\n[📥 INBOUND] From: {clean_sender} | Message: {user_input}")

    except (KeyError, IndexError) as e:
        print(f"[PAYLOAD PARSE ERROR] {e}")
        return {"status": "malformed_meta_payload"}

    # ── Load session ──────────────────────────────────────────────────────────
    session = database.get_session(clean_sender)
    session["phone_number"] = clean_sender

    # ── Guard: conversation already fully closed ──────────────────────────────
    if session["current_state"] in ["TICKET_RAISED", "CASE_CLOSED"]:
        print(f"[SESSION] Conversation already closed (state={session['current_state']}). Ignoring further input.")
        return {"status": "conversation_closed"}

    # ── Append user message to history ───────────────────────────────────────
    session["chat_history"].append({"role": "user", "text": user_input})

    # ── Core processing ───────────────────────────────────────────────────────
    reply_text, next_state, updated_json = process_message(session, user_input)

    # ── Append bot reply to history ───────────────────────────────────────────
    session["chat_history"].append({"role": "bot", "text": reply_text})

    # ── Persist session ───────────────────────────────────────────────────────
    database.save_session(
        phone_number=clean_sender,
        state=next_state,
        collected_json=updated_json,
        chat_history=session["chat_history"]
    )

    # ── Send reply ────────────────────────────────────────────────────────────
    send_whatsapp_meta(clean_sender, reply_text)
    print(f"[📤 OUTBOUND] To: {clean_sender} | State: {next_state} | Reply: {reply_text[:80]}...")

    return {"status": "processed"}


@app.post("/api/trigger-outage")
async def trigger_outage(payload: OutageRequest):
    """API Endpoint to initialize tracking downtime alerts from backend scripts."""
    initial_alert_msg = (
        f"Namaste Sir,\n\n"
        f"Vehicle {payload.vehicle_no} se GPS data receive nahi ho raha hai.\n\n"
        f"📍 Last Known Location: {payload.last_location}\n"
        f"🕐 Last Update: {payload.timestamp}\n\n"
        f"Kripya batayein ki aapki vehicle ki current status kya hai:\n\n"
        f"1️⃣ Workshop / Service Center\n"
        f"2️⃣ Accident\n"
        f"3️⃣ Battery Disconnect\n"
        f"4️⃣ GPS Removed\n"
        f"5️⃣ GPS Damaged\n"
        f"6️⃣ Vehicle Running but GPS Not Updating\n"
        f"7️⃣ Vehicle Standing\n"
        f"8️⃣ Other\n\n"
        f"Reply with the option number or describe the issue in your own words."
    )

    database.save_session(
        phone_number=payload.phone_number,
        state="INITIAL_ALERT",
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
            "driver_forwarded": False,
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )

    send_whatsapp_meta(payload.phone_number, initial_alert_msg)
    print(f"[🚨 OUTAGE ALERT] Fired for {payload.phone_number} | Vehicle: {payload.vehicle_no}")
    return {"status": "alert_fired"}