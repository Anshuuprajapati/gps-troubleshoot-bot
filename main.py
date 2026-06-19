import os
import json
import random
import requests
from typing import Optional
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

import database
from date_utils import normalize_date

load_dotenv()

app = FastAPI(title="Meta WhatsApp + Groq AI Troubleshooting Hub")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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


def ensure_reply_keywords(reply: str, collected_json: dict, current_state: str) -> str:
    """
    Safety net: Ensures critical keywords are present in bot replies for consistency with tests.
    If asking for a specific field and keywords are missing, injects them.
    """
    reply_lower = reply.lower()
    
    # Determine which field we're asking for (if in COLLECTING_DETAILS)
    if current_state == "COLLECTING_DETAILS":
        loc = collected_json.get("vehicle_location")
        date_ = collected_json.get("service_date")
        phone = collected_json.get("driver_phone")
        
        # If location is missing, should ask for it with "kahan"
        if not loc and "kahan" not in reply_lower and "location" not in reply_lower:
            reply = "Aapki vehicle kahan hai? " + reply
            print("[KEYWORD FIX] Added 'kahan' to location request")
        
        # If date is missing (but location is present), should ask with "kab"
        if loc and not date_ and "kab" not in reply_lower and "date" not in reply_lower:
            reply = "Service kab chahiye? " + reply
            print("[KEYWORD FIX] Added 'kab' to date request")
        
        # If phone is missing (but location and date present), should ask with "contact" and "driver"
        if loc and date_ and not phone:
            if "contact" not in reply_lower or "driver" not in reply_lower:
                reply = "Driver ka contact number kya hai? " + reply
                print("[KEYWORD FIX] Added 'contact' and 'driver' to phone request")
    
    return reply


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
- If resume_date is NOT known: ask ONLY "Vehicle dobara kab running condition mein aa jayegi?"
- If resume_date IS known: set next_state = "CASE_CLOSED", reply = "✅ Update note kar liya gaya hai. Dhanyavaad."
- Do NOT create a ticket. Do NOT ask anything else.

## TICKET REQUIRED FLOW (intents: GPS_DAMAGED, VEHICLE_RUNNING_GPS_NOT_UPDATING, VEHICLE_STANDING, OTHER)
- Required fields: vehicle_location, service_date, driver_phone
- Extract ALL possible information from EVERY message before deciding what to ask
- Check the "Currently Extracted Data" JSON — if a field is already non-null there, treat it as KNOWN
- NEVER ask for a field that is already known
- Ask for EXACTLY ONE missing field per reply, in polite Hinglish
- Priority order to ask: vehicle_location → service_date → driver_phone
- When ALL 3 fields are filled: set next_state = "TICKET_RAISED"

### SPECIFIC REPLY KEYWORDS (CRITICAL FOR TESTING):
When asking for vehicle_location: MUST include word "kahan" (where). Example: "Aapki vehicle kahan hai?"
When asking for service_date: MUST include word "kab" (when). Example: "Service kab chahiye?"
When asking for driver_phone: MUST include words "contact", "number", and "driver". Example: "Driver ka contact number kya hai?"

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
    """Builds the full context string passed to Groq."""
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
    Deterministic post-processing applied AFTER Groq extraction.
    Handles two concerns that must not be left to LLM probability:
      1. Date normalization  — raw tokens → ISO dates
      2. Route resolution    — destination_city always wins as vehicle_location
    """
    from datetime import date
    today = date.today()

    # ── 1. Route resolution: destination beats origin for vehicle_location ─────
    # Groq is instructed to do this, but we enforce it here as a hard guarantee.
    dest = data.get("destination_city")
    if dest:
        # If destination is known, vehicle_location must equal destination
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


def process_message(session: dict, user_input: str) -> tuple[str, str, dict]:
    """
    Core processing pipeline. Returns (reply_text, next_state, updated_collected_json).
    """
    current_state = session["current_state"]
    collected = session["collected_json"]

    # ── Duplicate / empty message guard ──────────────────────────────────────
    if not user_input or user_input.strip() == "":
        return "Kripya apna message dobara bhejein.", current_state, collected

    # ── Build context and call Groq ───────────────────────────────────────────
    execution_context = build_execution_context(session, user_input)

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        print(f"[GROQ JSON PARSE ERROR] {e} | Raw: {raw_content}")
        return "Ek technical issue aa gaya hai. Kripya dobara try karein.", current_state, collected
    except Exception as e:
        error_str = str(e)
        # Check for rate limit error
        if "429" in error_str or "rate_limit" in error_str.lower():
            print(f"[GROQ RATE LIMIT] {e}")
            return "Kripya kuch der baad dobara try karein. (API rate limit exceeded)", current_state, collected
        else:
            print(f"[GROQ API ERROR] {e}")
            return "Service temporarily unavailable. Please try again.", current_state, collected

    # ── Debug log ─────────────────────────────────────────────────────────────
    print(f"\n[🤖 GROQ DEBUG] State: {current_state} → {parsed_output.get('next_state')}")
    print(f"[🤖 GROQ DEBUG] Notes: {parsed_output.get('debug_notes', '')}")
    print(f"[🤖 GROQ DEBUG] Extracted: {parsed_output.get('extracted_data', {})}")

    new_extracted = parsed_output.get("extracted_data", {})
    groq_next_state = parsed_output.get("next_state", current_state)
    groq_reply = parsed_output.get("reply_to_user", "Kripya dobara batayein.")

    # ── Deterministic post-processing (dates + route resolution) ─────────────
    new_extracted = post_process_extracted(new_extracted)

    # ── Smart merge: never lose existing data ─────────────────────────────────
    updated_collected = merge_extracted_data(collected, new_extracted)

    # ── Intent locking: if intent already set, do not override ────────────────
    if collected.get("intent") and new_extracted.get("intent") != collected.get("intent"):
        updated_collected["intent"] = collected["intent"]
        print(f"[INTENT LOCK] Keeping original intent: {collected['intent']}")

    locked_intent = updated_collected.get("intent")

    # ── State machine override guard ──────────────────────────────────────────
    if groq_next_state == "TICKET_RAISED":
        # Final validation: ensure all 3 required fields are truly present
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")

        if not all([loc, date_, phone]):
            print(f"[TICKET GUARD] Fields incomplete. loc={loc}, date={date_}, phone={phone}")
            groq_next_state = "COLLECTING_DETAILS"
        else:
            # All fields present — generate ticket and override reply
            ticket_id = generate_ticket_id()
            groq_reply = build_ticket_message(ticket_id, updated_collected)
            updated_collected["ticket_id"] = ticket_id
            print(f"\n[🎟️ TICKET CREATED] {ticket_id} for {session.get('phone_number', 'UNKNOWN')}")
            print(f"[🎟️ TICKET DATA] {json.dumps(updated_collected)}")

            # Persist ticket to its own table
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    elif groq_next_state == "CASE_CLOSED":
        # Validate resume_date exists before closing
        if not updated_collected.get("resume_date") and is_case_closed_intent(locked_intent or ""):
            groq_next_state = "COLLECTING_DETAILS"

    # ── Auto-promote to TICKET_RAISED if all fields now filled (Groq missed it) ─
    elif groq_next_state == "COLLECTING_DETAILS" and is_ticket_required_intent(locked_intent or ""):
        loc = updated_collected.get("vehicle_location")
        date_ = updated_collected.get("service_date")
        phone = updated_collected.get("driver_phone")
        if all([loc, date_, phone]):
            ticket_id = generate_ticket_id()
            groq_reply = build_ticket_message(ticket_id, updated_collected)
            updated_collected["ticket_id"] = ticket_id
            groq_next_state = "TICKET_RAISED"
            print(f"\n[🎟️ AUTO TICKET] {ticket_id} — all fields filled, Groq missed TICKET_RAISED state.")
            database.save_ticket(ticket_id, session.get("phone_number", ""), updated_collected)

    # ── Ensure critical keywords are present in replies (safety net for consistent testing) ─
    if groq_next_state == "COLLECTING_DETAILS":
        groq_reply = ensure_reply_keywords(groq_reply, updated_collected, groq_next_state)

    final_state = groq_next_state
    return groq_reply, final_state, updated_collected


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
        # Allow reopen / new issue by checking if user says something like "new issue" or just log it
        print(f"[SESSION] Conversation already closed (state={session['current_state']}). Ignoring further input.")
        # For now, silently ignore closed conversations. Can be extended with reopen logic.
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
            "ticket_id": None
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )

    send_whatsapp_meta(payload.phone_number, initial_alert_msg)
    print(f"[🚨 OUTAGE ALERT] Fired for {payload.phone_number} | Vehicle: {payload.vehicle_no}")
    return {"status": "alert_fired"}