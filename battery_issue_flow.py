"""
GPS AI Support Chatbot - Battery Issue Flow
Complete single-file implementation covering all features from the checklist:
  - Outage detection & PRE_ANALYSIS
  - Full state machine with LLM brain
  - Entity extraction (name, phone, location, date, etc.)
  - Guidance system (how-to answers)
  - Confusion & side-question handling
  - Strategy change mid-conversation
  - Driver communication & separate driver session
  - Live GPS API recheck
  - Intent lock after remote failure
  - Service detail collection
  - Engineer assignment & ticket creation
  - Hindi / English / Hinglish support

Dependencies:
    pip install fastapi uvicorn openai python-dotenv requests
"""

import os
import json
import logging
import re
import sqlite3
import random
import string
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s: %(message)s"
)
logger = logging.getLogger("GPSChatbot")

# ──────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ──────────────────────────────────────────────────────────────────────────────
openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# ──────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GPS AI Support Chatbot - Battery Issue Flow")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE LAYER (SQLite, self-contained)
# ══════════════════════════════════════════════════════════════════════════════
DB_PATH = os.getenv("DB_PATH", "gps_chatbot.db")


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                phone_number  TEXT PRIMARY KEY,
                current_state TEXT NOT NULL,
                collected_json TEXT NOT NULL DEFAULT '{}',
                chat_history   TEXT NOT NULL DEFAULT '[]',
                updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id   TEXT PRIMARY KEY,
                phone_number TEXT NOT NULL,
                ticket_data  TEXT NOT NULL DEFAULT '{}',
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def save_session(phone_number: str, current_state: str,
                 collected_json: dict, chat_history: list):
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO sessions (phone_number, current_state, collected_json, chat_history, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(phone_number) DO UPDATE SET
                current_state  = excluded.current_state,
                collected_json = excluded.collected_json,
                chat_history   = excluded.chat_history,
                updated_at     = CURRENT_TIMESTAMP
        """, (phone_number, current_state,
              json.dumps(collected_json), json.dumps(chat_history)))


def get_session(phone_number: str) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE phone_number = ?", (phone_number,)
        ).fetchone()
    if not row:
        return None
    return {
        "phone_number": row["phone_number"],
        "current_state": row["current_state"],
        "collected_json": json.loads(row["collected_json"]),
        "chat_history": json.loads(row["chat_history"]),
    }


def delete_session(phone_number: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE phone_number = ?", (phone_number,))


def save_ticket(ticket_id: str, phone_number: str, ticket_data: dict):
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tickets (ticket_id, phone_number, ticket_data) VALUES (?, ?, ?)",
            (ticket_id, phone_number, json.dumps(ticket_data))
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATE NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

DATE_FORMATS = [
    "%d-%m-%Y", "%d/%m/%Y", "%d %m %Y",
    "%d-%m-%y", "%d/%m/%y",
    "%Y-%m-%d", "%B %d %Y", "%b %d %Y",
]


def normalize_date(raw: str) -> str:
    raw = raw.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    # Relative dates
    today = datetime.today()
    lower = raw.lower()
    if "kal" in lower or "tomorrow" in lower:
        return (today + timedelta(days=1)).strftime("%d-%m-%Y")
    if "parso" in lower or "day after" in lower:
        return (today + timedelta(days=2)).strftime("%d-%m-%Y")
    if "aaj" in lower or "today" in lower:
        return today.strftime("%d-%m-%Y")
    return raw  # Return as-is if unparseable


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WHATSAPP META TRANSPORT
# ══════════════════════════════════════════════════════════════════════════════

def send_whatsapp_meta(to_number: str, text_body: str):
    """Dispatch a WhatsApp message via Meta Cloud API."""
    if not to_number or to_number == "N/A":
        logger.warning("Skipping WhatsApp send — invalid number: %s", to_number)
        return
    url = (
        f"https://graph.facebook.com/v18.0/"
        f"{os.getenv('WHATSAPP_PHONE_NUMBER_ID')}/messages"
    )
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
        res = requests.post(url, headers=headers, json=payload, timeout=8)
        if res.status_code != 200:
            logger.error("WhatsApp dispatch failed [%s]: %s", res.status_code, res.text)
    except Exception as exc:
        logger.critical("WhatsApp transport error to %s: %s", to_number, exc)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GPS HARDWARE API RECHECK  (mock; swap with real endpoint)
# ══════════════════════════════════════════════════════════════════════════════

def run_hardware_api_recheck(vehicle_no: str) -> Dict[str, Any]:
    """
    Query hardware tracking API for live GPS status.
    Returns dict with keys: gps_working, main_power_connected, main_powervoltage.
    Replace the stub body with your real API call.
    """
    logger.info("Running live GPS recheck for vehicle: %s", vehicle_no)
    try:
        # --- REPLACE THIS WITH REAL API CALL ---
        # response = requests.get(f"{GPS_API_BASE}/status/{vehicle_no}", timeout=5)
        # data = response.json()
        # return {
        #     "gps_working": data["gpsStatus"] == 1,
        #     "main_power_connected": data["ismainpowerconnected"],
        #     "main_powervoltage": data["main_powervoltage"],
        # }
        return {
            "gps_working": False,
            "main_power_connected": "1",
            "main_powervoltage": 12.4,
        }
    except Exception as exc:
        logger.error("GPS API recheck failed for %s: %s", vehicle_no, exc)
        return {"gps_working": False, "main_power_connected": "1", "main_powervoltage": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTITY EXTRACTOR
# Extracts: phone numbers, names, locations, dates, cities from free text
# ══════════════════════════════════════════════════════════════════════════════

STOPWORDS = {
    "ko", "bol", "do", "par", "number", "is", "iss", "se", "ka", "ki",
    "ke", "hai", "hain", "aur", "ya", "the", "and", "or", "a", "an",
    "please", "karo", "karna", "karein", "bata", "batao",
}


def parse_inline_entities(text: str) -> Dict[str, Optional[str]]:
    """
    Extract structured entities from a raw user message.
    Returns a dict with optional keys: phone, name, date, location, city.
    """
    entities: Dict[str, Optional[str]] = {
        "phone": None, "name": None,
        "date": None, "location": None, "city": None,
    }

    # Phone: 10-digit Indian mobile or with 91/+91 prefix
    phone_match = re.search(r'\b(?:\+?91)?([6-9]\d{9})\b', text)
    if phone_match:
        raw_phone = phone_match.group(1)
        entities["phone"] = f"91{raw_phone}" if len(raw_phone) == 10 else raw_phone

    # Date: DD-MM-YYYY, DD/MM/YYYY, or relative words
    date_match = re.search(
        r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|kal|parso|aaj|tomorrow|today|day after)\b',
        text, re.IGNORECASE
    )
    if date_match:
        entities["date"] = normalize_date(date_match.group(1))

    # Name: capitalize non-stopword alpha words adjacent to phone or "driver"
    clean = re.sub(r'\b(?:\+?91)?[6-9]\d{9}\b', '', text)
    words = [
        w.strip(".,!?") for w in clean.split()
        if w.lower() not in STOPWORDS and re.match(r'^[A-Za-z]{3,}$', w)
    ]
    if words:
        entities["name"] = words[0].capitalize()

    return entities


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — LLM BRAIN  (single unified reasoning function)
# Handles: intent, guidance, confusion, side-questions, strategy change,
#          entity confirmation, state transitions — all in one call.
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_BRAIN_TEMPLATE = """
You are the AI brain for a GPS fleet-tracking support chatbot.
You must behave like an experienced, empathetic human support executive — NOT a form.

=== VEHICLE CONTEXT ===
Vehicle No        : {vehicle_no}
Root Cause        : {root_cause}
Driver Name       : {driver_name}
Driver Phone      : {driver_phone}
Last Location     : {last_location}
Physical Intent   : {physical_intent}
Intent Locked     : {intent_locked}
Current State     : {current_state}
Extracted Phone   : {extracted_phone}
Extracted Name    : {extracted_name}

=== CONVERSATION HISTORY (last 6 turns) ===
{history_snippet}

=== STATE MACHINE RULES ===

BATTERY_INITIAL_RESPONSE
  → If user says they (owner/manager) will charge/handle it:
      next_state: BATTERY_WAITING_FOR_CHARGE
  → If user mentions driver, operator, or someone else:
      next_state: BATTERY_DRIVER_CONFIRMATION
  → If user asks how to charge, where is battery, how long, etc.:
      Answer the question, stay in: BATTERY_INITIAL_RESPONSE
  → If user is confused or asks "what is this"?
      Explain patiently, stay in: BATTERY_INITIAL_RESPONSE
  → If user asks for vehicle location or driver details:
      Answer using context, stay in: BATTERY_INITIAL_RESPONSE
  → If user demands engineer visit immediately:
      Politely say let's try charging first, stay in: BATTERY_INITIAL_RESPONSE

BATTERY_DRIVER_CONFIRMATION
  → If user confirms driver or provides a number/name:
      next_state: BATTERY_WAITING_FOR_CHARGE
  → If user changes mind and says they will handle it:
      next_state: BATTERY_WAITING_FOR_CHARGE

BATTERY_WAITING_FOR_CHARGE
  → If user/driver says done, charged, connected, fixed:
      next_state: BATTERY_POST_CHECK (the system will call the GPS API automatically)
  → If user says battery is broken / karab / needs replacement:
      next_state: COLLECTING_SERVICE_DETAILS
  → If user changes mind, wants driver involved:
      next_state: BATTERY_DRIVER_CONFIRMATION

BATTERY_POST_CHECK  (system decides; LLM only generates reply)
  → If GPS API says active → next_state: COMPLETED
  → If main power is cut → next_state: MAIN_POWER_FLOW
  → Otherwise → next_state: AWAITING_PHYSICAL_DIAGNOSIS

AWAITING_PHYSICAL_DIAGNOSIS
  → Understand what the physical condition is.
  → Detect intent: GPS_DAMAGED | GPS_REMOVED | WORKSHOP | ACCIDENT |
                   VEHICLE_RUNNING | VEHICLE_STANDING | BATTERY_DISCONNECT | OTHER
  → next_state: COLLECTING_SERVICE_DETAILS

COLLECTING_SERVICE_DETAILS
  → Collect: Current Location, Destination, ETA date, Service City, Contact Person
  → If all 4+ fields collected: next_state: TICKET_CREATED
  → If missing fields: remain in COLLECTING_SERVICE_DETAILS, ask for missing ones only

MAIN_POWER_FLOW
  → Explain wiring issue. Ask owner or driver to check main power connection.
  → On confirmation of fix: next_state: BATTERY_POST_CHECK
  → On physical damage: next_state: AWAITING_PHYSICAL_DIAGNOSIS

COMPLETED / TICKET_CREATED
  → Close conversation gracefully.

=== OUTPUT FORMAT (strict JSON, no markdown) ===
{{
  "next_state": "<STATE_NAME>",
  "bot_reply": "<Reply in same language as user — Hindi/English/Hinglish>",
  "extracted_updates": {{
    "driver_name": "<if newly provided, else null>",
    "driver_phone": "<if newly provided, else null>",
    "location": "<if newly provided, else null>",
    "destination": "<if newly provided, else null>",
    "service_city": "<if newly provided, else null>",
    "eta_date": "<if newly provided, else null>",
    "contact_person": "<if newly provided, else null>",
    "physical_intent": "<if newly detected, else null>"
  }}
}}
"""


def execute_llm_brain(
    current_state: str,
    user_input: str,
    context: dict,
    chat_history: list,
    extracted_entities: Dict[str, Optional[str]],
) -> Tuple[str, str, dict]:
    """
    Single LLM call that handles ALL reasoning:
    intent detection, guidance, confusion, side-questions,
    strategy change, state transition, and reply generation.

    Returns: (next_state, bot_reply, extracted_updates_dict)
    """
    # Build a readable history snippet (last 6 turns)
    history_lines = []
    for turn in chat_history[-6:]:
        role = turn.get("role", "?")
        text = turn.get("text") or turn.get("content", "")
        history_lines.append(f"{role.upper()}: {text}")
    history_snippet = "\n".join(history_lines) if history_lines else "(no history)"

    system_prompt = SYSTEM_BRAIN_TEMPLATE.format(
        vehicle_no=context.get("vehicle_no", "N/A"),
        root_cause=context.get("root_cause", "BATTERY_ISSUE"),
        driver_name=context.get("driver_name", "N/A"),
        driver_phone=context.get("driver_phone", "N/A"),
        last_location=context.get("last_location", "N/A"),
        physical_intent=context.get("physical_intent", "N/A"),
        intent_locked=context.get("intent_locked", False),
        current_state=current_state,
        extracted_phone=extracted_entities.get("phone") or "N/A",
        extracted_name=extracted_entities.get("name") or "N/A",
        history_snippet=history_snippet,
    )

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        next_state = parsed.get("next_state", current_state)
        bot_reply = parsed.get("bot_reply", "")
        updates = parsed.get("extracted_updates", {})
        return next_state, bot_reply, updates

    except Exception as exc:
        logger.error("LLM brain error: %s", exc, exc_info=True)
        return current_state, "System mein thodi problem aayi. Kripya dobara try karein.", {}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENGINEER ASSIGNMENT  (mock; replace with real DB lookup)
# ══════════════════════════════════════════════════════════════════════════════

ENGINEER_POOL = [
    {"id": "ENG-101", "name": "Ramesh Kumar",  "phone": "919999900001", "cities": ["delhi", "noida", "gurgaon"]},
    {"id": "ENG-102", "name": "Suresh Singh",  "phone": "919999900002", "cities": ["mumbai", "pune", "thane"]},
    {"id": "ENG-103", "name": "Mahesh Yadav",  "phone": "919999900003", "cities": ["bangalore", "mysore"]},
    {"id": "ENG-104", "name": "Dinesh Patel",  "phone": "919999900004", "cities": ["ahmedabad", "surat", "vadodara"]},
    {"id": "ENG-999", "name": "Central Team",  "phone": "919999900099", "cities": []},  # fallback
]


def assign_engineer(service_city: str) -> dict:
    city_lower = service_city.lower()
    for eng in ENGINEER_POOL:
        if any(c in city_lower for c in eng["cities"]):
            return eng
    return ENGINEER_POOL[-1]  # fallback: central team


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TICKET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_ticket_id() -> str:
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"TKT-{suffix}"


def create_service_ticket(phone_number: str, context: dict) -> Tuple[str, dict]:
    """Build and persist a service ticket. Returns (ticket_id, ticket_data)."""
    engineer = assign_engineer(context.get("service_city", ""))
    ticket_id = generate_ticket_id()
    ticket_data = {
        "vehicle_no": context.get("vehicle_no"),
        "root_cause": context.get("root_cause"),
        "physical_intent": context.get("physical_intent", "OTHER"),
        "vehicle_location": context.get("location", "N/A"),
        "destination": context.get("destination", "N/A"),
        "service_city": context.get("service_city", "N/A"),
        "eta_date": context.get("eta_date", "N/A"),
        "contact_person": context.get("contact_person", "Manager"),
        "driver_phone": context.get("driver_phone", "N/A"),
        "engineer_id": engineer["id"],
        "engineer_name": engineer["name"],
        "engineer_phone": engineer["phone"],
        "assignment_status": "ASSIGNED",
        "created_at": datetime.now().isoformat(),
    }
    save_ticket(ticket_id, phone_number, ticket_data)
    logger.info("Ticket created: %s for vehicle %s", ticket_id, ticket_data["vehicle_no"])
    return ticket_id, ticket_data


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRE-ANALYSIS  (run before first message to customer)
# ══════════════════════════════════════════════════════════════════════════════

ROOT_CAUSE_RULES = {
    "BATTERY_ISSUE": lambda d: (
        d.get("main_powervoltage", 99) < 11.5 or
        str(d.get("ismainpoerconnected", "1")) == "0"  # note: typo in original field name preserved
    ),
    "MAIN_POWER_DISCONNECTED": lambda d: str(d.get("ismainpoerconnected", "1")) == "0",
    "GPS_INACTIVE": lambda d: d.get("gpsStatus") == 0,
}


def run_pre_analysis(gps_data: dict) -> str:
    """Determine root cause from latest GPS telemetry."""
    for cause, check in ROOT_CAUSE_RULES.items():
        try:
            if check(gps_data):
                return cause
        except Exception:
            pass
    return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class GpsData(BaseModel):
    gpstime: Optional[str] = None
    main_powervoltage: Optional[float] = None
    ismainpoerconnected: Optional[str] = None
    gpsStatus: Optional[int] = None
    driver_name: Optional[str] = None
    driver_phone: Optional[str] = None
    current_location: Optional[str] = None
    vehicle_state: Optional[str] = None


class TriggerPayload(BaseModel):
    phone_number: str
    vehicle_no: str
    last_location: Optional[str] = "Unknown"
    timestamp: Optional[str] = None
    gps_data: Optional[GpsData] = None


class WhatsAppWebhookMessage(BaseModel):
    phone_number: str
    message_text: str


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("GPS Chatbot service started. DB initialized.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — TRIGGER ENDPOINT  (called by your outage detection system)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/trigger-outage")
async def trigger_outage(payload: TriggerPayload):
    """
    Called when a vehicle has been offline for 24+ hours.
    Runs PRE_ANALYSIS, stores session, sends initial WhatsApp alert.
    """
    gps_info = payload.gps_data.model_dump() if payload.gps_data else {}

    # Feature 1: PRE_ANALYSIS before any customer message
    root_cause = run_pre_analysis(gps_info)
    logger.info("PRE_ANALYSIS for %s → root cause: %s", payload.vehicle_no, root_cause)

    d_name  = gps_info.get("driver_name")  or "N/A"
    d_phone = gps_info.get("driver_phone") or "N/A"
    last_loc = gps_info.get("current_location") or payload.last_location or "N/A"

    # Route to appropriate initial message
    if root_cause in ("BATTERY_ISSUE",):
        initial_msg = (
            "🚨 *GPS Connectivity Alert* 🚨\n\n"
            f"Vehicle *{payload.vehicle_no}* 24+ hours se offline hai.\n"
            "Hamare backend diagnostics se pata chala: *Battery Low / Charge-Discharge issue*.\n\n"
            "Tracking restore karne ke liye batayein:\n"
            "👉 Kya aap khud battery check/charge karwayenge, ya driver handle karega?"
        )
        first_state = "BATTERY_INITIAL_RESPONSE"
    elif root_cause == "MAIN_POWER_DISCONNECTED":
        initial_msg = (
            "🚨 *GPS Connectivity Alert* 🚨\n\n"
            f"Vehicle *{payload.vehicle_no}* offline hai.\n"
            "Diagnostics: *Main power line disconnected.*\n\n"
            "Kripya vehicle ka main power connection check karein aur reconnect karein.\n"
            "Kya aap ya driver ye check karenge?"
        )
        first_state = "MAIN_POWER_FLOW"
    else:
        initial_msg = (
            "🚨 *GPS Connectivity Alert* 🚨\n\n"
            f"Vehicle *{payload.vehicle_no}* 24+ hours se offline hai.\n"
            "Hum GPS status investigate kar rahe hain.\n\n"
            "Kripya vehicle ki current condition batayein."
        )
        first_state = "AWAITING_PHYSICAL_DIAGNOSIS"

    context_store = {
        "vehicle_no": payload.vehicle_no,
        "root_cause": root_cause,
        "driver_name": d_name,
        "driver_phone": d_phone,
        "last_location": last_loc,
        "physical_intent": None,
        "intent_locked": False,
        # service detail fields
        "location": None,
        "destination": None,
        "service_city": None,
        "eta_date": None,
        "contact_person": None,
    }

    save_session(
        phone_number=payload.phone_number,
        current_state=first_state,
        collected_json=context_store,
        chat_history=[{"role": "bot", "text": initial_msg}],
    )

    send_whatsapp_meta(payload.phone_number, initial_msg)
    logger.info("Outage flow triggered for %s | state: %s", payload.phone_number, first_state)
    return {"status": "triggered", "root_cause": root_cause, "initial_state": first_state}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — MAIN WEBHOOK  (all incoming WhatsApp replies)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/webhook")
async def handle_whatsapp_reply(webhook_data: WhatsAppWebhookMessage):
    """
    Central inbound handler. Implements the full feature checklist:
    - Entity extraction
    - LLM brain (guidance, confusion, side-questions, strategy change)
    - Driver session spin-up
    - Live GPS recheck
    - Intent lock
    - Ticket creation
    """
    user_phone   = webhook_data.phone_number
    incoming_text = webhook_data.message_text.strip()

    # ── Load session ──────────────────────────────────────────────────────────
    session = get_session(user_phone)
    if not session:
        logger.info("No active session for %s. Ignoring.", user_phone)
        return {"status": "ignored", "reason": "No active session."}

    current_state = session["current_state"]
    context       = session["collected_json"]
    history       = session["chat_history"]
    vehicle_no    = context.get("vehicle_no", "Unknown")

    logger.info("INCOMING [%s] state=%s | msg=%s", user_phone, current_state, incoming_text[:60])

    # ── Feature 5: Entity extraction ─────────────────────────────────────────
    entities = parse_inline_entities(incoming_text)

    # Merge freshly extracted entities into context (non-destructively)
    if entities.get("phone") and not context.get("driver_phone_override"):
        context["_extracted_phone"] = entities["phone"]
    if entities.get("name"):
        context["_extracted_name"] = entities["name"]
    if entities.get("date") and not context.get("eta_date"):
        context["eta_date"] = entities["date"]

    # ── Feature 6 & 4: LLM brain call ────────────────────────────────────────
    next_state, bot_reply, llm_updates = execute_llm_brain(
        current_state, incoming_text, context, history, entities
    )

    # Merge LLM-extracted updates into context
    for key, val in (llm_updates or {}).items():
        if val and val not in ("null", "N/A", ""):
            context[key] = val

    # Append turn to history
    history.append({"role": "user",  "text": incoming_text})
    history.append({"role": "bot",   "text": bot_reply})

    # ══════════════════════════════════════════════════════════════════════════
    # STATE-SPECIFIC SIDE EFFECTS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Driver session spin-up ────────────────────────────────────────────────
    #    Triggered when LLM moves to BATTERY_DRIVER_CONFIRMATION AND a phone
    #    number was extracted (owner is handing off to driver).
    if (
        next_state in ("BATTERY_DRIVER_CONFIRMATION", "BATTERY_WAITING_FOR_CHARGE")
        and entities.get("phone")
        and context.get("_extracted_phone")
    ):
        driver_phone = context["_extracted_phone"]
        driver_name  = context.get("_extracted_name") or context.get("driver_name", "Driver")

        # Update context with confirmed driver details
        context["driver_phone"] = driver_phone
        context["driver_name"]  = driver_name

        # Notify owner
        owner_ack = (
            f"Dhanyavaad. Hum {driver_name} ({driver_phone}) ko instructions bhej rahe hain.\n"
            "Battery charge hone ke baad GPS status verify kiya jayega."
        )
        send_whatsapp_meta(user_phone, owner_ack)
        history[-1]["text"] = owner_ack  # replace last bot reply

        # Owner session → TRANSITIONED_TO_DRIVER
        save_session(user_phone, "TRANSITIONED_TO_DRIVER", context, history)

        # Driver session
        driver_msg = (
            f"Hello, kya aap Vehicle *{vehicle_no}* ke sath hain?\n\n"
            "Is gaadi ka GPS offline hai — battery low detect hui hai.\n"
            "Kripya battery check karke charge karwayein aur hone ke baad\n"
            "yahan *'DONE'* reply karein."
        )
        driver_ctx = context.copy()
        save_session(
            phone_number=driver_phone,
            current_state="BATTERY_WAITING_FOR_CHARGE",
            collected_json=driver_ctx,
            chat_history=[{"role": "bot", "text": driver_msg}],
        )
        send_whatsapp_meta(driver_phone, driver_msg)

        logger.info("Driver session created for %s (vehicle %s)", driver_phone, vehicle_no)
        return {"status": "driver_session_created", "driver": driver_phone}

    # ── GPS live recheck after charge completion ──────────────────────────────
    if next_state == "BATTERY_POST_CHECK":
        send_whatsapp_meta(user_phone, "🔄 *Live GPS diagnostic chal rahi hai... please wait.*")
        api_result = run_hardware_api_recheck(vehicle_no)

        if api_result.get("gps_working"):
            success_msg = f"✅ *GPS Active!* Vehicle {vehicle_no} ka tracking normal hai. Issue resolve ho gaya. 🎉"
            send_whatsapp_meta(user_phone, success_msg)
            delete_session(user_phone)
            logger.info("Session closed — GPS verified for %s", vehicle_no)
            return {"status": "resolved", "vehicle": vehicle_no}

        elif api_result.get("main_power_connected") == "0":
            next_state = "MAIN_POWER_FLOW"
            bot_reply = (
                "⚠️ *Main Power Issue Detected*\n\n"
                "Battery theek lag rahi hai, lekin GPS ko vehicle se power nahi mil rahi.\n"
                "Main wiring connection check karna hoga.\n\n"
                "Kya aap ya driver wiring check karenge?"
            )
        else:
            next_state = "AWAITING_PHYSICAL_DIAGNOSIS"
            bot_reply = (
                "📊 Battery aur main power dono theek hain, lekin GPS abhi bhi offline hai.\n\n"
                "Iska matlab koi physical issue ho sakta hai.\n"
                "Kripya vehicle ki current condition describe karein\n"
                "(e.g., workshop mein hai, accident hua, device damaged dikh raha hai, etc.)"
            )

    # ── Intent lock after physical diagnosis ─────────────────────────────────
    if (
        next_state == "COLLECTING_SERVICE_DETAILS"
        and not context.get("intent_locked")
        and context.get("physical_intent")
    ):
        context["intent_locked"] = True
        logger.info("Intent locked for %s: %s", vehicle_no, context["physical_intent"])

    # ── Ticket creation ───────────────────────────────────────────────────────
    if next_state == "TICKET_CREATED":
        # Verify we have minimum fields; ask for missing ones
        required = ["location", "service_city"]
        missing  = [f for f in required if not context.get(f)]
        if missing:
            next_state = "COLLECTING_SERVICE_DETAILS"
            bot_reply = (
                "Almost done! Sirf ye details chahiye:\n"
                + "\n".join(f"• {f.replace('_', ' ').title()}" for f in missing)
                + "\nFormat: Location, Destination, Date (DD-MM-YYYY), Service City, Contact Name"
            )
        else:
            ticket_id, ticket_data = create_service_ticket(user_phone, context)
            bot_reply = (
                f"🎫 *Service Ticket Created!*\n\n"
                f"• *Ticket ID:* {ticket_id}\n"
                f"• *Issue:* {context.get('physical_intent', 'OTHER')}\n"
                f"• *Service City:* {ticket_data['service_city']}\n"
                f"• *Date:* {ticket_data['eta_date']}\n"
                f"• *Engineer:* {ticket_data['engineer_name']} ({ticket_data['engineer_phone']})\n\n"
                f"Engineer aapke contact person {ticket_data['contact_person']} se "
                f"arrival se pehle coordinate karenge."
            )

    # ── Session cleanup on completion ─────────────────────────────────────────
    if next_state in ("COMPLETED", "TICKET_CREATED") and not context.get("intent_locked") is False:
        send_whatsapp_meta(user_phone, bot_reply)
        delete_session(user_phone)
        logger.info("Session closed for %s | final state: %s", user_phone, next_state)
        return {"status": "completed", "final_state": next_state}

    # ── Persist and respond ───────────────────────────────────────────────────
    save_session(user_phone, next_state, context, history)
    send_whatsapp_meta(user_phone, bot_reply)

    logger.info("State transition [%s]: %s → %s", user_phone, current_state, next_state)
    return {"status": "processed", "transition": f"{current_state} → {next_state}"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "GPS AI Support Chatbot"}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — COMPATIBILITY ALIASES
# main.py imports these exact names:
#   from battery_issue_flow import start_battery_flow, handle_whatsapp_replies, WhatsAppWebhookMessage
# ══════════════════════════════════════════════════════════════════════════════

async def start_battery_flow(payload: dict):
    """
    Alias called by main.py routing logic when root_cause == BATTERY_ISSUE.
    Converts the raw dict payload from the central router into a TriggerPayload
    and delegates to trigger_outage().
    """
    from fastapi import Request as _Request

    gps_raw = payload.get("gps_data") or {}
    gps_model = GpsData(
        gpstime=gps_raw.get("gpstime"),
        main_powervoltage=gps_raw.get("main_powervoltage"),
        ismainpoerconnected=gps_raw.get("ismainpoerconnected"),
        gpsStatus=gps_raw.get("gpsStatus"),
        driver_name=gps_raw.get("driver_name"),
        driver_phone=gps_raw.get("driver_phone"),
        current_location=gps_raw.get("current_location"),
        vehicle_state=gps_raw.get("vehicle_state"),
    ) if gps_raw else None

    trigger = TriggerPayload(
        phone_number=payload["phone_number"],
        vehicle_no=payload["vehicle_no"],
        last_location=payload.get("last_location", "Unknown"),
        timestamp=payload.get("timestamp"),
        gps_data=gps_model,
    )
    return await trigger_outage(trigger)


async def handle_whatsapp_replies(webhook_data: WhatsAppWebhookMessage):
    """
    Alias called by main.py webhook router when root_cause == BATTERY_ISSUE.
    Delegates directly to handle_whatsapp_reply().
    """
    return await handle_whatsapp_reply(webhook_data)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("battery_issue_flow:app", host="0.0.0.0", port=8000, reload=True)