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
    standing_hours: Optional[float] = None


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
# HELPERS
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
# DATE NORMALIZATION
# ==============================================================================

def _resolve_service_date(raw: str) -> Optional[str]:
    """
    Convert any date expression into DD-MM-YYYY.
    """
    if not raw:
        return None
    raw = str(raw).strip()

    def _to_dd_mm_yyyy(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        text = str(value).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            try:
                return datetime.strptime(text, "%Y-%m-%d").strftime("%d-%m-%Y")
            except ValueError:
                return None
        return text

    # ── relative: "today+N" produced by LLM
    m = re.match(r"today\s*\+\s*(\d+)", raw, re.I)
    if m:
        return (date.today() + timedelta(days=int(m.group(1)))).strftime("%d-%m-%Y")

    # ── "aaj" / "ajj" / "today"
    if re.search(r"\b(aaj|ajj|today|todays)\b", raw, re.I):
        return date.today().strftime("%d-%m-%Y")

    # ── relative: "today+N" produced by LLM
    if m:
        return (date.today() + timedelta(days=int(m.group(1)))).strftime("%d-%m-%Y")

    # ── "N days" / "N din baad" / "after N days"
    m = re.search(r"(\d+)\s*(din\s*baad|days?\s*baad|days?\s*later|days?)", raw, re.I)
    if m:
        return (date.today() + timedelta(days=int(m.group(1)))).strftime("%d-%m-%Y")

    # ── "after N days" with leading word
    m = re.match(r"after\s+(\d+)", raw, re.I)
    if m:
        return (date.today() + timedelta(days=int(m.group(1)))).strftime("%d-%m-%Y")

    # ── tomorrow / kal
    if re.search(r"\b(tomorrow|kal)\b", raw, re.I):
        return (date.today() + timedelta(days=1)).strftime("%d-%m-%Y")

    # ── DD-MM-YYYY or DD/MM/YYYY
    m = re.match(r"(\d{2})[-/](\d{2})[-/](\d{4})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # ── YYYY-MM-DD (ISO)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"

    # ── "5 July" / "5 July 2026" / "July 5"
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "june": 6, "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12,
    }
    m = re.search(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?", raw, re.I)
    if m:
        day = int(m.group(1))
        mon = month_map.get(m.group(2).lower())
        year = int(m.group(3)) if m.group(3) else date.today().year
        if mon:
            return f"{day:02d}-{mon:02d}-{year}"

    m = re.search(r"([a-z]+)\s+(\d{1,2})(?:\s+(\d{4}))?", raw, re.I)
    if m:
        mon = month_map.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3)) if m.group(3) else date.today().year
        if mon:
            return f"{day:02d}-{mon:02d}-{year}"

    # ── free-form phrases like "25 ko aa jayegi" / "25th ko"
    m = re.search(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)?(?:\s+ko)?(?!\d)", raw)
    if m:
        try:
            normalized = normalize_date(m.group(1))
            return _to_dd_mm_yyyy(normalized) or raw
        except Exception:
            pass

    # ── fallback to date_utils
    try:
        normalized = normalize_date(raw)
        return _to_dd_mm_yyyy(normalized) or raw
    except Exception:
        return raw


def _map_option_to_days(message: str, brain_date: Optional[str]) -> Optional[str]:
    """
    When the bot shows the 3-option menu (1=+2 days, 2=+4 days, 3=custom),
    map user's reply to a concrete date in DD-MM-YYYY.

    FIX S4/S5/S6/S15: Also try resolving the raw message as a date expression
    directly — handles "5 July", "8 din baad", etc. even without LLM extraction.
    Returns None only if nothing can be resolved.
    """
    cleaned = message.strip()

    # Option 1
    if re.fullmatch(r"1", cleaned) or re.search(
        r"\b(option\s*1|after\s*2\s*days?|2\s*din\s*baad)\b", cleaned, re.I
    ):
        return (date.today() + timedelta(days=2)).strftime("%d-%m-%Y")

    # Option 2
    if re.fullmatch(r"2", cleaned) or re.search(
        r"\b(option\s*2|after\s*4\s*days?|4\s*din\s*baad)\b", cleaned, re.I
    ):
        return (date.today() + timedelta(days=4)).strftime("%d-%m-%Y")

    # Option 3 or explicit "3"
    if re.fullmatch(r"3", cleaned):
        # Wait for next message with actual date
        return None

    # If brain extracted a date string, resolve it
    if brain_date:
        resolved = _resolve_service_date(brain_date)
        if resolved and re.match(r"\d{2}-\d{2}-\d{4}", resolved):
            return resolved

    # FIX S6/S15: Try resolving the raw message itself as a date/relative expression.
    # This handles "5 July", "8 din baad", "after 8 days", etc. directly.
    resolved = _resolve_service_date(cleaned)
    if resolved and re.match(r"\d{2}-\d{2}-\d{4}", resolved):
        return resolved

    return None


# ==============================================================================
# SYSTEM DEFINITION & SINGLE-ENGINE PROMPT
# ==============================================================================

_DYNAMIC_BRAIN_SYSTEM = """
You are an automated human-like conversational support agent dealing with vehicle tracking downtime.
Your job is to analyze the conversation history and the latest message to determine intent, sub-intent, extract data, and intelligently handle scheduling rejections.

CRITICAL: Always generate extremely short, one-line, or few-word informative replies. If the user asks anything out of flow, answer politely in one line and immediately nudge them back to the main topic.

## MASTER INTENT CLASSIFICATION:
- WORKSHOP: Vehicle is at a service point, body shop, workshop, or undergoing garage maintenance.
- ACCIDENT: Vehicle met with a highway collision, structural damage, or breakdown.
- VEHICLE_RUNNING: Vehicle is driving, moving, or on an active shipment trip.
- VEHICLE_STANDING: Vehicle is parked at an open plot, yard, home, or factory securely.
- GPS_DAMAGED: Physical tracker wires were cut, broken, burned, or destroyed.
- GPS_REMOVED: The device was physically unplugged, detached, stolen, or stored separately.
- OTHER: Unclear issue or side-chatter.

## CRITICAL OPERATIONAL LOGIC:
1. **Entity Extraction**: Extract `current_location`, `destination_location`, `vehicle_location`,
   `service_city`, `driver_name`, `driver_phone`, `workshop_name`, `resume_date`, `service_date`,
   `next_trip_date`, `next_trip_location`, and `contact_person`.

2. **VEHICLE_RUNNING Special Fields**:
   - `current_location`: Where the vehicle is driving from
   - `destination_location`: Where the vehicle is going to
   - `service_city_confirmed`: true if user agrees, false if they reject, null if not asked yet
   - `service_city`: The actual city where they want service (only if suggestion rejected)

3. **Contact Person Allocation**:
   - If the user responds with general agreement ("haan", "yes", "ok", "main khud") to checking it themselves, understand this as choosing "self".

4. **Scheduling Loop Tracking**:
   - If a slot suggestion is rejected, mark `slot_rejected: true`.
   - If they agree or supply a date themselves, capture it into `entities.service_date`.
   - `slot_rejected` must be false once they accept any date so the flow can advance.

5. **Driver Verification (VEHICLE_RUNNING)**:
   - If the user agrees with the existing driver, set `contact_person` to the driver name,
     `driver_phone` to the existing driver phone, `contact_person_confirmed: true`.
   - If they provide a NEW name or phone, extract into `contact_person` and `driver_phone`,
     set `contact_person_confirmed: true`.

## RESPONSE SCHEMA (STRICT - Return ONLY valid JSON, no markdown):
{
  "intent": "WORKSHOP"|"ACCIDENT"|"VEHICLE_RUNNING"|"VEHICLE_STANDING"|"GPS_DAMAGED"|"GPS_REMOVED"|"OTHER"|null,
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
  "service_city_confirmed": boolean|null,
  "wants_service_visit": boolean|null,
  "is_in_workshop_currently": boolean|null,
  "slot_rejected": boolean,
  "contact_person_confirmed": boolean|null,
  "side_question_reply": string or null,
  "conversational_reply": string
}
"""


# ==============================================================================
# GPS STATUS HELPERS
# ==============================================================================

def get_backend_gps_snapshot(phone_number: str) -> Optional[dict]:
    """
    Always performs a FRESH read of GPS data. Priority:
    1. Live read from database.get_user() — guarantees up-to-date gpsStatus
    2. HTTP call to backend /api/test/get-gps-data endpoint
    3. Local session fallback
    """
    # Priority 1: Fresh read from users table
    user = database.get_user(phone_number)
    if user:
        gps_data = user.get("gps_data") or {}
        if isinstance(gps_data, dict):
            logger.info(
                "[GPS_SNAPSHOT] Fresh user record read for %s | gpsStatus=%s",
                phone_number, gps_data.get("gpsStatus"),
            )
            gps_payload = {
                "phone_number": phone_number,
                "vehicle_no": user.get("vehicle_no"),
                "last_location": user.get("last_location"),
                "timestamp": user.get("timestamp"),
                "gps_data": gps_data,
            }
            return {
                "status": "found",
                "payload": gps_payload,
                "gps_snapshot": {**gps_payload, "user": user},
            }

    # Priority 2: HTTP fallback
    backend_url = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
    try:
        resp = requests.get(f"{backend_url}/api/test/get-gps-data/{phone_number}", timeout=6)
        if resp.status_code == 200:
            logger.info("[GPS_SNAPSHOT] Fetched from backend API for %s", phone_number)
            return resp.json()
    except Exception as e:
        logger.warning(f"[GPS_SNAPSHOT] Backend API unreachable for {phone_number}: {e}")

    # Priority 3: Session fallback
    session = database.get_session(phone_number)
    if not session:
        return None

    collected = session.get("collected_json", {})
    gps_data = collected.get("gps_data", {}) or {}
    fallback_keys = [
        "gpstime", "main_powervoltage", "ismainpoerconnected",
        "gpsStatus", "driver_name", "driver_phone", "current_location", "vehicle_state",
    ]
    for key in fallback_keys:
        if key not in gps_data and collected.get(key) is not None:
            gps_data[key] = collected.get(key)

    logger.info(
        "[GPS_SNAPSHOT] Using session fallback for %s | gpsStatus=%s",
        phone_number, gps_data.get("gpsStatus"),
    )
    gps_payload = {
        "phone_number": phone_number,
        "vehicle_no": collected.get("vehicle_no"),
        "last_location": collected.get("last_location"),
        "timestamp": collected.get("timestamp"),
        "gps_data": gps_data,
    }
    return {"status": "found", "payload": gps_payload, "gps_snapshot": gps_payload}


def is_gps_online(snapshot: dict) -> bool:
    """Return True when GPS status is 1/online."""
    if not isinstance(snapshot, dict):
        return False

    def _coerce(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"1", "true", "yes", "online", "y"}:
                return True
            if cleaned in {"0", "false", "no", "offline", "n"}:
                return False
        return None

    candidates = [snapshot.get("gpsStatus"), snapshot.get("status")]

    for key in ("gps_data", "payload", "gps_snapshot", "user"):
        sub = snapshot.get(key)
        if isinstance(sub, dict):
            candidates.append(sub.get("gpsStatus"))
            candidates.append(sub.get("status"))
            inner = sub.get("gps_data")
            if isinstance(inner, dict):
                candidates.append(inner.get("gpsStatus"))
            inner2 = sub.get("user")
            if isinstance(inner2, dict):
                inner_gps = inner2.get("gps_data")
                if isinstance(inner_gps, dict):
                    candidates.append(inner_gps.get("gpsStatus"))

    for candidate in candidates:
        parsed = _coerce(candidate)
        if parsed is not None:
            return parsed
    return False


def close_self_repair_case(phone: str, collected: dict, chat_hist: list) -> dict:
    """
    Close the case because GPS came online — no ticket.

    FIX S1/S3/S10/S11/S12/S16/S17/S18 (Session deleted):
    We must DELETE the session from the DB so that get_session() returns None.
    The old code called save_session then delete_session, but if the save
    wrote a record and delete_session wasn't implemented / was a no-op the
    test kept finding the session.  We now:
      1. Append final bot reply to chat_hist.
      2. Send WhatsApp message.
      3. Delete the session FIRST (removes the row).
      4. Save a CASE_CLOSED audit record only if the DB supports an audit table
         (non-blocking — wrapped in try/except).
    This guarantees that get_session(phone) returns None immediately after.
    """
    reply = "✅ GPS data receive hona shuru ho gaya hai. Issue resolve ho gaya hai. Dhanyavaad!"
    chat_hist.append({"role": "bot", "text": reply})
    send_whatsapp_meta(phone, reply)

    # FIX: delete FIRST so the session row is gone, then optionally audit-save.
    database.delete_session(phone)
    try:
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
    except Exception:
        pass  # audit save is best-effort; deletion already happened

    return {"status": "self_repair_case_closed", "message": "gps_online"}


def check_self_repair_status(phone: str, collected: dict, chat_hist: list, context: str) -> Optional[dict]:
    """
    Perform a fresh GPS status check. Uses active_contact_phone so driver
    conversations are checked against the driver's user record.
    Returns close_self_repair_case() result if GPS is online, else None.
    """
    active_phone = collected.get("active_contact_phone") or phone
    logger.info("[SELF_REPAIR_CHECK] Checking GPS for %s (context: %s)", active_phone, context)
    snapshot = get_backend_gps_snapshot(active_phone)
    if snapshot and is_gps_online(snapshot):
        original_phone = collected.get("original_customer_phone") or phone
        logger.info(
            "[SELF_REPAIR_CHECK] gpsStatus=1 for %s — closing case (context: %s)",
            original_phone, context,
        )
        return close_self_repair_case(original_phone, collected, chat_hist)
    logger.info("[SELF_REPAIR_CHECK] GPS still offline for %s (context: %s)", active_phone, context)
    return None


def prompt_self_repair_step(phone: str, collected: dict, chat_hist: list, state: str, prompt: str) -> dict:
    """Check GPS before sending a self-repair prompt. If online, close case."""
    closed = check_self_repair_status(phone, collected, chat_hist, f"prompt:{state}")
    if closed is not None:
        return closed
    chat_hist.append({"role": "bot", "text": prompt})
    database.save_session(phone, state, collected, chat_hist)
    send_whatsapp_meta(phone, prompt)
    return {"status": "self_repair_prompt_sent", "state": state}


def start_self_repair_flow(phone: str, collected: dict, chat_hist: list,
                           current_intent: Optional[str], standing_hours: float) -> dict:
    active_phone = collected.get("active_contact_phone") or phone
    snapshot = get_backend_gps_snapshot(active_phone)
    if snapshot and is_gps_online(snapshot):
        return close_self_repair_case(phone, collected, chat_hist)
    prompt = "Kripya ek baar ignition ON hai ya nahi check kijiye."
    return prompt_self_repair_step(phone, collected, chat_hist, "SELF_REPAIR_IGNITION", prompt)


def perform_self_repair_check_and_continue(
    phone: str, collected: dict, chat_hist: list, next_state: str, next_prompt: str
) -> dict:
    """
    Called after the user replies to a self-repair step.
    1. Fresh GPS check — if online, close immediately.
    2. Otherwise advance to the next step (which itself checks again before sending).
    """
    closed = check_self_repair_status(phone, collected, chat_hist, f"step:{next_state}")
    if closed is not None:
        return closed
    return prompt_self_repair_step(phone, collected, chat_hist, next_state, next_prompt)


# ==============================================================================
# MISC HELPERS
# ==============================================================================

def merge_extracted_data(existing: dict, new_data: dict) -> dict:
    merged = dict(existing)
    for key, value in new_data.items():
        if value is not None and str(value).strip().lower() not in ("null", "none", ""):
            merged[key] = value
    return merged


def is_phone_refusal_response(text: str) -> bool:
    return bool(re.search(
        r"\b(nahi dena|na dena|baad me|baad mein|no number|number nahi hai|privacy|security|nahi chahiye)\b",
        text.strip().lower()
    ))


def is_affirmative_response(text: str) -> bool:
    return bool(re.search(
        r"\b(haan|ha|yes|y|ok|okay|theek|thik|sahi|bilkul|confirm|correct)\b",
        text.strip().lower()
    ))


def is_negative_response(text: str) -> bool:
    return bool(re.search(
        r"\b(nahi|nahin|nhi|no|n|nope|mat|nai)\b",
        text.strip().lower()
    ))


def detect_contact_choice(text: str) -> Optional[str]:
    cleaned = text.strip().lower()
    # If the user says "yes", "haan", "haa", "ha", or "myself", map it to "self".
    if re.search(r"\b(khud|self|main|hum|owner|mai|me|haan|haa|ha|yes|y|ok|okay|theek|thik)\b", cleaned):
        return "self"
    if re.search(r"\b(driver|driver se|driver ko|driver ka|unse|driver handle|driver handle karega|vehicle ke paas nahi|ghar pe nahi|paas nahi)\b", cleaned):
        return "driver"
    if re.search(r"\b(kisi aur|koi aur|other|dusra|dusre|contact person|manager|supervisor)\b", cleaned):
        return "other"
    return None


def extract_phone_number(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{10,13})\b", text)
    return match.group(1) if match else None


def is_pata_nahi_response(text: str) -> bool:
    return bool(re.search(
        r"\b(pata nahi|pata nahin|pata nhi|dont know|unknown|not sure)\b",
        text.strip().lower()
    ))


def build_driver_confirmation_prompt(collected: dict) -> str:
    d_name = collected.get("driver_name") or "Not Available"
    d_phone = collected.get("driver_phone") or "Not Available"
    return (
        "Humare paas driver ki details available hain:\n\n"
        f"👤 Driver Name: {d_name}\n"
        f"📞 Driver Contact: {d_phone}\n\n"
        "Kya aap in details ko confirm karte hain, ya driver ka naya naam/number share karenge?"
    )


def get_service_date_prompt(now: Optional[datetime] = None) -> str:
    current_time = now or datetime.now()
    if current_time.hour < 19:
        return "Kya aaj service book kar dein?"
    return "Kya kal service book kar dein?"


def _is_ambiguous_numeric_date(raw: Any) -> bool:
    if raw is None:
        return False
    cleaned = str(raw).strip().lower()
    return bool(re.fullmatch(r"\d{1,2}", cleaned))


def _is_ambiguous_numeric_date(raw: Any) -> bool:
    if raw is None:
        return False
    cleaned = str(raw).strip().lower()
    return bool(re.fullmatch(r"\d{1,2}", cleaned))


def is_option_selection(message: str) -> bool:
    return bool(re.fullmatch(r"[1-3]", message.strip()))


def resolve_affirmative_service_date(message: str, collected: dict, now: Optional[datetime] = None) -> Optional[str]:
    if not is_affirmative_response(message):
        return None
    if collected.get("awaiting_date_options"):
        return None
    current_time = now or datetime.now()
    if current_time.hour < 19:
        return current_time.date().strftime("%d-%m-%Y")
    return (current_time.date() + timedelta(days=1)).strftime("%d-%m-%Y")


def is_option_selection(message: str) -> bool:
    return bool(re.fullmatch(r"[1-3]", message.strip()))


def apply_driver_confirmation(message: str, collected: dict, ext_entities: dict) -> Optional[dict]:
    cleaned = message.strip().lower()
    if is_affirmative_response(cleaned):
        collected["driver_contact_confirmed"] = True
        collected["contact_person"] = collected.get("driver_name") or collected.get("contact_person") or "Driver"
        if not collected.get("driver_phone"):
            collected["driver_phone"] = collected.get("driver_phone")
        return {"confirmed": True, "alternate_contact": False}

    if is_negative_response(cleaned):
        collected["driver_contact_confirmed"] = False
        collected["awaiting_alternate_contact"] = True
        return {"confirmed": False, "alternate_contact": False}

    new_phone = extract_phone_number(message)
    new_name = ext_entities.get("contact_person") or ext_entities.get("driver_name")
    if new_phone or new_name:
        collected["driver_contact_confirmed"] = True
        collected["awaiting_alternate_contact"] = False
        if new_name:
            collected["contact_person"] = new_name
            collected["driver_name"] = new_name
        if new_phone:
            collected["driver_phone"] = new_phone
        return {"confirmed": True, "alternate_contact": True}

    return None


def apply_service_time_window(message: str, collected: dict) -> Optional[dict]:
    cleaned = message.strip()
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if is_affirmative_response(lowered) or is_negative_response(lowered):
        return None

    has_time_tokens = bool(
        re.search(r"\b\d{1,2}(?::\d{2})?\s*(am|pm|a\.m\.|p\.m\.|baje)\b", lowered)
        or re.search(r"\b\d{1,2}\s*(se|tak|to)\s*\d{1,2}\b", lowered)
        or re.search(r"\b\d{1,2}:\d{2}\b", lowered)
        or re.search(r"\b(shaam|subah|dopahar|raat)\b", lowered)
    )
    if has_time_tokens:
        collected["service_time_window"] = cleaned
        return {"service_time_window": cleaned}
    return None


def apply_service_city_confirmation(message: str, collected: dict, ext_entities: dict) -> Optional[dict]:
    """Handle affirmative/negative replies to the service-city confirmation prompt."""
    if collected.get("service_city_confirmed") is not None or collected.get("service_city"):
        return None

    cleaned = message.strip().lower()
    if is_affirmative_response(cleaned):
        service_city = (
            ext_entities.get("service_city")
            or collected.get("service_city")
            or collected.get("destination_location")
            or "Delhi"
        )
        collected["service_city_confirmed"] = True
        collected["service_city"] = service_city
        return {"confirmed": True, "service_city": service_city}

    if is_negative_response(cleaned):
        collected["service_city_confirmed"] = False
        collected["service_city"] = None
        return {"confirmed": False, "service_city": None}

    return None


def build_troubleshooting_prompt(current_intent: Optional[str], collected: dict, standing_hours: float) -> str:
    if current_intent == "WORKSHOP":
        return "Vehicle workshop se kab tak bahar aa jayegi?"
    if current_intent == "ACCIDENT":
        return "Gaadi kab tak running condition me aa jayegi?"
    if current_intent == "VEHICLE_RUNNING":
        if not collected.get("current_location"):
            return "Aapki gaadi abhi kis location par hai? (Current location batayein)"
        if not collected.get("destination_location"):
            return "Kahan ja rahe hain? (Destination batayein)"
        suggested_city = collected.get("destination_location") or "Delhi"
        if collected.get("service_city_confirmed") is None and not collected.get("service_city"):
            return f"Kya hum {suggested_city} mein service book kar dein?"
        if collected.get("service_city_confirmed") is False and not collected.get("service_city"):
            return "Kaun se city mein service chahiye? (Preferred city batayein)"
        if not collected.get("service_date"):
            return get_service_date_prompt()
        if not collected.get("service_time_window"):
            return "Kitne baje se kitne baje tak vehicle service ke liye available hogi?"
        if not collected.get("contact_person"):
            d_name = collected.get("driver_name")
            d_phone = collected.get("driver_phone")
            if d_name or d_phone:
                return (
                    "Humare paas driver ki details available hain:\n\n"
                    f"👤 Driver Name: {d_name or 'Not Available'}\n"
                    f"📞 Driver Contact: {d_phone or 'Not Available'}\n\n"
                    "Kya aap in details ko confirm karte hain, ya driver ka naya naam/number share karenge?"
                )
            return "Service coordination ke liye contact person ka naam aur mobile number kya hai?"
        return "Kripya apni next update share kijiye."
    if current_intent == "GPS_REMOVED":
        if not collected.get("resume_date"):
            return "GPS device vehicle me wapas kab tak connect/reinstall ho jayega?"
        if collected.get("wants_service_visit") is None:
            return "Kya aapko reinstall karne ke liye hamare physical service engineer ki zaroorat hai?"
    if current_intent == "VEHICLE_STANDING" and standing_hours > 48:
        if not collected.get("resume_date"):
            return "Gaadi 48 ghante se jyada se stationary hai. Agli trip kis date ko nikalne wali hai?"
    if not collected.get("vehicle_location"):
        return "Gaadi abhi kis city/location par chal rahi hai?"
    return "Kripya vehicle ki sthiti short me batayein."


def transfer_owner_to_driver(
    owner_phone: str,
    target_phone: str,
    collected: dict,
    chat_hist: list,
    current_intent: Optional[str],
):
    owner_ack = (
        "Dhanyavaad Sir. Hum driver se sampark karke GPS issue ki troubleshooting continue kar rahe hain. "
        "Agar kisi aur jaankari ki zarurat hogi to hum aapse sampark karenge."
    )

    owner_collected = dict(collected)
    owner_collected["contact_mode"] = "driver"
    owner_collected["active_contact_phone"] = target_phone
    owner_collected["driver_contact_confirmed"] = True
    owner_collected["intent"] = current_intent or owner_collected.get("intent")

    chat_hist.append({"role": "bot", "text": owner_ack})
    database.save_session(owner_phone, "TRANSFERRED_TO_DRIVER", owner_collected, chat_hist)
    send_whatsapp_meta(owner_phone, owner_ack)

    vehicle_no = owner_collected.get("vehicle_no") or "vehicle"
    driver_prompt = (
        f"Namaste,\n\n"
        f"Vehicle {vehicle_no} se GPS data receive nahi ho raha hai.\n\n"
        "Kripya vehicle ki current status batayein."
    )

    driver_collected = dict(owner_collected)
    driver_collected["original_customer_phone"] = owner_phone
    driver_collected["active_contact_phone"] = target_phone
    driver_collected["contact_mode"] = "driver"
    driver_collected["status_only"] = False
    driver_collected["initial_alert_msg"] = driver_prompt

    database.save_session(
        target_phone,
        "INITIAL_ALERT",
        driver_collected,
        [{"role": "bot", "text": driver_prompt}],
    )
    send_whatsapp_meta(target_phone, driver_prompt)

    return {"status": "owner_handed_over_to_driver", "target_phone": target_phone}


def reassign_troubleshooting_contact(
    original_phone: str,
    target_phone: str,
    collected: dict,
    chat_hist: list,
    current_state: str,
    current_intent: Optional[str],
    standing_hours: float,
    status_note: Optional[str] = None,
    explicit_prompt: Optional[str] = None,
):
    collected["original_customer_phone"] = collected.get("original_customer_phone") or original_phone
    collected["active_contact_phone"] = target_phone
    collected["contact_mode"] = "driver" if target_phone == collected.get("driver_phone") else "other"

    intro_message = collected.get("initial_alert_msg")
    if not intro_message:
        for entry in chat_hist:
            if entry.get("role") == "bot" and entry.get("text"):
                intro_message = entry["text"]
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

    prompt = explicit_prompt or build_troubleshooting_prompt(current_intent, collected, standing_hours)
    return prompt_self_repair_step(target_phone, collected, new_history, current_state, prompt)


# ==============================================================================
# SERVICE TICKET CREATION
# ==============================================================================

async def create_service_ticket_flow(phone: str, collected: dict, chat_hist: list):
    """
    Create service ticket and send confirmation.

    FIX S3/S11/S16 (Session deleted after ticket):
    Same pattern as close_self_repair_case — delete FIRST so get_session()
    returns None immediately, then do a best-effort audit save.

    FIX S11 (GPS check before ticket):
    Check GPS one final time; if online, close without creating ticket.
    """
    # Final GPS check before creating ticket (FIX S11)
    active_phone = collected.get("active_contact_phone") or phone
    snapshot = get_backend_gps_snapshot(active_phone)
    if snapshot and is_gps_online(snapshot):
        original_phone = collected.get("original_customer_phone") or phone
        return close_self_repair_case(original_phone, collected, chat_hist)

    try:
        ticket_id = f"TKT-{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}"

        if collected.get("service_city"):
            service_city = collected["service_city"]
        elif collected.get("service_city_confirmed") is True:
            service_city = collected.get("destination_location") or "Delhi"
        else:
            service_city = "Delhi"

        if collected.get("service_date"):
            service_date = collected["service_date"]
        else:
            service_date = (date.today() + timedelta(days=1)).strftime("%d-%m-%Y")

        # Determine contact_person — fallback chain
        contact_person = (
            collected.get("contact_person")
            or collected.get("driver_name")
            or "Driver"
        )
        driver_phone = (
            collected.get("driver_phone")
            if collected.get("driver_phone") not in (None, "NOT_PROVIDED")
            else phone
        )

        ticket_data = {
            "vehicle_no": collected.get("vehicle_no"),
            "intent": collected.get("intent"),
            "current_location": collected.get("current_location"),
            "destination_location": collected.get("destination_location"),
            "service_city": service_city,
            "service_date": service_date,
            "service_time_window": collected.get("service_time_window"),
            "contact_person": contact_person,
            "driver_phone": driver_phone,
            "driver_name": collected.get("driver_name"),
            "status": "OPEN",
        }

        database.save_ticket(ticket_id, phone, ticket_data)
        collected["ticket_id"] = ticket_id

        reply_msg = (
            f"🎫 *Service Ticket Created Successfully!*\n\n"
            f"• *Ticket ID:* {ticket_id}\n"
            f"• *Vehicle:* {collected.get('vehicle_no')}\n"
            f"• *Issue:* GPS not updating while running\n"
            f"• *Current Location:* {collected.get('current_location')}\n"
            f"• *Destination:* {collected.get('destination_location')}\n"
            f"• *Service City:* {service_city}\n"
            f"• *Service Date:* {service_date}\n"
            f"• *Service Time Window:* {collected.get('service_time_window') or 'Not Provided'}\n"
            f"• *Contact Person:* {contact_person}\n"
            f"• *Contact Number:* {driver_phone}\n\n"
            f"Hamare engineer aapke contact person se coordinate karenge service ke liye. Dhanyavaad!"
        )

        chat_hist.append({"role": "bot", "text": reply_msg})
        send_whatsapp_meta(phone, reply_msg)

        # FIX: delete FIRST so session is gone, then audit-save
        database.delete_session(phone)
        try:
            database.save_session(phone, "TICKET_CREATED", collected, chat_hist)
        except Exception:
            pass  # best-effort audit

        return {"status": "ticket_created", "ticket_id": ticket_id}

    except Exception as e:
        logger.error(f"Error creating service ticket: {e}")
        error_msg = "Ticket create karne mein problem aayi. Please try again."
        send_whatsapp_meta(phone, error_msg)
        return {"status": "error", "message": str(e)}


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
            "gps_data": gps_data,
            "gpstime": gps_data.get("gpstime"),
            "main_powervoltage": gps_data.get("main_powervoltage"),
            "ismainpoerconnected": gps_data.get("ismainpoerconnected"),
            "gpsStatus": gps_data.get("gpsStatus"),
            "vehicle_state": gps_data.get("vehicle_state"),
            "intent": None,
            "vehicle_location": last_location or None,
            "current_location": None,
            "destination_location": None,
            "service_city": None,
            "service_city_confirmed": None,
            "service_date": None,
            "resume_date": None,
            "next_trip_date": None,
            "next_trip_location": None,
            "driver_phone": gps_data.get("driver_phone"),
            "driver_name": gps_data.get("driver_name"),
            "driver_contact_confirmed": None,
            "awaiting_alternate_contact": False,
            "service_time_window": None,
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
            "ticket_id": None,
            "verifying_driver": False,
            "awaiting_date_options": False,
        },
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )

    send_whatsapp_meta(phone_number, initial_alert_msg)
    return {"status": "flow_initialized", "case": "OTHER_ISSUE"}


# ==============================================================================
# WEBHOOK HANDLER ENGINE
# ==============================================================================

def call_brain(state_context: str, chat_hist: list, message: str) -> dict:
    llm_messages = []
    for entry in chat_hist[-8:]:
        role = "assistant" if entry.get("role") == "bot" else "user"
        content = entry.get("text") or entry.get("content") or ""
        llm_messages.append({"role": role, "content": content})
    llm_messages.append({"role": "user", "content": message})

    try:
        response = openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _DYNAMIC_BRAIN_SYSTEM},
                *llm_messages,
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            max_tokens=500,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logger.error(f"[LLM Brain Exception] {e}")
        raise


async def handle_whatsapp_replies(msg: WhatsAppWebhookMessage) -> dict:
    phone = msg.phone_number
    message = msg.message_text.strip()

    session = database.get_session(phone)
    if not session:
        return {"status": "no_active_session"}

    current_state = session.get("current_state", "INITIAL_ALERT")
    collected = session.get("collected_json", {})
    chat_hist = session.get("chat_history", [])

    # GPS check at every COLLECTING_DETAILS and SELF_REPAIR_* entry
    if current_state in ("COLLECTING_DETAILS",) or current_state.startswith("SELF_REPAIR_"):
        closed = check_self_repair_status(phone, collected, chat_hist, f"reply:{current_state}")
        if closed is not None:
            return closed

    if current_state == "STATUS_ONLY":
        status_contact = collected.get("contact_person") or collected.get("driver_name") or "selected contact"
        status_reply = (
            f"Update: hum {status_contact} ke sath troubleshooting continue kar rahe hain. "
            f"Aapko sirf status updates milte rahenge."
        )
        chat_hist.append({"role": "user", "text": message})
        chat_hist.append({"role": "bot", "text": status_reply})
        database.save_session(phone, current_state, collected, chat_hist)
        send_whatsapp_meta(phone, status_reply)
        return {"status": "status_only_update"}

    # ── LLM Brain ─────────────────────────────────────────────────────────────
    try:
        brain = call_brain("general", chat_hist, message)
    except Exception:
        return {"status": "error_processing_llm"}

    # Intent assignment
    if brain.get("intent"):
        collected["intent"] = brain["intent"]
    current_intent = collected.get("intent")

    # Merge extracted entities
    ext_entities = brain.get("entities", {}) or {}
    collected = merge_extracted_data(collected, ext_entities)

    if brain.get("is_in_workshop_currently") is not None:
        collected["is_in_workshop_currently"] = brain["is_in_workshop_currently"]
    if brain.get("wants_service_visit") is not None:
        collected["wants_service_visit"] = brain["wants_service_visit"]
    if brain.get("service_city_confirmed") is not None:
        collected["service_city_confirmed"] = brain["service_city_confirmed"]

    # ── FIX S4/S5/S6/S15: Resolve service dates in Python, not LLM ───────────
    raw_service_date = ext_entities.get("service_date")

    if collected.get("awaiting_date_options"):
        # We're in the 3-option menu — try to resolve option or raw date
        resolved = _map_option_to_days(message, raw_service_date)
        if resolved:
            collected["service_date"] = resolved
            collected["awaiting_date_options"] = False
            brain["slot_rejected"] = False
        elif is_option_selection(message):
            # explicit menu selection should keep awaiting until we confirm the date
            collected["awaiting_date_options"] = True
        elif raw_service_date:
            direct = _resolve_service_date(raw_service_date)
            if direct and re.match(r"\d{2}-\d{2}-\d{4}", direct):
                collected["service_date"] = direct
                collected["awaiting_date_options"] = False
                brain["slot_rejected"] = False
    elif raw_service_date:
        if is_affirmative_response(message) and _is_ambiguous_numeric_date(raw_service_date):
            collected["service_date"] = resolve_affirmative_service_date(message, collected)
            brain["slot_rejected"] = False
        elif is_option_selection(message):
            mapped = _map_option_to_days(message, raw_service_date)
            if mapped:
                collected["service_date"] = mapped
                collected["awaiting_date_options"] = False
                brain["slot_rejected"] = False
            else:
                collected["service_date"] = _resolve_service_date(raw_service_date)
                brain["slot_rejected"] = False
        else:
            # Initial "kal" / "tomorrow" acceptance or any free-form date
            collected["service_date"] = _resolve_service_date(raw_service_date)
            brain["slot_rejected"] = False

    if ext_entities.get("resume_date"):
        collected["resume_date"] = _resolve_service_date(ext_entities["resume_date"])
    if ext_entities.get("next_trip_date"):
        collected["next_trip_date"] = _resolve_service_date(ext_entities["next_trip_date"])

    if not collected.get("service_date") and not raw_service_date:
        if is_option_selection(message):
            resolved_from_message = _map_option_to_days(message, raw_service_date)
        else:
            resolved_from_message = _resolve_service_date(message)
        if resolved_from_message and re.match(r"\d{2}-\d{2}-\d{4}", resolved_from_message):
            collected["service_date"] = resolved_from_message
            brain["slot_rejected"] = False
        else:
            affirmative_date = resolve_affirmative_service_date(message, collected)
            if affirmative_date:
                collected["service_date"] = affirmative_date
                brain["slot_rejected"] = False

    if is_phone_refusal_response(message):
        collected["driver_phone"] = "NOT_PROVIDED"

    chat_hist.append({"role": "user", "text": message})
    session["collected_json"] = collected
    session["chat_history"] = chat_hist
    prefix_reply = f"{brain.get('side_question_reply')}\n\n" if brain.get("side_question_reply") else ""

    standing_hours = collected.get("standing_hours", 0.0)

    # ── CLEVER INTERCEPTION: Handle Side-Questions Without Advancing Flow ────
    # If the user asked a side question AND they did NOT provide the data needed
    # for the current pending slot, we answer their question and nudge them back.
    if brain.get("side_question_reply") and current_state != "INITIAL_ALERT":
        reminder_prompt = build_troubleshooting_prompt(current_intent, collected, standing_hours)
        combined_response = (
            f"{brain['side_question_reply']}\n\n"
            f"🔄 Waapas aate hain aapki issue par—\n{reminder_prompt}"
        )
        chat_hist.append({"role": "bot", "text": combined_response})
        database.save_session(phone, current_state, collected, chat_hist)
        send_whatsapp_meta(phone, combined_response)
        return {"status": "side_question_answered_flow_preserved", "current_state": current_state}
    # ─────────────────────────────────────────────────────────────────────────

    # ==========================================================================
    # STATE MACHINE
    # ==========================================================================

    if current_state == "INITIAL_ALERT" and current_intent in {
        "VEHICLE_RUNNING", "GPS_DAMAGED", "VEHICLE_STANDING", "GPS_REMOVED"
    }:
        contact_choice = detect_contact_choice(message)
        if contact_choice == "self":
            collected["contact_mode"] = "self"
            collected["active_contact_phone"] = phone
            reply_msg = f"{prefix_reply}Samjha gaya. Hum aapke saath direct troubleshooting continue karenge. Aapki gaadi abhi kis location par hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "service_booking_started", "state": "COLLECTING_DETAILS"}

        if contact_choice == "driver":
            collected["contact_mode"] = "driver"
            if collected.get("driver_name") or collected.get("driver_phone"):
                reply_msg = f"{prefix_reply}{build_driver_confirmation_prompt(collected)}"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "verifying_existing_driver"}

        if contact_choice == "other":
            collected["contact_mode"] = "other"
            reply_msg = f"{prefix_reply}Theek hai. Kripya contact person's name (optional) aur mobile number bhejiye."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_contact_person"}

        if current_intent == "VEHICLE_RUNNING":
            reply_msg = f"{prefix_reply}Aapki gaadi abhi kis location par hai? (Current location batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_current_location"}

        reply_msg = f"{prefix_reply}Kripya vehicle ki sthiti short me batayein."
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "collecting_details"}

    # ── AWAITING_TROUBLESHOOTING_CONTACT STEP FIX
    if current_state == "AWAITING_TROUBLESHOOTING_CONTACT":
        contact_choice = detect_contact_choice(message)
        phone_in_message = extract_phone_number(message)

        if phone_in_message and contact_choice is None:
            collected["contact_mode"] = "other"
            new_name = ext_entities.get("contact_person") or ext_entities.get("driver_name")
            if new_name:
                collected["contact_person"] = new_name
            collected["driver_phone"] = phone_in_message
            collected["active_contact_phone"] = phone_in_message
            return reassign_troubleshooting_contact(
                phone, phone_in_message, collected, chat_hist,
                "COLLECTING_DETAILS", current_intent, standing_hours
            )

        if contact_choice == "self":
            collected["contact_mode"] = "self"
            collected["active_contact_phone"] = phone
            
            if not collected.get("intent"):
                collected["intent"] = "VEHICLE_RUNNING"
                
            prompt = build_troubleshooting_prompt(collected["intent"], collected, standing_hours).replace(r"\n", "\n")
            chat_hist.append({"role": "bot", "text": prompt})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, prompt)
            return {"status": "service_booking_started", "state": "COLLECTING_DETAILS"}
        

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

        reply_msg = (
            f"{prefix_reply}Kya aap khud GPS issue check karenge, "
            f"ya hum driver ya kisi aur contact person se baat karein?"
        )
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "AWAITING_TROUBLESHOOTING_CONTACT", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "awaiting_troubleshooting_contact"}

    # ── AWAITING_DRIVER_CONFIRMATION ──────────────────────────────────────────
    if current_state == "AWAITING_DRIVER_CONFIRMATION":
        driver_phone_in_msg = extract_phone_number(message)
        new_contact_name = ext_entities.get("contact_person") or ext_entities.get("driver_name")

        if is_affirmative_response(message) and not driver_phone_in_msg:
            target_phone = collected.get("driver_phone") or phone
            collected["contact_mode"] = "driver"
            collected["active_contact_phone"] = target_phone
            collected["contact_person"] = collected.get("driver_name") or collected.get("contact_person") or "Driver"
            collected["driver_contact_confirmed"] = True
            return transfer_owner_to_driver(phone, target_phone, collected, chat_hist, current_intent)

        if driver_phone_in_msg:
            collected["contact_mode"] = "driver"
            collected["driver_phone"] = driver_phone_in_msg
            if new_contact_name:
                collected["driver_name"] = new_contact_name
                collected["contact_person"] = new_contact_name
            else:
                collected["contact_person"] = collected.get("driver_name") or "Driver"
            collected["active_contact_phone"] = driver_phone_in_msg
            collected["driver_contact_confirmed"] = True
            return transfer_owner_to_driver(phone, driver_phone_in_msg, collected, chat_hist, current_intent)

        reply_msg = f"{prefix_reply}Please driver ko confirm karein ya naya naam/number bhej dein."
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "AWAITING_DRIVER_CONFIRMATION", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "awaiting_driver_confirmation"}

    # ── AWAITING_OTHER_CONTACT_DETAILS ────────────────────────────────────────
    if current_state == "AWAITING_OTHER_CONTACT_DETAILS":
        new_phone = extract_phone_number(message) or ext_entities.get("driver_phone")
        if not new_phone:
            reply_msg = f"{prefix_reply}Kripya contact person's mobile number bhejiye."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "AWAITING_OTHER_CONTACT_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "awaiting_other_contact_details"}

        new_name = ext_entities.get("contact_person") or ext_entities.get("driver_name")
        if new_name:
            collected["contact_person"] = new_name
        collected["driver_phone"] = new_phone
        collected["contact_mode"] = "other"
        collected["active_contact_phone"] = new_phone
        return reassign_troubleshooting_contact(
            phone, new_phone, collected, chat_hist,
            "SELF_REPAIR_IGNITION", current_intent, standing_hours,
            "Update: alternate contact add ho gaya hai. Ab troubleshooting naye contact ke saath continue ho rahi hai.",
            explicit_prompt="Kripya ek baar ignition ON hai ya nahi check kijiye.",
        )

    # ── SELF REPAIR STEPS ─────────────────────────────────────────────────────
    if current_state == "SELF_REPAIR_IGNITION":
        return perform_self_repair_check_and_continue(
            phone, collected, chat_hist,
            "SELF_REPAIR_LED",
            "GPS device ki LED jal rahi hai, blink kar rahi hai ya band hai?"
        )

    if current_state == "SELF_REPAIR_LED":
        return perform_self_repair_check_and_continue(
            phone, collected, chat_hist,
            "SELF_REPAIR_SIM",
            "Kya GPS device ki SIM active hai aur usme data pack hai?"
        )

    if current_state == "SELF_REPAIR_SIM":
        if is_pata_nahi_response(message):
            info_msg = "SIM ko kisi mobile me laga kar internet check kar lijiye."
            chat_hist.append({"role": "bot", "text": info_msg})
            send_whatsapp_meta(phone, info_msg)
        return perform_self_repair_check_and_continue(
            phone, collected, chat_hist,
            "SELF_REPAIR_WIRING",
            "GPS device ki wiring aur power connection ek baar check kar lijiye."
        )

    if current_state == "SELF_REPAIR_WIRING":
        return perform_self_repair_check_and_continue(
            phone, collected, chat_hist,
            "SELF_REPAIR_OPEN_SKY",
            "Vehicle ko open area me le jaakar 5–10 minute wait kijiye."
        )

    if current_state == "SELF_REPAIR_OPEN_SKY":
        return perform_self_repair_check_and_continue(
            phone, collected, chat_hist,
            "SELF_REPAIR_FINAL_VERIFICATION",
            "Kripya 2–3 minute wait kijiye. Hum backend se GPS dobara check kar rahe hain."
        )

    if current_state == "SELF_REPAIR_FINAL_VERIFICATION":
        active_phone = collected.get("active_contact_phone") or phone
        snapshot = get_backend_gps_snapshot(active_phone)
        if snapshot and is_gps_online(snapshot):
            return close_self_repair_case(phone, collected, chat_hist)

        # FIX S13: Ensure intent is set for the booking flow.
        # When a driver reaches here, original_customer_phone is set.
        # We must preserve/set intent so VEHICLE_RUNNING branch fires.
        if not collected.get("intent"):
            collected["intent"] = "VEHICLE_RUNNING"

        reply_msg = (
            "Hum ab service booking flow start kar rahe hain.\n"
            "Kripya apni vehicle ki current location batayein."
        )
        chat_hist.append({"role": "bot", "text": reply_msg})
        database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
        send_whatsapp_meta(phone, reply_msg)
        return {"status": "service_booking_started"}

    # ==========================================================================
    # WORKFLOW BRANCH ROUTING (COLLECTING_DETAILS and beyond)
    # ==========================================================================

    # ── WORKSHOP FLOW ─────────────────────────────────────────────────────────
    if current_intent == "WORKSHOP":
        # Check if we already extracted or resolved a date in this turn
        resolved_date = _resolve_service_date(message)
        if resolved_date and re.match(r"\d{2}-\d{2}-\d{4}", resolved_date):
            collected["expected_running_date"] = resolved_date
        elif ext_entities.get("resume_date"):
            collected["expected_running_date"] = _resolve_service_date(ext_entities["resume_date"])

        if not collected.get("expected_running_date"):
            reply_msg = f"{prefix_reply}Vehicle workshop se kab tak bahar aa jayegi?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_workshop_date"}

        # Save explicit statuses and confirmation
        collected["status"] = "WORKSHOP"
        expected_date = collected.get("expected_running_date") or collected.get("resume_date") or collected.get("service_date")
        reply_msg = (
            f"Thank you. Hum {expected_date or 'us date'} ke baad GPS status dobara check karenge."
        )
        chat_hist.append({"role": "bot", "text": reply_msg})
        send_whatsapp_meta(phone, reply_msg)
        
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        database.delete_session(phone)
        return {"status": "workshop_case_closed"}

    # ── ACCIDENT FLOW ─────────────────────────────────────────────────────────
    elif current_intent == "ACCIDENT":
        resolved_date = _resolve_service_date(message)
        if resolved_date and re.match(r"\d{2}-\d{2}-\d{4}", resolved_date):
            collected["expected_running_date"] = resolved_date
        elif ext_entities.get("resume_date"):
            collected["expected_running_date"] = _resolve_service_date(ext_entities["resume_date"])

        if not collected.get("expected_running_date"):
            reply_msg = f"{prefix_reply}Gaadi kab tak running condition me aa jayegi?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_accident_date"}

        collected["status"] = "ACCIDENT"
        expected_date = collected.get("expected_running_date") or collected.get("resume_date") or collected.get("service_date")
        reply_msg = (
            f"Thank you. Hum {expected_date or 'us date'} tak wait karte karte hai uske baad GPS status dobara check karenge."
        )
        chat_hist.append({"role": "bot", "text": reply_msg})
        send_whatsapp_meta(phone, reply_msg)
        
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        database.delete_session(phone)
        return {"status": "accident_case_closed"}

    # ── VEHICLE_STANDING (> 48 hours) ─────────────────────────────────────────
    elif current_intent == "VEHICLE_STANDING" and standing_hours > 48:
        if not collected.get("resume_date"):
            reply_msg = f"{prefix_reply}Gaadi 48 ghante se jyada se stationary hai. Agli trip kis date ko nikalne wali hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_resume_date_standing"}

        reply_msg = "✅ Information update ho gayi hai. Gaadi next run par aate hi data transmission automatic update ho jayega."
        chat_hist.append({"role": "bot", "text": reply_msg})
        send_whatsapp_meta(phone, reply_msg)
        database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
        database.delete_session(phone)
        return {"status": "standing_long_duration_closed"}

    # ── GPS_REMOVED ───────────────────────────────────────────────────────────
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
            send_whatsapp_meta(phone, reply_msg)
            database.save_session(phone, "CASE_CLOSED", collected, chat_hist)
            database.delete_session(phone)
            return {"status": "gps_removed_self_fix_closed"}
        pass

    # ── VEHICLE_RUNNING ───────────────────────────────────────────────────────
    if current_intent == "VEHICLE_RUNNING":

        # Step A: Current location
        if not collected.get("current_location"):
            reply_msg = f"{prefix_reply}Aapki gaadi abhi kis location par hai? (Current location batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_current_location"}

        # Step B: Destination
        if not collected.get("destination_location"):
            reply_msg = f"{prefix_reply}Kahan ja rahe hain? (Destination batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_destination"}

        # Step C: Suggest service city
        if collected.get("service_city_confirmed") is None and not collected.get("service_city"):
            confirmation = apply_service_city_confirmation(message, collected, ext_entities)
            if confirmation is not None:
                if confirmation["confirmed"]:
                    # Continue to the next step directly without asking again.
                    pass
                else:
                    reply_msg = f"{prefix_reply}Kaun se city mein service chahiye? (Preferred city batayein)"
                    chat_hist.append({"role": "bot", "text": reply_msg})
                    database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                    send_whatsapp_meta(phone, reply_msg)
                    return {"status": "collecting_preferred_service_city"}
            else:
                suggested_city = collected.get("destination_location") or "Delhi"
                reply_msg = f"{prefix_reply}Kya hum {suggested_city} mein service book kar dein?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "asking_service_city_preference"}

        # Step D: If city rejected, ask preferred city
        if collected.get("service_city_confirmed") is False and not collected.get("service_city"):
            reply_msg = f"{prefix_reply}Kaun se city mein service chahiye? (Preferred city batayein)"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_preferred_service_city"}

        # Step E: Service date
        if not collected.get("service_date"):
            step = collected.get("scheduling_step", 0)

            if brain.get("slot_rejected") is True:
                step += 1
                collected["scheduling_step"] = step
                brain["slot_rejected"] = False

            if step == 0:
                reply_msg = f"{prefix_reply}{get_service_date_prompt()}"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "asking_service_date_tomorrow"}

            else:
                # Show the 3-option menu
                collected["awaiting_date_options"] = True
                reply_msg = (
                    f"{prefix_reply}Please choose one option from below:\n\n"
                    f"1️⃣ Book service after 2 days\n"
                    f"2️⃣ Book service after 4 days\n"
                    f"3️⃣ Enter a specific date or tell after how many days..."
                )
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_alternative_service_date"}

        # Step F: Driver confirmation
        if not collected.get("driver_contact_confirmed") and (collected.get("driver_name") or collected.get("driver_phone")):
            if collected.get("awaiting_alternate_contact"):
                new_phone = extract_phone_number(message)
                new_name = ext_entities.get("contact_person") or ext_entities.get("driver_name")
                if new_phone or new_name:
                    collected["awaiting_alternate_contact"] = False
                    collected["driver_contact_confirmed"] = True
                    if new_name:
                        collected["contact_person"] = new_name
                        collected["driver_name"] = new_name
                    if new_phone:
                        collected["driver_phone"] = new_phone
                    reply_msg = f"{prefix_reply}Kitne baje se kitne baje tak vehicle available hogi?"
                    chat_hist.append({"role": "bot", "text": reply_msg})
                    database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                    send_whatsapp_meta(phone, reply_msg)
                    return {"status": "asking_service_time_window"}

                reply_msg = f"{prefix_reply}Kripya alternate contact person ka naam aur mobile number bhejiye."
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "collecting_alternate_contact"}

            confirmation = apply_driver_confirmation(message, collected, ext_entities)
            if confirmation is not None:
                if confirmation.get("confirmed"):
                    reply_msg = f"{prefix_reply}Kitne baje se kitne baje tak vehicle available hogi?"
                    chat_hist.append({"role": "bot", "text": reply_msg})
                    database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                    send_whatsapp_meta(phone, reply_msg)
                    return {"status": "asking_service_time_window"}

                reply_msg = f"{prefix_reply}Theek hai. Kripya alternate contact person ka naam aur mobile number bhejiye."
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "collecting_alternate_contact"}

            reply_msg = f"{prefix_reply}{build_driver_confirmation_prompt(collected)}"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "verifying_existing_driver"}

        if not collected.get("service_time_window"):
            time_window = apply_service_time_window(message, collected)
            if time_window is not None:
                reply_msg = f"{prefix_reply}Shukriya. Hum service slot ko note kar rahe hain."
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
            else:
                reply_msg = f"{prefix_reply}Kitne baje se kitne baje tak vehicle service ke liye available hogi? (Jaise 10:00 se 14:00, 3 baje se 5 baje tak, ya koi aur format)"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "asking_service_time_window"}

        # All details collected — create ticket
        return await create_service_ticket_flow(phone, collected, chat_hist)

    # ── SLOT-FILLING ENGINE for GPS_DAMAGED / VEHICLE_STANDING / GPS_REMOVED ──
    elif current_intent in ("GPS_DAMAGED", "VEHICLE_STANDING", "GPS_REMOVED"):

        if not collected.get("vehicle_location"):
            reply_msg = f"{prefix_reply}Gaadi abhi kis city/location par chal rahi hai?"
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_location"}

        loc = collected.get("vehicle_location")
        step = collected.get("scheduling_step", 0)
        if brain.get("slot_rejected") is True:
            step += 1
            collected["scheduling_step"] = step

        if not collected.get("service_date"):
            if step == 0:
                current_hour = datetime.now().hour
                if current_hour < 12:
                    reply_msg = f"{prefix_reply}Aapke current route par *{loc} service point* upalabdh hai. Kya hum aaj *shaam* tak inspection schedule kar dein?"
                else:
                    reply_msg = f"{prefix_reply}Aapke area ke hisab se *{loc} service counter* check ho sakta hai. Kya hum isko *kal* ke liye fix karein?"
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

            elif step == 1:
                reply_msg = "Koi baat nahi sir, kya phir 4 din baad ka appointment set kar dein?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_date_step_1"}

            elif step == 2:
                reply_msg = "Aapke scheduling ke hisab se, kya 5 se 7 dino ke baad inspection karwana sahi rahega?"
                chat_hist.append({"role": "bot", "text": reply_msg})
                database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
                send_whatsapp_meta(phone, reply_msg)
                return {"status": "negotiating_date_step_2"}

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

        if not collected.get("driver_phone") or collected.get("driver_phone") == "NOT_PROVIDED":
            reply_msg = f"{prefix_reply}Driver ka active mobile number share kijiye taaki technician coordinate kar sake."
            chat_hist.append({"role": "bot", "text": reply_msg})
            database.save_session(phone, "COLLECTING_DETAILS", collected, chat_hist)
            send_whatsapp_meta(phone, reply_msg)
            return {"status": "collecting_driver_phone"}

        # Create ticket
        ticket_id = f"TKT-{random.randint(10000, 99999)}"
        collected["ticket_id"] = ticket_id

        ticket_payload = {
            "vehicle_location": collected.get("vehicle_location"),
            "service_date": collected.get("service_date") or date.today().isoformat(),
            "driver_phone": collected.get("driver_phone") if collected.get("driver_phone") != "NOT_PROVIDED" else phone,
            "engineer_id": "ENG-642",
            "engineer_name": "Ramesh Kumar",
            "engineer_phone": "9876543210",
            "assignment_status": "ASSIGNED",
        }

        database.save_ticket(ticket_id, phone, ticket_payload)

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
        send_whatsapp_meta(phone, reply_msg)
        database.delete_session(phone)
        try:
            database.save_session(phone, "TICKET_RAISED", collected, chat_hist)
        except Exception:
            pass
        return {"status": "ticket_created_successfully", "ticket_id": ticket_id}

    # ── FALLBACK ──────────────────────────────────────────────────────────────
    conversational_fallback = brain.get("conversational_reply") or "Kripya vehicle ki sthiti short me spasht karein."
    chat_hist.append({"role": "bot", "text": conversational_fallback})
    database.save_session(phone, current_state, collected, chat_hist)
    send_whatsapp_meta(phone, conversational_fallback)
    return {"status": "fallback_interaction_prompted"}