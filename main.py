import os
import json
import requests
from typing import Optional, Literal
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import Response, PlainTextResponse
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

import database

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
# 2. GROQ SYSTEM DEFINITION
# ==========================================

# Updated System Instructions to force structural compliance manually via JSON_OBJECT mode
SYSTEM_INSTRUCTION = """
You are an automated AI-powered GPS Troubleshooting Bot. Your job is to process incoming customer messages regarding vehicle GPS downtime.

You MUST respond strictly with a valid JSON object matching this exact structural schema:
{
    "extracted_data": {
        "vehicle_location": string or null,
        "service_date": string or null,
        "driver_phone": string or null
    },
    "next_state": "INITIAL_ALERT" | "CASE_CLOSED" | "COLLECTING_DETAILS" | "TICKET_RAISED",
    "reply_to_user": string
}

FLOW RULES:
1. State: INITIAL_ALERT
   - If user selects options 1, 2, 3, or 4 (or indicates tracking was stopped intentionally), update next_state to 'CASE_CLOSED' and say thank you.
   - If user selects 5, 6, 7, 8 OR sends general text indicating a valid fault, transition to 'COLLECTING_DETAILS'.

2. State: COLLECTING_DETAILS
   - Extract 'vehicle_location', 'service_date', and 'driver_phone' into 'extracted_data' without overwriting existing data.
   - If ALL 3 fields are full, set next_state to 'TICKET_RAISED' and confirm a technician is assigned.
   - If fields are missing, stay in 'COLLECTING_DETAILS' and ask for EXACTLY ONE missing item in polite, brief Hinglish.

GUARDRAILS:
- Talk in casual, respectful Hinglish. Keep text output short (1 line max).
- If off-topic, give a fast 1-line reply and immediately ask for your missing item.
- Return raw JSON only. Do not wrap in Markdown blocks.
"""

# ==========================================
# 3. META WHATSAPP OUTBOUND UTILITY
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
            print(f"Meta API Error response: {response.text}")
    except Exception as e:
        print(f"Exception trying to push message to Meta Graph API: {e}")

# ==========================================
# 4. SYSTEM WEBHOOKS & API CHANNELS
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
        user_input = message_data["text"]["body"].strip()
    except (KeyError, IndexError):
        return {"status": "malformed_meta_payload"}

    session = database.get_session(clean_sender)
    session["chat_history"].append({"role": "user", "text": user_input})
    
    execution_context = (
        f"Current Bot State: {session['current_state']}\n"
        f"Currently Extracted Data JSON: {json.dumps(session['collected_json'])}\n\n"
        f"Recent Chat History:\n"
        + "\n".join([f"{m['role'].upper()}: {m['text']}" for m in session['chat_history']])
    )
    
    # ✅ FIX: Changed response_format to "json_object" which is natively supported by Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": execution_context}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    parsed_output = json.loads(response.choices[0].message.content)
    
    session["chat_history"].append({"role": "bot", "text": parsed_output["reply_to_user"]})
    database.save_session(
        phone_number=clean_sender,
        state=parsed_output["next_state"],
        collected_json=parsed_output["extracted_data"],
        chat_history=session["chat_history"]
    )
    
    send_whatsapp_meta(clean_sender, parsed_output["reply_to_user"])
    
    if parsed_output["next_state"] == "TICKET_RAISED":
        print(f"\n[🎟️ CRM UPDATE] Raised Ticket for Account {clean_sender} with Data: {parsed_output['extracted_data']}\n")
        
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
        f"Reply with the option number."
    )
    
    database.save_session(
        phone_number=payload.phone_number,
        state="INITIAL_ALERT",
        collected_json={"vehicle_location": None, "service_date": None, "driver_phone": None},
        chat_history=[{"role": "bot", "text": initial_alert_msg}]
    )
    
    send_whatsapp_meta(payload.phone_number, initial_alert_msg)
    return {"status": "alert_fired"}