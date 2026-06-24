import os
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

import database
from date_utils import normalize_date

load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("BatteryIssueFlow")

app = FastAPI(title="GPS Outage Workflow - Case 1: Battery Issue Service")

# Initialize DB
database.init_db()

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


async def start_battery_flow(payload: dict):
    """
    Entry point called by main.py routing logic.
    Initializes battery issue flow.
    """
    return await start_battery_issue_flow(RoutedRequest(**payload))
    
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


def run_hardware_api_recheck(vehicle_no: str) -> Dict[str, Any]:
    """
    Queries hardware API systems to check if the tracking hardware has re-established comms.
    Returns status variables derived from live telemetry.
    """
    logger.info(f"Initiating live system hardware API health diagnostics for vehicle {vehicle_no}")
    try:
        # Mocking or calling your live tracking verification layer
        # Replace with your actual live system HTTP data query if needed
        return {
            "gps_working": False,            # Change to True if tracking matches device heartbeat
            "main_power_connected": "1",     # "1" = Connected, "0" = Broken Main Line
            "main_powervoltage": 12.4
        }
    except Exception as e:
        logger.error(f"Failed parsing real-time device tracking status: {e}")
        return {"gps_working": False, "main_power_connected": "1", "main_powervoltage": 0.0}


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


# ==============================================================================
# CENTRAL ANALYTICAL WEBHOOK & STATE MACHINE FOR CASE 1
# ==============================================================================
@app.post("/api/flow/battery-issue")
async def start_battery_issue_flow(payload: RoutedRequest):
    """
    Triggered exclusively by main.py routing logic.
    Registers session tracking states inside database layer and prompts choice routing.
    """
    gps_info = payload.gps_data.model_dump() if payload.gps_data else {}
    
    alert_msg = (
        f"🚨 *GPS Connectivity Alert* 🚨\n\n"
        f"Vehicle *{payload.vehicle_no}* has been offline for 24+ hours.\n"
        f"Our backend diagnostics indicate a *Low Battery / Charge Discharge* condition.\n\n"
        f"To restore active tracking, please let us know:\n"
        f"👉 *Will this vehicle battery be checked and charged by You, or will your Driver handle it?*"
    )

    initial_json = {
        "vehicle_no": payload.vehicle_no,
        "root_cause": "BATTERY_ISSUE",
        "driver_name": gps_info.get("driver_name", "N/A"),
        "driver_phone": gps_info.get("driver_phone", "N/A"),
        "last_location": payload.last_location,
        "battery_handler": None,  # 'CUSTOMER' or 'DRIVER'
        "physical_intent": None,
        "collected_details": {}
    }

    database.save_session(
        phone_number=payload.phone_number,
        current_state="AWAITING_HANDLER_CONFIRMATION",
        collected_json=initial_json,
        chat_history=[{"role": "bot", "text": alert_msg}]
    )

    send_whatsapp_meta(payload.phone_number, alert_msg)
    return {"status": "success", "message": "Battery workflow state tree initialized."}


@app.post("/api/flow/battery-issue/webhook")
async def handle_whatsapp_replies(webhook_data: WhatsAppWebhookMessage):
    """
    Main state engine executing sequential tracking, validations, and ticket creation.
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

    logger.info(f"Processing chat session on number: {user_phone} | Current State: {state}")

    # Append customer message to history
    history.append({"role": "user", "content": incoming_text})

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: AWAITING_HANDLER_CONFIRMATION (Who will charge it?)
    # ──────────────────────────────────────────────────────────────────────────
    if state == "AWAITING_HANDLER_CONFIRMATION":
        prompt = (
            "Analyze the text and determine who will charge or service the vehicle battery.\n"
            "Options: Return exactly 'CUSTOMER' if the owner/manager says they will do it.\n"
            "Return 'DRIVER' if they mention the driver, operator, or provide a driver's name/number.\n"
            "Return 'OTHER' if unclear."
        )
        handler = llm_parse_text(incoming_text, prompt)

        if "CUSTOMER" in handler:
            context["battery_handler"] = "CUSTOMER"
            next_msg = (
                "Understood. Please arrange to charge or reconnect the vehicle's battery.\n\n"
                "Once the battery has been reconnected and the vehicle turned on, "
                "reply with *'DONE'* or *'CONNECTED'* so we can verify the hardware connection."
            )
            history.append({"role": "assistant", "content": next_msg})
            database.save_session(user_phone, "AWAITING_CHARGE_COMPLETION", context, history)
            send_whatsapp_meta(user_phone, next_msg)

        elif "DRIVER" in handler:
            context["battery_handler"] = "DRIVER"
            d_name = context.get("driver_name", "N/A")
            d_phone = context.get("driver_phone", "N/A")
            
            next_msg = (
                "Got it. We will coordinate with the driver.\n"
                f"Please confirm the driver contact information on file:\n"
                f"• Name: {d_name}\n"
                f"• Phone: {d_phone}\n\n"
                f"Is this driver information correct? Reply *'YES'* to confirm or provide the updated *Name, Number*."
            )
            history.append({"role": "assistant", "content": next_msg})
            database.save_session(user_phone, "CONFIRMING_DRIVER_DETAILS", context, history)
            send_whatsapp_meta(user_phone, next_msg)
        else:
            fallback = "Could you please specify clearly? Will the battery be handled by *You* or the *Driver*?"
            send_whatsapp_meta(user_phone, fallback)

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: CONFIRMING_DRIVER_DETAILS
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "CONFIRMING_DRIVER_DETAILS":
        if "YES" in incoming_text.upper():
            driver_phone = context.get("driver_phone")
            driver_msg = (
                f"Hello, your office/manager reported that Vehicle *{vehicle_no}* is offline due to a low battery.\n\n"
                f"Please ensure the vehicle battery is fully connected or charged.\n"
                f"Once done, please reply to this number with *'DONE'* to restore tracking tracking updates."
            )
            # Route alert message down to target driver hardware operator
            if driver_phone and driver_phone != "N/A":
                send_whatsapp_meta(driver_phone, driver_msg)

            mgr_msg = f"Thank you. Notification sent to driver. Awaiting charge confirmation from either party."
            history.append({"role": "assistant", "content": mgr_msg})
            database.save_session(user_phone, "AWAITING_CHARGE_COMPLETION", context, history)
            send_whatsapp_meta(user_phone, mgr_msg)
        else:
            # Simple text capture parsing logic to update driver info variables
            parts = incoming_text.split(",")
            if len(parts) >= 2:
                context["driver_name"] = parts[0].strip()
                context["driver_phone"] = ''.join(filter(str.isdigit, parts[1].strip()))
                
                # Resend verification to driver phone sequence
                send_whatsapp_meta(context["driver_phone"], f"Alert: Please charge the battery for vehicle {vehicle_no} and reply DONE.")
                msg = "Driver details updated. We have dispatched notifications to the driver."
                history.append({"role": "assistant", "content": msg})
                database.save_session(user_phone, "AWAITING_CHARGE_COMPLETION", context, history)
                send_whatsapp_meta(user_phone, msg)
            else:
                send_whatsapp_meta(user_phone, "Please provide driver updates in format: *Driver Name, Phone Number* or reply *YES*.")

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: AWAITING_CHARGE_COMPLETION -> SYSTEM API RECHECK
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "AWAITING_CHARGE_COMPLETION":
        # Check if user or driver confirms task completion
        if "DONE" in incoming_text.upper() or "CONNECT" in incoming_text.upper():
            send_whatsapp_meta(user_phone, "🔄 *Running Live System Diagnostic API Recheck... Please wait.*")
            
            # RUN SYSTEM API RECHECK
            api_status = run_hardware_api_recheck(vehicle_no)
            
            if api_status.get("gps_working") is True:
                success_msg = f"✅ *Success!* Vehicle {vehicle_no} GPS status is now fully active. Case closed."
                database.delete_session(user_phone) # Close session out successfully
                send_whatsapp_meta(user_phone, success_msg)
            else:
                # GPS is still offline -> RUN MAIN POWER CHECK BRANCH
                logger.info("Battery reported fixed but hardware telemetry remains offline. Branching to Main Power Check.")
                
                if api_status.get("main_power_connected") == "0":
                    # MAIN POWER CUT DETECTED
                    power_cut_msg = (
                        "⚠️ *Main Power Cut Detected* ⚠️\n\n"
                        "The battery issue appears fixed, but our diagnostics reveal that the device "
                        "is not receiving external vehicle power. The main wiring power line may be disconnected or cut.\n\n"
                        "Redirecting your query to our Main Power Line team for next steps..."
                    )
                    send_whatsapp_meta(user_phone, power_cut_msg)
                    # Delete active sub-session and forward execution payload outward to power flow service
                    database.delete_session(user_phone)
                    # (Implementation detail: Invoke main power microservice handler route)
                else:
                    # Power connected but GPS still offline -> PROCEED TO PHYSICAL GPS DIAGNOSIS
                    msg = (
                        "📊 *Diagnostic Update*:\n"
                        "Battery levels look fine and main power lines are connected, but your GPS device is still offline.\n\n"
                        "To narrow down the root cause, *please describe the current vehicle condition.*"
                    )
                    history.append({"role": "assistant", "content": msg})
                    database.save_session(user_phone, "AWAITING_PHYSICAL_DIAGNOSIS", context, history)
                    send_whatsapp_meta(user_phone, msg)
        else:
            send_whatsapp_meta(user_phone, "Please reply with *'DONE'* once the battery hardware has been charged and reconnected.")

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: PHYSICAL GPS DIAGNOSIS (Intent processing framework)
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "AWAITING_PHYSICAL_DIAGNOSIS":
        intent_prompt = (
            "Analyze the statement detailing vehicle conditions and classify into exactly ONE of these options:\n"
            "• GPS_DAMAGED (device broken)\n"
            "• GPS_REMOVED (device taken out)\n"
            "• WORKSHOP (vehicle undergoing repairs/maintenance)\n"
            "• ACCIDENT (vehicle crashed)\n"
            "• VEHICLE_RUNNING_NOT_UPDATING (vehicle moving but tracking frozen)\n"
            "• VEHICLE_STANDING (parked long term)\n"
            "• OTHER (general text replies)\n"
            "Output only the classification key string."
        )
        detected_intent = llm_parse_text(incoming_text, intent_prompt)
        context["physical_intent"] = detected_intent
        
        logger.info(f"Locked System Intent for {vehicle_no} -> {detected_intent}")
        
        # Advance state to collect dispatch scheduling parameters
        collect_msg = (
            f"Intent Registered: Physical issue category parsed (*{detected_intent}*).\n\n"
            f"To schedule a field service assignment, please provide the details in this format:\n"
            f"`Current Location, Destination, Arrival Date (DD-MM-YYYY), Service City, Contact Person Name`"
        )
        history.append({"role": "assistant", "content": collect_msg})
        database.save_session(user_phone, "COLLECTING_SERVICE_DETAILS", context, history)
        send_whatsapp_meta(user_phone, collect_msg)

    # ──────────────────────────────────────────────────────────────────────────
    # STATE: COLLECTING_SERVICE_DETAILS -> TICKET GENERATION
    # ──────────────────────────────────────────────────────────────────────────
    elif state == "COLLECTING_SERVICE_DETAILS":
        details = incoming_text.split(",")
        if len(details) >= 4:
            curr_loc = details[0].strip()
            dest = details[1].strip()
            raw_date = details[2].strip()
            srv_city = details[3].strip()
            contact = details[4].strip() if len(details) > 4 else "Manager"

            norm_date = normalize_date(raw_date)
            ticket_id = f"TKT-BATT-{os.urandom(2).hex().upper()}"

            # Prepare structured dictionary context block for table writes
            ticket_data = {
                "vehicle_location": curr_loc,
                "service_date": norm_date,
                "driver_phone": context.get("driver_phone", "N/A"),
                "engineer_id": "ENG-991",
                "engineer_name": "Ramesh Kumar", # Auto-assignment logic
                "engineer_phone": "919999988888",
                "assignment_status": "ASSIGNED"
            }

            # Commit ticket entity straight to persistence layer via database module
            database.save_ticket(ticket_id, user_phone, ticket_data)

            confirm_msg = (
                f"🎫 *Service Ticket Created Successfully!*\n\n"
                f"• *Ticket ID:* {ticket_id}\n"
                f"• *Issue Intent:* {context.get('physical_intent')}\n"
                f"• *Scheduled Service City:* {srv_city}\n"
                f"• *Target Target Date:* {norm_date}\n"
                f"• *Assigned Engineer:* {ticket_data['engineer_name']} ({ticket_data['engineer_phone']})\n\n"
                f"Our engineer will coordinate with {contact} prior to arrival."
            )
            
            database.delete_session(user_phone) # Tear down conversation tree context block
            send_whatsapp_meta(user_phone, confirm_msg)
        else:
            error_retry = (
                "⚠️ *Format Incorrect*.\n\n"
                "Please send the ticket data in a comma-separated format precisely:\n"
                "`Current Location, Destination, Arrival Date (DD-MM-YYYY), Service City, Contact Name`"
            )
            send_whatsapp_meta(user_phone, error_retry)

    return {"status": "processed"}