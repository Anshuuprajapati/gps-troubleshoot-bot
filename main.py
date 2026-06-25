import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from battery_issue_flow import start_battery_flow, handle_whatsapp_replies as battery_webhook, WhatsAppWebhookMessage, init_db as battery_init_db
from main_power_cut_flow import start_main_power_flow, handle_whatsapp_replies as main_power_webhook
from other_issue_flow import start_other_issue_flow
import database

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("CentralRoutingHub")

app = FastAPI(title="GPS Outage Central Routing Hub & Pre-Analysis Engine")

@app.on_event("startup")
async def startup_event():
    database.init_db()
    battery_init_db()
    logger.info("Databases initialized on startup.")

# ==============================================================================
# 2. PYDANTIC SCHEMAS FOR DATA VALIDATION
# ==============================================================================

class GpsData(BaseModel):
    gpstime: Optional[str] = None
    main_powervoltage: Optional[float] = None
    ismainpoerconnected: Optional[str] = None  # "1" = connected, "0" = disconnected
    gpsStatus: Optional[int] = None            # 0 = no fix, 1 = fix
    driver_name: Optional[str] = None
    driver_phone: Optional[str] = None
    current_location: Optional[str] = None
    vehicle_state: Optional[str] = None


class OutageRequest(BaseModel):
    phone_number: str = Field(..., description="Customer/Manager phone number")
    vehicle_no: str
    last_location: str
    timestamp: str
    gps_data: Optional[GpsData] = None


# ==============================================================================
# 3. PRE-ANALYSIS ROUTING LOGIC
# ==============================================================================

def analyze_and_route_outage(payload: OutageRequest) -> str:
    """
    Evaluates telemetry data fields:
      - Main Power Connection
      - Battery Voltage
      - GPS Status
      - Vehicle Status
    Returns root_cause_string.
    """
    gps_data = payload.gps_data.model_dump() if payload.gps_data else {}
    
    # Extract telemetry metrics
    is_main_connected = str(gps_data.get("ismainpoerconnected", "1")).strip()
    voltage_raw = gps_data.get("main_powervoltage")
    gps_status = gps_data.get("gpsStatus")
    vehicle_state = gps_data.get("vehicle_state")

    logger.info(
        f"Analyzing Vehicle: {payload.vehicle_no} | Main Power Connected: {is_main_connected} | "
        f"Voltage: {voltage_raw}V | GPS Fix Status: {gps_status} | State: {vehicle_state}"
    )

    # ── CASE 2: MAIN POWER LINE CUT ──────────────────────────────────────────
    # Priority 1: Hardware flag explicitly reports loss of primary external power
    if is_main_connected == "0":
        logger.info(f"Root Cause Identified -> MAIN_POWER_CUT for Vehicle {payload.vehicle_no}")
        return "MAIN_POWER_CUT"

    # ── CASE 1: BATTERY ISSUE ────────────────────────────────────────────────
    # Priority 2: Main power lines are physically connected, but operational voltage drops below normal thresholds
    if voltage_raw is not None and float(voltage_raw) < 11.0:
        logger.info(f"Root Cause Identified -> BATTERY_ISSUE for Vehicle {payload.vehicle_no}")
        return "BATTERY_ISSUE"

    # ── CASE 3: OTHER ISSUES ─────────────────────────────────────────────────
    # Priority 3: Fallback state for errors such as firmware lockup, detached antenna, or missing updates
    logger.info(f"Root Cause Identified -> OTHER_ISSUE for Vehicle {payload.vehicle_no}")
    return "OTHER_ISSUE"


# ==============================================================================
# 4. ROUTER API ENDPOINT
# ==============================================================================

@app.post("/api/trigger-outage")
async def trigger_outage(payload: OutageRequest):
    """
    Central Entrypoint endpoint for tracking downtime issues (24+ hours).
    Ingests payload, completes core telemetry pre-analysis, and hands off complete
    payload context to the respective dedicated flow handler.
    """
    logger.info(f"Inbound 24+ Hour Outage Alert triggered for Vehicle: {payload.vehicle_no}")

    # 1. Execute deterministic hardware telemetry checks
    root_cause = analyze_and_route_outage(payload)

    # 2. Wrap payload with computed pre-analysis metadata for downline processing
    forward_payload = {
        "root_cause": root_cause,
        "phone_number": payload.phone_number,
        "vehicle_no": payload.vehicle_no,
        "last_location": payload.last_location,
        "timestamp": payload.timestamp,
        "gps_data": payload.gps_data.model_dump() if payload.gps_data else None
    }

    # 3. Call appropriate flow handler directly
    try:
        logger.info(f"Routing to {root_cause} flow handler")
        
        if root_cause == "BATTERY_ISSUE":
            result = await start_battery_flow(forward_payload)
        elif root_cause == "MAIN_POWER_CUT":
            result = await start_main_power_flow(forward_payload)
        else:  # OTHER_ISSUE
            result = await start_other_issue_flow(forward_payload)
            
        return {
            "status": "routed_successfully",
            "root_cause": root_cause,
            "flow_result": result
        }

    except Exception as e:
        logger.critical(f"Error in flow handler: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to process request: {str(e)}"
        )


# ==============================================================================
# 5. WHATSAPP WEBHOOK ENDPOINT
# ==============================================================================

@app.post("/webhook/")
async def whatsapp_webhook(request: Request):
    """
    WhatsApp webhook endpoint that receives incoming messages.
    Routes messages to appropriate flow based on session state.
    """
    try:
        body = await request.json()
        logger.info(f"Received webhook: {body}")
        
        # Parse WhatsApp webhook format (Meta/360dialog format)
        if "entry" in body:
            try:
                entry = body["entry"][0]
                changes = entry["changes"][0]
                value = changes["value"]
                
                # Check if there are messages
                if "messages" not in value:
                    logger.info("No messages in webhook payload")
                    return {"status": "ok"}
                
                message = value["messages"][0]
                phone_number = message["from"]
                message_text = message.get("text", {}).get("body", "")
                
                if not message_text:
                    logger.info("Empty message text")
                    return {"status": "ok"}
                
                logger.info(f"Message from {phone_number}: {message_text}")
                
                # Get session to determine which flow to route to
                session = database.get_session(phone_number)
                
                if not session:
                    logger.info(f"No active session found for {phone_number}")
                    return {"status": "no_session", "message": "No active conversation found"}
                
                root_cause = session.get("collected_json", {}).get("root_cause")
                logger.info(f"Routing to flow: {root_cause}")
                
                # Route to appropriate flow webhook handler
                if root_cause == "BATTERY_ISSUE":
                    webhook_msg = WhatsAppWebhookMessage(
                        phone_number=phone_number,
                        message_text=message_text
                    )
                    result = await battery_webhook(webhook_msg)
                    return result
                    
                elif root_cause == "MAIN_POWER_CUT" or root_cause == "MAIN_POWER":
                    webhook_msg = WhatsAppWebhookMessage(
                        phone_number=phone_number,
                        message_text=message_text
                    )
                    result = await main_power_webhook(webhook_msg)
                    return result
                    
                elif root_cause == "OTHER_ISSUE":
                    # TODO: Implement other issue webhook handler
                    logger.info("Other issue flow webhook - not yet implemented")
                    return {"status": "ok", "message": "Other issue flow"}
                    
                else:
                    logger.warning(f"Unknown root cause: {root_cause}")
                    return {"status": "ok", "message": "Unknown flow"}
                
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing webhook payload: {e}")
                return {"status": "error", "message": "Invalid webhook format"}
        
        else:
            logger.warning("Unknown webhook format")
            return {"status": "ok"}
            
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return {"status": "error", "message": str(e)}


# ==============================================================================
# 6. TEMPORARY TESTING API (FOR DEVELOPMENT ONLY)
# ==============================================================================

class SessionUpdateRequest(BaseModel):
    phone_number: str
    updates: dict = Field(..., description="Fields to update in session")

@app.put("/api/test/update-session")
async def update_session_for_testing(payload: SessionUpdateRequest):
    """
    Temporary API for testing - allows updating session data during conversation.
    
    Use this to simulate state changes, GPS data updates, etc. without real actions.
    """
    try:
        phone = payload.phone_number
        updates = payload.updates
        
        # Get current session
        session = database.get_session(phone)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active session found for {phone}"
            )
        
        old_state = session.get("current_state")
        old_data = session.get("collected_json", {}).copy()
        
        # Apply updates
        new_state = updates.get("current_state", old_state)
        collected_json = session.get("collected_json", {})
        history = session.get("chat_history", [])
        
        # Update collected_json fields
        if "collected_json" in updates:
            for key, value in updates["collected_json"].items():
                collected_json[key] = value
        
        # Update GPS data if provided
        if "gps_data" in updates:
            for key, value in updates["gps_data"].items():
                collected_json[key] = value
        
        # Save updated session
        database.save_session(phone, new_state, collected_json, history)
        
        logger.info(f"[TEST API] Updated session for {phone}: State {old_state} -> {new_state}")
        
        return {
            "status": "updated",
            "phone_number": phone,
            "old_state": old_state,
            "new_state": new_state,
            "old_data": old_data,
            "new_data": collected_json,
            "message": "Session updated successfully for testing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update session: {str(e)}"
        )


@app.get("/api/test/get-session/{phone_number}")
async def get_session_for_testing(phone_number: str):
    """
    Get current session data for a phone number (for testing/debugging).
    """
    try:
        session = database.get_session(phone_number)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No session found for {phone_number}"
            )
        
        return {
            "status": "found",
            "phone_number": phone_number,
            "session": session
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )