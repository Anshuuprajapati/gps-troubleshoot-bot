import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from battery_issue_flow import start_battery_flow, handle_whatsapp_replies as battery_webhook, WhatsAppWebhookMessage, init_db as battery_init_db
from main_power_cut_flow import start_main_power_flow, handle_whatsapp_replies as main_power_webhook
from other_issue_flow import start_other_issue_flow, handle_whatsapp_replies as other_issue_webhook
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
    gps_payload = payload.gps_data.model_dump() if payload.gps_data else {}
    forward_payload = {
        "root_cause": root_cause,
        "phone_number": payload.phone_number,
        "vehicle_no": payload.vehicle_no,
        "last_location": payload.last_location,
        "timestamp": payload.timestamp,
        "gps_data": gps_payload,
        "gpstime": gps_payload.get("gpstime"),
        "main_powervoltage": gps_payload.get("main_powervoltage"),
        "ismainpoerconnected": gps_payload.get("ismainpoerconnected"),
        "gpsStatus": gps_payload.get("gpsStatus"),
        "vehicle_state": gps_payload.get("vehicle_state"),
        "driver_name": gps_payload.get("driver_name"),
        "driver_phone": gps_payload.get("driver_phone")
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
                normalized_root_cause = str(root_cause or "").upper()
                logger.info(f"Routing to flow: {root_cause}")
                
                # Route to appropriate flow webhook handler
                if normalized_root_cause in {"BATTERY_ISSUE", "BATTERY_LOW"}:
                    webhook_msg = WhatsAppWebhookMessage(
                        phone_number=phone_number,
                        message_text=message_text
                    )
                    result = await battery_webhook(webhook_msg)
                    return result
                    
                elif normalized_root_cause in {"MAIN_POWER_CUT", "MAIN_POWER", "MAIN_POWER_CONNECTION"}:
                    webhook_msg = WhatsAppWebhookMessage(
                        phone_number=phone_number,
                        message_text=message_text
                    )
                    result = await main_power_webhook(webhook_msg)
                    return result
                    
                elif normalized_root_cause in {"OTHER_ISSUE", "OTHER"}:
                    webhook_msg = WhatsAppWebhookMessage(
                        phone_number=phone_number,
                        message_text=message_text
                    )
                    result = await other_issue_webhook(webhook_msg)
                    return result
                    
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
    current_state: Optional[str] = None
    payload: Optional[dict] = None
    gps_snapshot: Optional[dict] = None
    collected_json: Optional[dict] = None
    gps_data: Optional[dict] = None
    append_chat_history: Optional[list] = None
    replace_chat_history: Optional[list] = None

@app.put("/api/test/update-session")
async def update_session_for_testing(payload: SessionUpdateRequest):
    """
    Temporary API for testing - allows updating live session data during conversation.

    Use this to simulate state changes, GPS data updates, and chat history updates in a
    running flow without resetting the session.
    """
    try:
        phone = payload.phone_number

        # Get current session
        session = database.get_session(phone)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active session found for {phone}"
            )

        old_state = session.get("current_state")
        old_data = session.get("collected_json", {}).copy()
        collected_json = session.get("collected_json", {})
        history = session.get("chat_history", [])
        new_state = payload.current_state or old_state

        def merge_gps_fields(gps_updates: dict):
            collected_json["gps_data"] = collected_json.get("gps_data", {}) or {}
            for key, value in gps_updates.items():
                if value is None:
                    collected_json["gps_data"].pop(key, None)
                    if key in [
                        "gpstime",
                        "main_powervoltage",
                        "ismainpoerconnected",
                        "gpsStatus",
                        "driver_name",
                        "driver_phone",
                        "current_location",
                        "vehicle_state",
                    ]:
                        collected_json.pop(key, None)
                    continue

                collected_json["gps_data"][key] = value
                if key in [
                    "gpstime",
                    "main_powervoltage",
                    "ismainpoerconnected",
                    "gpsStatus",
                    "driver_name",
                    "driver_phone",
                    "current_location",
                    "vehicle_state",
                ]:
                    collected_json[key] = value

        # Merge payload object updates like GET payload
        if payload.payload:
            for key, value in payload.payload.items():
                if key == "gps_data" and isinstance(value, dict):
                    merge_gps_fields(value)
                    continue
                collected_json[key] = value

        # Merge snapshot object updates like GET gps_snapshot
        if payload.gps_snapshot:
            for key, value in payload.gps_snapshot.items():
                if key == "gps_data" and isinstance(value, dict):
                    merge_gps_fields(value)
                    continue
                if key == "phone_number":
                    continue
                if key == "current_state":
                    new_state = value
                    continue
                collected_json[key] = value

        # Merge general collected_json updates
        if payload.collected_json:
            for key, value in payload.collected_json.items():
                collected_json[key] = value

        # Merge direct gps_data updates if provided
        if payload.gps_data:
            merge_gps_fields(payload.gps_data)

        # Update chat history in real time if requested
        if payload.replace_chat_history is not None:
            history = payload.replace_chat_history
        elif payload.append_chat_history:
            history.extend(payload.append_chat_history)

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
            "chat_history": history,
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


@app.get("/api/test/get-gps-data/{phone_number}")
async def get_gps_data_for_testing(phone_number: str):
    """
    Get the current GPS-related session snapshot for a phone number.

    This is useful during a live flow to inspect the latest GPS data,
    routing state, and user/contact details without modifying the session.
    """
    try:
        session = database.get_session(phone_number)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No session found for {phone_number}"
            )

        collected = session.get("collected_json", {})
        gps_data = collected.get("gps_data", {}) or {}
        fallback_keys = [
            "gpstime",
            "main_powervoltage",
            "ismainpoerconnected",
            "gpsStatus",
            "driver_name",
            "driver_phone",
            "current_location",
            "vehicle_state",
        ]
        for key in fallback_keys:
            if key not in gps_data and collected.get(key) is not None:
                gps_data[key] = collected.get(key)
        if not gps_data:
            gps_data = {
                key: collected.get(key)
                for key in fallback_keys
                if collected.get(key) is not None
            }
        gps_payload = {
            "phone_number": phone_number,
            "vehicle_no": collected.get("vehicle_no"),
            "last_location": collected.get("last_location"),
            "timestamp": collected.get("timestamp"),
            "gps_data": gps_data,
        }

        gps_snapshot = {
            "phone_number": phone_number,
            "current_state": session.get("current_state"),
            "root_cause": collected.get("root_cause"),
            "vehicle_no": collected.get("vehicle_no"),
            "last_location": collected.get("last_location"),
            "timestamp": collected.get("timestamp"),
            "gps_data": gps_data,
            "driver_name": collected.get("driver_name"),
            "driver_phone": collected.get("driver_phone"),
            "current_location": collected.get("current_location"),
            "destination_location": collected.get("destination_location"),
            "vehicle_location": collected.get("vehicle_location"),
            "service_city": collected.get("service_city"),
            "service_city_confirmed": collected.get("service_city_confirmed"),
            "service_date": collected.get("service_date"),
            "resume_date": collected.get("resume_date"),
            "contact_person": collected.get("contact_person"),
            "active_contact_phone": collected.get("active_contact_phone"),
            "contact_mode": collected.get("contact_mode"),
            "status_only": collected.get("status_only", False),
            "standing_hours": collected.get("standing_hours"),
        }

        return {
            "status": "found",
            "phone_number": phone_number,
            "payload": gps_payload,
            "gps_snapshot": gps_snapshot,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching GPS snapshot: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch GPS snapshot: {str(e)}"
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


# ==============================================================================
# 7. USER CRUD APIs
# ==============================================================================

@app.post("/api/users")
async def create_user(payload: OutageRequest):
    """
    Save a user record to the database.
    Accepts the same OutageRequest payload used for outage triggering.
    """
    try:
        data = {
            "phone_number": payload.phone_number,
            "vehicle_no": payload.vehicle_no,
            "last_location": payload.last_location,
            "timestamp": payload.timestamp,
            "gps_data": payload.gps_data.model_dump() if payload.gps_data else {}
        }
        database.save_user(data)
        return {"status": "success", "message": "User saved successfully"}
    except Exception as e:
        logger.error(f"Error saving user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save user: {str(e)}"
        )


@app.get("/api/users")
async def list_users():
    """
    Return all users stored in the database.
    """
    try:
        users = database.get_all_users()
        return {"status": "success", "users": users}
    except Exception as e:
        logger.error(f"Error fetching users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch users: {str(e)}"
        )


@app.get("/api/users/{phone_number}")
async def get_user(phone_number: str):
    """
    Return a single user by phone number.
    Returns 404 if not found.
    """
    try:
        user = database.get_user(phone_number)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with phone number {phone_number} not found"
            )
        return {"status": "success", "user": user}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user {phone_number}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user: {str(e)}"
        )


@app.put("/api/users/{phone_number}")
async def update_user(phone_number: str, payload: OutageRequest):
    """
    Update an existing user record.
    Uses save_user() which performs an UPSERT.
    """
    try:
        data = {
            "phone_number": phone_number,
            "vehicle_no": payload.vehicle_no,
            "last_location": payload.last_location,
            "timestamp": payload.timestamp,
            "gps_data": payload.gps_data.model_dump() if payload.gps_data else {}
        }
        database.save_user(data)
        return {"status": "success", "message": "User updated successfully"}
    except Exception as e:
        logger.error(f"Error updating user {phone_number}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}"
        )


@app.delete("/api/users/{phone_number}")
async def remove_user(phone_number: str):
    """
    Delete a user by phone number.
    """
    try:
        user = database.get_user(phone_number)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with phone number {phone_number} not found"
            )
        database.delete_user(phone_number)
        return {"status": "success", "message": f"User {phone_number} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {phone_number}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


@app.post("/api/users/{phone_number}/trigger")
async def trigger_outage_for_user(phone_number: str):
    """
    Load a stored user by phone number and trigger the outage flow automatically.

    Flow:
        Load user → Analyze root cause → Create session → Send initial WhatsApp message → Start conversation
    """
    try:
        user = database.get_user(phone_number)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with phone number {phone_number} not found"
            )

        # Convert stored gps_data dict back into GpsData model
        gps_data_dict = user.get("gps_data") or {}
        gps_data_model = GpsData(**gps_data_dict) if gps_data_dict else None

        # Build OutageRequest from stored user data
        outage_payload = OutageRequest(
            phone_number=user["phone_number"],
            vehicle_no=user["vehicle_no"] or "",
            last_location=user["last_location"] or "",
            timestamp=user["timestamp"] or "",
            gps_data=gps_data_model
        )

        # Reuse existing trigger_outage logic — no duplication
        logger.info(f"Triggering outage flow for stored user: {phone_number}")
        result = await trigger_outage(outage_payload)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering outage for user {phone_number}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger outage for user: {str(e)}"
        )