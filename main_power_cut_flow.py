import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import requests

import database

logger = logging.getLogger("MainPowerCutFlow")
app = FastAPI(title="GPS Flow - Case 2: Main Power Cut Handler")

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


async def start_main_power_flow(payload: dict):
    """
    Entry point called by main.py routing logic.
    Initializes main power cut flow.
    """
    return await handle_main_power_cut(RoutedRequest(**payload))


def send_whatsapp_meta(to_number: str, text_body: str):
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
            logger.error(f"[META API ERROR] Status {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"[META SEND EXCEPTION] {e}")

@app.post("/api/flow/main-power-cut")
async def handle_main_power_cut(payload: RoutedRequest):
    gps_data = payload.gps_data.model_dump() if payload.gps_data else {}
    gpstime = gps_data.get("gpstime", "N/A")
    last_location = gps_data.get("current_location") or payload.last_location
    vehicle_state = gps_data.get("vehicle_state", "N/A")

    # Build Case 2 specific alert message
    alert_msg = (
        f"Hello,\n\n"
        f"We have analyzed the GPS status for vehicle {payload.vehicle_no}.\n"
        f"Our system indicates that the GPS device may not be receiving main power from the vehicle.\n\n"
        f"Last GPS Update: {gpstime}\n"
        f"Last Known Location: {last_location}\n"
        f"Current Vehicle Status: {vehicle_state}\n\n"
        f"Could you please confirm whether there has been any recent electrical work, "
        f"wiring issue, fuse issue, or power disconnection in the vehicle?\n\n"
        f"Please let us know so we can assist you further."
    )

    # Initialize the database tracking state for Case 2
    database.save_session(
        phone_number=payload.phone_number,
        current_state="INITIAL_ALERT",
        collected_json={
            "battery_issue": False,
            "main_power_issue": True,
            "root_cause": "MAIN_POWER_CUT",
            "gps_gpstime": gpstime,
            "gps_location": last_location,
            "gps_vehicle_state": vehicle_state,
            "driver_name": gps_data.get("driver_name"),
            "driver_phone": gps_data.get("driver_phone"),
            "intent": None,
            "vehicle_location": last_location or None,
            "service_date": None,
            "arrival_date": None,
            "contact_person": None,
            "origin_city": None,
            "destination_city": None,
            "resume_date": None,
            "ticket_id": None,
            "driver_forwarded": False,
        },
        chat_history=[{"role": "bot", "text": alert_msg}]
    )

    send_whatsapp_meta(payload.phone_number, alert_msg)
    return {"status": "flow_initialized", "case": "MAIN_POWER_CUT"}