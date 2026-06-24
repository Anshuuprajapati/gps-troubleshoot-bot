import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import requests

import database

logger = logging.getLogger("OtherIssueFlow")
app = FastAPI(title="GPS Flow - Case 3: Other Issues Handler")

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


async def start_other_issue_flow(payload: dict):
    """
    Entry point called by main.py routing logic.
    Initializes other issue flow.
    """
    return await handle_other_issue(RoutedRequest(**payload))


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

@app.post("/api/flow/other-issue")
async def handle_other_issue(payload: RoutedRequest):
    gps_data = payload.gps_data.model_dump() if payload.gps_data else {}
    gpstime = gps_data.get("gpstime", "N/A")
    driver_name = gps_data.get("driver_name", "N/A")
    driver_phone = gps_data.get("driver_phone", "N/A")
    cur_location = gps_data.get("current_location") or payload.last_location
    vehicle_state = gps_data.get("vehicle_state", "N/A")

    # Build Case 3 specific option list message
    alert_msg = (
        f"Hello,\n\n"
        f"We have analyzed the GPS status for vehicle {payload.vehicle_no}.\n"
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

    # Initialize the database tracking state for Case 3
    database.save_session(
        phone_number=payload.phone_number,
        current_state="INITIAL_ALERT",
        collected_json={
            "battery_issue": False,
            "main_power_issue": False,
            "root_cause": "OTHER_ISSUE",
            "gps_gpstime": gpstime,
            "gps_location": cur_location,
            "gps_vehicle_state": vehicle_state,
            "driver_name": driver_name,
            "driver_phone": driver_phone,
            "intent": None,
            "vehicle_location": cur_location or None,
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
    return {"status": "flow_initialized", "case": "OTHER_ISSUE"}