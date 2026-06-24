"""
gps_analysis_api.py — Pre-analysis wrapper for GPS device status.

Called BEFORE the first WhatsApp message is sent. Queries the GPS analysis
endpoint and classifies the root cause into one of three categories:

    BATTERY_ISSUE   — battery disconnect or battery discharge
    MAIN_POWER_CUT  — main/external power supply has been cut
    OTHER_ISSUE     — neither of the above; user must describe the problem

The function returns a dict conforming to the session schema extension:
    {
        "battery_issue":    bool,
        "main_power_issue": bool,
        "root_cause":       "BATTERY_ISSUE" | "MAIN_POWER_CUT" | "OTHER_ISSUE"
    }

Integration notes
-----------------
* Set GPS_ANALYSIS_API_URL in your .env to point at the real endpoint.
* The real API is expected to return a JSON body that contains at minimum:
      { "battery_disconnect": bool, "battery_discharge": bool,
        "main_power_cut": bool, ... }
  If the shape differs, adjust _parse_api_response() below.
* On any network / parse error the function falls back to OTHER_ISSUE so the
  conversation can still proceed.
"""

import os
import requests
from typing import TypedDict


class GpsAnalysisResult(TypedDict):
    battery_issue: bool
    main_power_issue: bool
    root_cause: str  # "BATTERY_ISSUE" | "MAIN_POWER_CUT" | "OTHER_ISSUE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_api_response(data: dict) -> GpsAnalysisResult:
    """
    Translate the raw API JSON into a normalised GpsAnalysisResult.

    Adjust the key names below if your actual API uses different field names.
    """
    battery_disconnect: bool = bool(data.get("battery_disconnect", False))
    battery_discharge: bool  = bool(data.get("battery_discharge",  False))
    main_power_cut: bool     = bool(data.get("main_power_cut",     False))

    battery_issue   = battery_disconnect or battery_discharge
    main_power_issue = main_power_cut

    if battery_issue:
        root_cause = "BATTERY_ISSUE"
    elif main_power_issue:
        root_cause = "MAIN_POWER_CUT"
    else:
        root_cause = "OTHER_ISSUE"

    return GpsAnalysisResult(
        battery_issue=battery_issue,
        main_power_issue=main_power_issue,
        root_cause=root_cause,
    )


def _fallback_result() -> GpsAnalysisResult:
    """Returned whenever the API call fails so the flow can still continue."""
    return GpsAnalysisResult(
        battery_issue=False,
        main_power_issue=False,
        root_cause="OTHER_ISSUE",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_gps_device(vehicle_no: str, phone_number: str) -> GpsAnalysisResult:
    """
    Call the external GPS analysis API and return a classified result.

    Parameters
    ----------
    vehicle_no   : Registration / asset number of the vehicle.
    phone_number : Owner / fleet manager contact (may be used by the API for
                   audit / lookup purposes).

    Returns
    -------
    GpsAnalysisResult with battery_issue, main_power_issue, root_cause.
    """
    api_url = os.getenv("GPS_ANALYSIS_API_URL", "").strip()

    if not api_url:
        # URL not configured — treat as OTHER_ISSUE and log a warning.
        print("[GPS ANALYSIS] GPS_ANALYSIS_API_URL not set. Defaulting to OTHER_ISSUE.")
        return _fallback_result()

    api_key = os.getenv("GPS_ANALYSIS_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "vehicle_no":    vehicle_no,
        "phone_number":  phone_number,
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=int(os.getenv("GPS_ANALYSIS_TIMEOUT_SECONDS", "10")),
        )
        response.raise_for_status()
        data = response.json()
        result = _parse_api_response(data)
        print(
            f"[GPS ANALYSIS] vehicle={vehicle_no} | "
            f"battery_issue={result['battery_issue']} | "
            f"main_power_issue={result['main_power_issue']} | "
            f"root_cause={result['root_cause']}"
        )
        return result

    except requests.exceptions.Timeout:
        print(f"[GPS ANALYSIS] Timeout calling {api_url}. Defaulting to OTHER_ISSUE.")
        return _fallback_result()

    except requests.exceptions.RequestException as e:
        print(f"[GPS ANALYSIS] Request error: {e}. Defaulting to OTHER_ISSUE.")
        return _fallback_result()

    except (ValueError, KeyError) as e:
        print(f"[GPS ANALYSIS] Response parse error: {e}. Defaulting to OTHER_ISSUE.")
        return _fallback_result()