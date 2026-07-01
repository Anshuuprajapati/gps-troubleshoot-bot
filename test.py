import asyncio
import os
from unittest.mock import patch

import battery_issue_flow as battery_flow
import main
import main_power_cut_flow as main_power_flow
import other_issue_flow as other_flow


class ScenarioFailure(AssertionError):
    pass


class FakeSessionStore:
    def __init__(self):
        self.sessions = {}

    def get_session(self, phone_number):
        return self.sessions.get(phone_number)

    def save_session(self, phone_number, current_state, collected_json, chat_history):
        self.sessions[phone_number] = {
            "current_state": current_state,
            "collected_json": collected_json,
            "chat_history": chat_history,
        }

    def delete_session(self, phone_number):
        self.sessions.pop(phone_number, None)


class FakeWhatsAppRequest:
    def __init__(self, phone_number, message_text):
        self.phone_number = phone_number
        self.message_text = message_text

    async def json(self):
        return {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {"from": self.phone_number, "text": {"body": self.message_text}}
                                ]
                            }
                        }
                    ]
                }
            ]
        }


def assert_contains(text, expected_substring, context):
    if expected_substring not in text:
        raise ScenarioFailure(f"{context}: expected to find {expected_substring!r} in {text!r}")


async def run_flow_scenario(module, scenario_name, state, context, brain_result, expected_state=None, expected_reply=None, allowed_states=None):
    store = FakeSessionStore()
    phone = f"{scenario_name}@test"
    store.save_session(
        phone,
        state,
        context,
        [{"role": "user", "text": "start"}],
    )

    sent_messages = []

    def fake_get_session(p):
        return store.get_session(p)

    def fake_save_session(p, new_state, collected_json, chat_history):
        store.save_session(p, new_state, collected_json, chat_history)

    def fake_send_whatsapp(to_number, text_body):
        sent_messages.append((to_number, text_body))

    def fake_call_brain(state_context, chat_hist, message):
        return brain_result

    def fake_get_backend_gps_snapshot(p):
        return {"gpsStatus": 0, "payload": {"gps_data": {"ismainpoerconnected": "0"}}}

    patchers = [
        patch.object(module.database, "get_session", side_effect=fake_get_session),
        patch.object(module.database, "save_session", side_effect=fake_save_session),
        patch.object(module.database, "delete_session", side_effect=store.delete_session),
        patch.object(module, "send_whatsapp_meta", side_effect=fake_send_whatsapp),
        patch.object(module, "get_backend_gps_snapshot", side_effect=fake_get_backend_gps_snapshot),
    ]
    if hasattr(module, "call_brain"):
        patchers.append(patch.object(module, "call_brain", side_effect=fake_call_brain))

    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
        if len(patchers) > 5:
            with patchers[5]:
                result = await module.handle_whatsapp_replies(
                    module.WhatsAppWebhookMessage(phone_number=phone, message_text="sample")
                )
        else:
            result = await module.handle_whatsapp_replies(
                module.WhatsAppWebhookMessage(phone_number=phone, message_text="sample")
            )

    session = store.get_session(phone)
    if not session:
        raise ScenarioFailure(f"{scenario_name}: no session saved")

    if expected_state is not None:
        accepted_states = {expected_state}
        if allowed_states:
            accepted_states.update(allowed_states)
        if session["current_state"] not in accepted_states:
            raise ScenarioFailure(f"{scenario_name}: expected state in {accepted_states}, got {session['current_state']}")

    if expected_reply is not None and not any(expected_reply in body for _, body in sent_messages):
        raise ScenarioFailure(f"{scenario_name}: expected reply containing {expected_reply!r} but got {sent_messages}")

    return result, session, sent_messages


async def run_main_routing_scenario(scenario_name, root_cause, expected_flow):
    store = FakeSessionStore()
    phone = f"{scenario_name}@test"
    store.save_session(
        phone,
        "INITIAL_ALERT",
        {"root_cause": root_cause, "vehicle_no": "MH12AB1234"},
        [{"role": "user", "text": "start"}],
    )
    routed = []

    async def fake_battery_webhook(payload):
        routed.append("battery")
        return {"status": "ok"}

    async def fake_main_webhook(payload):
        routed.append("main")
        return {"status": "ok"}

    async def fake_other_webhook(payload):
        routed.append("other")
        return {"status": "ok"}

    with patch.object(main.database, "get_session", side_effect=store.get_session), \
         patch.object(main, "battery_webhook", side_effect=fake_battery_webhook), \
         patch.object(main, "main_power_webhook", side_effect=fake_main_webhook), \
         patch.object(main, "other_issue_webhook", side_effect=fake_other_webhook):
        request = FakeWhatsAppRequest(phone, "hello")
        result = await main.whatsapp_webhook(request)

    if result.get("status") != "ok":
        raise ScenarioFailure(f"{scenario_name}: unexpected route result {result}")
    if routed != [expected_flow]:
        raise ScenarioFailure(f"{scenario_name}: expected route {expected_flow}, got {routed}")
    return result


async def main_async():
    scenarios = []

    # Battery flow scenarios
    scenarios.append((
        "battery_self_check",
        battery_flow,
        battery_flow.ST_INITIAL_ALERT,
        {"vehicle_no": "MH12AB1234", "driver_name": "Ravi", "driver_phone": "9999999999"},
        {"wants_self_check": True, "wants_driver": False, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        battery_flow.ST_SELF_CHECK_WAITING,
        None,
    ))
    scenarios.append((
        "battery_driver_redirect",
        battery_flow,
        battery_flow.ST_INITIAL_ALERT,
        {"vehicle_no": "MH12AB1234", "driver_name": "Ravi", "driver_phone": "9999999999"},
        {"wants_self_check": False, "wants_driver": True, "driver_name": None, "driver_phone": "7777777777", "confirms_existing_driver": None, "work_done": None, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        battery_flow.ST_STATUS_ONLY,
        "Dhanyavaad Sir",
    ))
    scenarios.append((
        "battery_driver_confirmation",
        battery_flow,
        battery_flow.ST_DRIVER_CONFIRMATION,
        {"vehicle_no": "MH12AB1234", "driver_name": "Ravi", "driver_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": True, "work_done": None, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        battery_flow.ST_STATUS_ONLY,
        None,
    ))
    scenarios.append((
        "battery_off_topic",
        battery_flow,
        battery_flow.ST_SELF_CHECK_WAITING,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": True, "conversational_reply": "Main issue ke baare mein batayein"},
        battery_flow.ST_SELF_CHECK_WAITING,
        "Main issue",
    ))
    scenarios.append((
        "battery_unformatted_date",
        battery_flow,
        battery_flow.ST_COLLECTING_RESUME_DATE,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": "5 July", "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        battery_flow.ST_COLLECTING_RESUME_DATE,
        "05-07",
        [battery_flow.ST_CASE_CLOSED, battery_flow.ST_TICKET_CREATED],
    ))
    scenarios.append((
        "battery_llm_ack",
        battery_flow,
        battery_flow.ST_DRIVER_WAITING,
        {"vehicle_no": "MH12AB1234", "active_contact_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": True, "battery_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        battery_flow.ST_BATTERY_DAMAGE_CHECK,
        None,
        [battery_flow.ST_VEHICLE_STATUS_CHECK],
    ))

    # Main power scenarios
    scenarios.append((
        "main_self_check",
        main_power_flow,
        main_power_flow.ST_INITIAL_ALERT,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": True, "wants_driver": False, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        main_power_flow.ST_SELF_CHECK_WAITING,
        None,
    ))
    scenarios.append((
        "main_driver_redirect",
        main_power_flow,
        main_power_flow.ST_INITIAL_ALERT,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": False, "wants_driver": True, "driver_name": None, "driver_phone": "7777777777", "confirms_existing_driver": None, "work_done": None, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        main_power_flow.ST_STATUS_ONLY,
        "Dhanyavaad Sir",
    ))
    scenarios.append((
        "main_driver_confirmation",
        main_power_flow,
        main_power_flow.ST_DRIVER_CONFIRMATION,
        {"vehicle_no": "MH12AB1234", "driver_name": "Ravi", "driver_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": True, "work_done": None, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        main_power_flow.ST_STATUS_ONLY,
        None,
    ))
    scenarios.append((
        "main_off_topic",
        main_power_flow,
        main_power_flow.ST_WIRING_DAMAGE_CHECK,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": True, "conversational_reply": "Main issue ke baare mein batayein"},
        main_power_flow.ST_WIRING_DAMAGE_CHECK,
        "Main issue",
    ))
    scenarios.append((
        "main_unformatted_date",
        main_power_flow,
        main_power_flow.ST_COLLECTING_RESUME_DATE,
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": "3 din baad", "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        main_power_flow.ST_COLLECTING_RESUME_DATE,
        "Dhanyavaad",
        [main_power_flow.ST_CASE_CLOSED, main_power_flow.ST_TICKET_CREATED],
    ))
    scenarios.append((
        "main_llm_ack",
        main_power_flow,
        main_power_flow.ST_DRIVER_WAITING,
        {"vehicle_no": "MH12AB1234", "active_contact_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": True, "wiring_damaged": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        main_power_flow.ST_WIRING_DAMAGE_CHECK,
        None,
        [main_power_flow.ST_VEHICLE_STATUS_CHECK],
    ))

    # Other issue scenarios
    scenarios.append((
        "other_self_check",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": True, "wants_driver": False, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        "INITIAL_ALERT",
        None,
    ))
    scenarios.append((
        "other_driver_redirect",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": False, "wants_driver": True, "driver_name": None, "driver_phone": "7777777777", "confirms_existing_driver": None, "work_done": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        "INITIAL_ALERT",
        "Kripya vehicle ki sthiti short me spasht karein",
        ["STATUS_ONLY"],
    ))
    scenarios.append((
        "other_driver_confirmation",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234", "driver_name": "Ravi", "driver_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": True, "work_done": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        "INITIAL_ALERT",
        None,
    ))
    scenarios.append((
        "other_off_topic",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": True, "conversational_reply": "Main issue ke baare mein batayein"},
        "INITIAL_ALERT",
        "Main issue ke baare mein batayein",
    ))
    scenarios.append((
        "other_unformatted_date",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": None, "vehicle_status_intent": None, "expected_date": "25/07/2026", "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        "INITIAL_ALERT",
        "Kripya vehicle ki sthiti short me spasht karein",
        ["STATUS_ONLY", "TICKET_CREATED"],
    ))
    scenarios.append((
        "other_llm_ack",
        other_flow,
        "INITIAL_ALERT",
        {"vehicle_no": "MH12AB1234", "active_contact_phone": "9999999999"},
        {"wants_self_check": None, "wants_driver": None, "driver_name": None, "driver_phone": None, "confirms_existing_driver": None, "work_done": True, "vehicle_status_intent": None, "expected_date": None, "vehicle_location": None, "is_off_topic": False, "conversational_reply": ""},
        "INITIAL_ALERT",
        None,
        ["INITIAL_ALERT", "STATUS_ONLY"],
    ))

    # Routing smoke checks
    scenarios.append(("main_routing_battery", None, None, None, None, None, None, None))
    scenarios.append(("main_routing_main", None, None, None, None, None, None, None))

    for index, case in enumerate(scenarios, 1):
        name = case[0]
        if name == "main_routing_battery":
            await run_main_routing_scenario(name, "BATTERY_LOW", "battery")
            print(f"[PASS] {index}. {name}")
            continue
        if name == "main_routing_main":
            await run_main_routing_scenario(name, "MAIN_POWER", "main")
            print(f"[PASS] {index}. {name}")
            continue

        module = case[1]
        state = case[2]
        context = case[3]
        brain_result = case[4]
        expected_state = case[5]
        expected_reply = case[6]
        allowed_states = case[7] if len(case) > 7 else None
        await run_flow_scenario(module, name, state, context, brain_result, expected_state=expected_state, expected_reply=expected_reply, allowed_states=allowed_states)
        print(f"[PASS] {index}. {name}")

    # Direct date normalization checks for unformatted dates
    assert battery_flow.resolve_expected_date("5 July") == "05-07-2026" or battery_flow.resolve_expected_date("5 July").startswith("05-07")
    assert main_power_flow.resolve_expected_date("3 din baad").startswith("0") or main_power_flow.resolve_expected_date("3 din baad") == "03-07-2026"
    assert other_flow._resolve_service_date("25/07/2026") == "25-07-2026"
    print("[PASS] direct date normalization checks")


if __name__ == "__main__":
    asyncio.run(main_async())
    print("All 20 scenarios passed.")
