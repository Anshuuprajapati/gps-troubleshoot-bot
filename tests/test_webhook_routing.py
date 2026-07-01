import importlib

import main


def test_root_cause_aliases_route_to_battery_flow(monkeypatch):
    called = {}

    async def fake_battery_webhook(payload):
        called['flow'] = 'battery'
        return {'status': 'ok'}

    monkeypatch.setattr(main, 'battery_webhook', fake_battery_webhook)
    monkeypatch.setattr(main.database, 'get_session', lambda phone: {
        'collected_json': {'root_cause': 'BATTERY_LOW'}
    })

    class Req:
        async def json(self):
            return {
                'entry': [{'changes': [{'value': {'messages': [{'from': '123', 'text': {'body': 'hi'}}]}}]}]
            }

    import asyncio
    result = asyncio.run(main.whatsapp_webhook(Req()))

    assert result['status'] == 'ok'
    assert called['flow'] == 'battery'
