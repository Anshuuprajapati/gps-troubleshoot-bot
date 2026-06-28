import json, asyncio
import database
from main import trigger_outage, OutageRequest

database.init_db()
payload = {
    'phone_number': '918882374849',
    'vehicle_no': 'MH12AB1234',
    'last_location': 'Mumbai',
    'timestamp': '2026-06-19 18:00:00',
    'gps_data': {
        'gpstime': '06 June 2026 18:14',
        'main_powervoltage': 12,
        'ismainpoerconnected': '1',
        'gpsStatus': 0,
        'driver_name': 'Salman',
        'driver_phone': '9105853736',
        'current_location': 'Noida',
        'vehicle_state': 'Running'
    }
}
request = OutageRequest(**payload)
res = asyncio.run(trigger_outage(request))
print('trigger response:', res)
print('session:', json.dumps(database.get_session('918882374849'), indent=2))
