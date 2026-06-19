# 🚗 GPS Troubleshoot Bot

An intelligent WhatsApp chatbot powered by Groq AI that automates GPS downtime troubleshooting for vehicle fleet management. The bot intelligently collects issue information and creates service tickets with minimal user friction.

**Status**: ✅ Fully tested and production-ready

---

## 🎯 Features

### Smart Conversation Flow
- **8 Intent Types**: Recognizes workshop, accidents, GPS issues, battery problems, and more
- **Natural Language Understanding**: Accepts both option numbers and free-form Hinglish descriptions
- **Intent Locking**: Once an issue is identified, bot stays focused on that flow
- **Smart Extraction**: Automatically pulls phone numbers, dates, and locations from natural text

### Intelligent Data Collection
- **Adaptive Questioning**: Only asks for missing information
- **Date Normalization**: Converts "kal", "parso", bare numbers, and various formats to ISO dates
- **Route Understanding**: Correctly identifies destination vs origin cities
- **Multi-turn Context**: Maintains full conversation history across messages

### Two Workflow Types
1. **Case Closed** (Simple Issues): Workshop, accidents, battery, GPS removal
   - Asks only for resume date
   - Closes with confirmation
   - No ticket needed

2. **Ticket Required** (Complex Issues): GPS damaged, running/not updating, vehicle standing
   - Collects: location, service date, driver phone
   - Auto-generates ticket ID
   - Persists in SQLite database

### Built-in Safeguards
- ✅ Duplicate message prevention
- ✅ Intent locking (can't switch flows mid-conversation)
- ✅ All 3 fields validated before ticket creation
- ✅ Deterministic date normalization (no silent failures)
- ✅ Session persistence across disconnects

---

## 📋 Prerequisites

### Required
- **Python 3.10+**
- **Groq API Key** (free tier available at [console.groq.com](https://console.groq.com))
- **Meta WhatsApp Business Account** (for production) or test locally

### Optional
- **SQLite3** (usually included with Python)
- **Ngrok** (for tunneling webhooks during development)

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/Anshuuprajapati/gps-troubleshoot-bot.git
cd gps-troubleshoot-bot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file in project root:
```bash
GROQ_API_KEY=your_groq_api_key_here
WHATSAPP_TOKEN=your_meta_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_id_here
DB_PATH=gps_bot.db
```

**Get Groq API Key** (FREE):
1. Visit [console.groq.com](https://console.groq.com/keys)
2. Create API key
3. Copy to `.env`

### 3. Run the Server
```bash
# Start FastAPI server
uvicorn main:app --reload --port 8000

# Bot is now running on http://localhost:8000
```

### 4. Test Locally (No WhatsApp Needed)
```bash
# Run full test suite (10 scenarios)
python test_bot.py

# Run specific scenario
python test_bot.py --scenario 1

# Run with verbose output
python test_bot.py --verbose
```

---

## 📊 Project Structure

```
gps-troubleshoot-bot/
├── main.py                 # FastAPI server + core bot logic
├── database.py             # SQLite session/ticket management
├── date_utils.py           # Date normalization utilities
├── test_bot.py             # End-to-end test suite (10 scenarios)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
├── README.md              # This file
├── BUG_FIXES_SUMMARY.md   # Recent fixes documentation
├── gps_bot.db             # SQLite database (auto-created)
└── test_bot.db            # Test database (auto-created)
```

---

## 🧪 Testing

### Run All Scenarios
```bash
python test_bot.py
```

### Test Scenarios Included (10 total)

| # | Scenario | Type | Tests |
|---|----------|------|-------|
| 1 | Workshop / Service Center → Case Closed | Simple | ✅ Resume date capture |
| 2 | Accident → Case Closed | Simple | ✅ Resume date capture |
| 3 | GPS Damaged → Step-by-step Ticket | Complex | ✅ Multi-turn collection |
| 4 | All Info in One Message → Instant Ticket | Complex | ✅ Full extraction |
| 5 | Route Understanding — Destination = Service Location | Route | ✅ City parsing |
| 6 | Smart Phone Extraction + No Duplicate Ask | Extraction | ✅ No redundant questions |
| 7 | Date Normalization — bare number, kal, parso | Dates | ✅ Multiple date formats |
| 8 | Natural Language Intent — No Option Number | NLP | ✅ Free-form input |
| 9 | Intent Lock — Flow Cannot Be Switched | Safety | ✅ Intent consistency |
| 10 | Side Question Handling — Bot Stays on Track | Robustness | ✅ Off-topic resilience |

### Run Single Scenario
```bash
python test_bot.py --scenario 5
```

### Verbose Testing
```bash
python test_bot.py --scenario 3 --verbose
```

### Example Test Output
```
=================================================================
   GPS Support Bot — Full Flow Test Runner
   Date: 2026-06-20
=================================================================

─────────────────────────────────────────────────────────────────
  Scenario 1: Workshop / Service Center → Case Closed
  User picks option 1. Bot asks resume date only. No ticket created.
─────────────────────────────────────────────────────────────────

  Turn 1
  👤 User: 1

  🤖 Bot: Aapki vehicle workshop mein hai. Vehicle dobara kab running 
          condition mein aa jayegi?
  ✅ State: COLLECTING_DETAILS

  Turn 2
  👤 User: Kal tak aa jayegi

  🤖 Bot: ✅ Update note kar liya gaya hai. Dhanyavaad.
  ✅ State: CASE_CLOSED

=================================================================
  RESULTS SUMMARY
=================================================================
  ✅ PASS  Workshop / Service Center → Case Closed

  Total: 1  |  Passed: 1  |  Failed: 0
=================================================================
```

---

## 🏗️ Architecture

### Message Processing Pipeline

```
Incoming Message
      ↓
[Duplicate Check] → Prevent re-processing same message_id
      ↓
[Groq AI Analysis] → Extract intent, data, determine next state
      ↓
[Post-Processing]
  ├─ Date Normalization (kal → 2026-06-21)
  ├─ Route Resolution (destination = location)
  └─ Keyword Safety Net (ensure "kahan", "kab", etc.)
      ↓
[Smart Merge] → Preserve existing data, add new fields
      ↓
[State Machine]
  ├─ INITIAL_ALERT → User sees 8 options
  ├─ COLLECTING_DETAILS → Bot asks for missing fields
  ├─ CASE_CLOSED → Simple issue resolved
  └─ TICKET_RAISED → Complex issue with ticket
      ↓
[Persistence]
  ├─ Save session to SQLite
  ├─ Create ticket (if needed)
  └─ Mark message as processed
      ↓
[Send Reply] → User receives bot response
```

### Data Flow

```
User Input
    ↓
Session Retrieval (SQLite)
    ↓
Groq API Call (llama-3.3-70b-versatile)
    ↓
JSON Parsing + Validation
    ↓
Extracted Data Post-Processing
    ↓
Smart Merge with Existing Data
    ↓
State Transition Logic
    ↓
Database Update
    ↓
Response Generation
    ↓
WhatsApp Send (or mock in tests)
```

---

## 🔄 State Machine

```
INITIAL_ALERT
    ↓
[User selects intent]
    ↓
COLLECTING_DETAILS ← (Intent locked here)
    ├─ Case Close Flow (resume_date)
    │   ↓
    │   CASE_CLOSED
    │
    └─ Ticket Flow (location, date, phone)
        ├─ [All 3 fields collected]
        ↓
        TICKET_RAISED
```

---

## 📱 Intent Types & Flows

### Case Close Intents (No Ticket)
```
1️⃣ Workshop / Service Center
2️⃣ Accident
3️⃣ Battery Disconnect
4️⃣ GPS Removed
```
**Bot asks**: "Vehicle dobara kab running condition mein aa jayegi?"

### Ticket Required Intents
```
5️⃣ GPS Damaged
6️⃣ Vehicle Running but GPS Not Updating
7️⃣ Vehicle Standing
8️⃣ Other
```
**Bot asks**:
1. Vehicle location (kahan)
2. Service date (kab)
3. Driver phone (contact number)

---

## 🗂️ Database Schema

### Tables

#### `sessions` Table
```sql
CREATE TABLE sessions (
    phone_number    TEXT PRIMARY KEY,
    current_state   TEXT,                    -- INITIAL_ALERT, COLLECTING_DETAILS, etc.
    collected_json  TEXT,                    -- Extracted data (JSON)
    chat_history    TEXT,                    -- Conversation history (JSON array)
    created_at      TEXT,                    -- ISO timestamp
    updated_at      TEXT
)
```

#### `tickets` Table
```sql
CREATE TABLE tickets (
    ticket_id       TEXT PRIMARY KEY,        -- TKT-1234
    phone_number    TEXT,
    vehicle_location TEXT,
    service_date    TEXT,                    -- ISO date (YYYY-MM-DD)
    driver_phone    TEXT,
    status          TEXT,                    -- OPEN, CLOSED, REOPENED
    created_at      TEXT,
    updated_at      TEXT
)
```

#### `processed_messages` Table
```sql
CREATE TABLE processed_messages (
    message_id  TEXT PRIMARY KEY,
    received_at TEXT
)
```

---

## 📅 Date Normalization

The bot intelligently converts various date formats to ISO (YYYY-MM-DD):

| Input | Converts To | Example |
|-------|-------------|---------|
| "kal" | Tomorrow | 2026-06-21 |
| "parso" | Day after tomorrow | 2026-06-22 |
| "aaj" | Today | 2026-06-20 |
| "25" | 25th of current/next month | 2026-06-25 |
| "25 June" | That date, next year if passed | 2026-06-25 |
| "DD-MM-YYYY" | Parsed directly | 2026-06-21 |
| "DD/MM/YYYY" | Parsed directly | 2026-06-21 |
| "25th" | Bare day number | 2026-06-25 |

---

## 🔐 Security & Reliability

### Data Integrity
- ✅ **Duplicate Prevention**: Message IDs tracked to prevent re-processing
- ✅ **Transaction Safety**: All database writes use transactions
- ✅ **Data Merging**: Never overwrites existing non-null fields
- ✅ **Validation Guards**: Ensures all required fields before ticket creation

### Error Handling
- ✅ **Groq API Errors**: Catches and logs with specific error types
- ✅ **Rate Limit Detection**: Identifies 429 errors and advises retry
- ✅ **JSON Parsing**: Validates Groq responses before processing
- ✅ **Missing Fields**: Gracefully continues instead of crashing

### Session Management
- ✅ **Persistence**: Sessions survive server restarts
- ✅ **Intent Locking**: Cannot switch issue type mid-conversation
- ✅ **Chat History**: Full conversation maintained per session
- ✅ **Auto-Recovery**: Sessions automatically restored from DB

---

## ⚙️ Configuration

### Environment Variables

```bash
# Groq AI Configuration
GROQ_API_KEY=gsk_...                    # Free API key from console.groq.com

# Meta WhatsApp Configuration (for production)
WHATSAPP_TOKEN=EAABsbCS...              # Meta Graph API token
WHATSAPP_PHONE_NUMBER_ID=1234567890    # Your WhatsApp Business Phone ID

# Database
DB_PATH=gps_bot.db                      # SQLite database file path
```

### Optional Customization

Edit `main.py` to customize:
```python
# Model selection (line 17)
model="llama-3.3-70b-versatile"         # Can use other Groq models

# Temperature (line 31)
temperature=0.1                          # Lower = more deterministic

# System prompt (line 65+)
SYSTEM_INSTRUCTION = """..."""           # Customize bot behavior
```

---

## 🐛 Known Issues & Troubleshooting

### Issue: "Rate limit exceeded" (429 Error)
**Cause**: Groq free tier has 100K tokens/day limit
**Solution**: 
- Wait for daily reset (UTC midnight)
- Upgrade to Groq Pro tier
- Use mock testing mode

### Issue: Tests passing but bot not working on WhatsApp
**Cause**: Missing Meta WhatsApp token or phone ID
**Solution**: 
- Get tokens from Meta for Developers dashboard
- Set `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` in `.env`
- Verify webhook URL is publicly accessible

### Issue: "ModuleNotFoundError: No module named 'groq'"
**Cause**: Dependencies not installed
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: Bot not responding to WhatsApp messages
**Cause**: Webhook endpoint not verified
**Solution**:
1. Meta sends GET request to `/webhook`
2. Verify token and respond with challenge
3. Check `main.py` webhook handler (line ~420)

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.115.0 | Web framework |
| uvicorn | 0.30.1 | ASGI server |
| groq | 0.18.0 | Groq AI API client |
| httpx | 0.27.0 | HTTP client |
| pydantic | 2.9.2 | Data validation |
| requests | 2.32.3 | HTTP requests |
| python-dotenv | 1.0.1 | Environment variables |

---

## 🚢 Deployment

### Local Development
```bash
python -m uvicorn main:app --reload --port 8000
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Ngrok Tunneling (Development)
```bash
# Install ngrok
choco install ngrok  # Windows

# Create tunnel
ngrok http 8000

# Get public URL and set in Meta Webhooks
# https://abc123.ngrok.io/webhook
```

---

## 📞 API Endpoints

### Webhooks

#### `GET /webhook`
**Purpose**: Meta WhatsApp webhook verification

**Parameters**:
- `hub.mode=subscribe`
- `hub.challenge=<challenge_string>`
- `hub.verify_token=<verify_token>`

**Response**: 200 OK with challenge string

#### `POST /webhook`
**Purpose**: Receive WhatsApp messages from Meta

**Payload**:
```json
{
  "object": "whatsapp_business_account",
  "entry": [{
    "changes": [{
      "value": {
        "messages": [{
          "from": "919876543210",
          "id": "wamid.xxx",
          "text": { "body": "User message here" }
        }]
      }
    }]
  }]
}
```

### Trigger Endpoints

#### `POST /api/trigger-outage`
**Purpose**: Manually trigger GPS outage alert

**Payload**:
```json
{
  "phone_number": "919876543210",
  "vehicle_no": "MH12AA0001",
  "last_location": "Mumbai",
  "timestamp": "2026-06-20 18:00:00"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Alert sent to 919876543210"
}
```

---

## 📊 Example Conversations

### Example 1: Workshop Case (Simple)
```
Bot: Vehicle MH12AA0001 se GPS data receive nahi ho raha hai.
     📍 Last Known Location: Mumbai
     Kripya batayein ki aapki vehicle ki current status kya hai:
     1️⃣ Workshop / Service Center
     2️⃣ Accident
     ... (8 options)

User: 1

Bot: Aapki vehicle workshop mein hai. Vehicle dobara kab running 
     condition mein aa jayegi?

User: Kal tak aa jayegi

Bot: ✅ Update note kar liya gaya hai. Dhanyavaad.
     [Conversation ends - Case CLOSED]
```

### Example 2: GPS Damaged (Complex)
```
Bot: Vehicle status kya hai?

User: GPS damaged, koi signal nahi aa raha

Bot: Aapki vehicle kahan hai?

User: Pune mein, Wakad area

Bot: Service kab chahiye?

User: Kal service chahiye

Bot: Driver ka contact number kya hai?

User: 9876543210

Bot: ✅ Service request create kar di gayi hai!
     📋 Ticket Details:
     🎫 Ticket ID: TKT-4821
     📍 Location: Pune
     📅 Service Date: 2026-06-21
     📞 Contact: 9876543210
     [Ticket saved in database]
```

---

## 📖 Documentation

- [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md) - Recent fixes and improvements
- [main.py](main.py) - Core bot logic (well-commented)
- [date_utils.py](date_utils.py) - Date normalization rules
- [database.py](database.py) - Database operations

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📧 Support

For issues, questions, or suggestions:
- GitHub Issues: [Create Issue](https://github.com/Anshuuprajapati/gps-troubleshoot-bot/issues)
- Email: support@example.com

---

## 🎓 Learning Resources

- [Groq AI Documentation](https://console.groq.com/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/)
- [SQLite Guide](https://www.sqlite.org/quickstart.html)
- [Meta WhatsApp API](https://developers.facebook.com/docs/whatsapp/cloud-api)

---

## 🎉 Acknowledgments

Built with:
- ❤️ Groq AI (llama-3.3-70b-versatile model)
- 🚀 FastAPI & Uvicorn
- 💾 SQLite
- 🔗 Meta WhatsApp API

---

**Made with ❤️ for better GPS troubleshooting automation**

Last Updated: June 20, 2026 | Version: 1.0.0 ✅
