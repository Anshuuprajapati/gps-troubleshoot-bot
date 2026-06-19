import sqlite3
import json

DB_NAME = "bot_sessions.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                phone_number TEXT PRIMARY KEY,
                current_state TEXT DEFAULT 'INITIAL_ALERT',
                collected_json TEXT DEFAULT '{}',
                chat_history TEXT DEFAULT '[]'
            )
        """)
        conn.commit()

def get_session(phone_number: str) -> dict:
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE phone_number = ?", (phone_number,))
        row = cursor.fetchone()
        
        if row:
            return {
                "phone_number": row["phone_number"],
                "current_state": row["current_state"],
                "collected_json": json.loads(row["collected_json"]),
                "chat_history": json.loads(row["chat_history"])
            }
        else:
            return {
                "phone_number": phone_number,
                "current_state": "INITIAL_ALERT",
                "collected_json": {"vehicle_location": None, "service_date": None, "driver_phone": None},
                "chat_history": []
            }

def save_session(phone_number: str, state: str, collected_json: dict, chat_history: list):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (phone_number, current_state, collected_json, chat_history)
            VALUES (?, ?, ?, ?)
        """, (phone_number, state, json.dumps(collected_json), json.dumps(chat_history)))
        conn.commit()