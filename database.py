import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "gps_bot.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all required database tables."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # ── Sessions table ─────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                phone_number    TEXT PRIMARY KEY,
                current_state   TEXT NOT NULL DEFAULT 'INITIAL_ALERT',
                collected_json  TEXT NOT NULL DEFAULT '{}',
                chat_history    TEXT NOT NULL DEFAULT '[]',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── Processed messages table (duplicate prevention) ────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id  TEXT PRIMARY KEY,
                received_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── Tickets table ──────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id         TEXT PRIMARY KEY,
                phone_number      TEXT NOT NULL,
                vehicle_location  TEXT,
                service_date      TEXT,
                driver_phone      TEXT,
                engineer_id       TEXT,
                engineer_name     TEXT,
                engineer_phone    TEXT,
                assignment_status TEXT,
                status            TEXT NOT NULL DEFAULT 'OPEN',
                created_at        TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # Add any missing columns to tickets (safe migration) ──────────────────
        cursor.execute("PRAGMA table_info(tickets)")
        existing_ticket_cols = {row[1] for row in cursor.fetchall()}
        for col in ["engineer_id", "engineer_name", "engineer_phone", "assignment_status"]:
            if col not in existing_ticket_cols:
                cursor.execute(f"ALTER TABLE tickets ADD COLUMN {col} TEXT")

        conn.commit()
    print("[DB] Database initialized successfully.")


# ==========================================
# SESSION OPERATIONS
# ==========================================

def get_session(phone_number: str) -> dict:
    """
    Retrieves the session for a phone number.
    Returns a fresh default session dict if no session exists.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_state, collected_json, chat_history FROM sessions WHERE phone_number = ?",
            (phone_number,)
        )
        row = cursor.fetchone()

    if row:
        return {
            "current_state": row["current_state"],
            "collected_json": json.loads(row["collected_json"]),
            "chat_history": json.loads(row["chat_history"])
        }

    # Fresh session defaults
    return {
        "current_state": "INITIAL_ALERT",
        "collected_json": {
            "intent": None,
            "vehicle_location": None,
            "service_date": None,
            "arrival_date": None,
            "driver_phone": None,
            "driver_name": None,
            "contact_person": None,
            "origin_city": None,
            "destination_city": None,
            "resume_date": None,
            "ticket_id": None,
            "service_booking_stage": None,
            "engineer_id": None,
            "engineer_name": None,
            "engineer_phone": None,
            "assignment_status": None,
            "conversation_completed": False,
            "battery_issue": False,
            "main_power_issue": False,
            "root_cause": "OTHER_ISSUE",
        },
        "chat_history": []
    }


def save_session(phone_number: str, current_state: str, collected_json: dict, chat_history: list):
    """
    Upserts (insert or update) a session record.
    Uses 'current_state' to match the actual column name in the DB schema.
    """
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (phone_number, current_state, collected_json, chat_history, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(phone_number) DO UPDATE SET
                current_state  = excluded.current_state,
                collected_json = excluded.collected_json,
                chat_history   = excluded.chat_history,
                updated_at     = excluded.updated_at
        """, (
            phone_number,
            current_state,
            json.dumps(collected_json),
            json.dumps(chat_history),
            now,
            now
        ))
        conn.commit()


def delete_session(phone_number: str):
    """Deletes a session (for testing or manual reset)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE phone_number = ?", (phone_number,))
        conn.commit()


# ==========================================
# DUPLICATE MESSAGE PREVENTION
# ==========================================

def is_duplicate_message(message_id: str) -> bool:
    """Returns True if this message_id has already been processed."""
    if not message_id:
        return False
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = ?",
            (message_id,)
        )
        return cursor.fetchone() is not None


def mark_message_processed(message_id: str):
    """Marks a message_id as processed to prevent duplicate handling."""
    if not message_id:
        return
    with get_connection() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO processed_messages (message_id, received_at) VALUES (?, ?)",
            (message_id, datetime.utcnow().isoformat())
        )
        conn.commit()


# ==========================================
# TICKET OPERATIONS
# ==========================================

def save_ticket(ticket_id: str, phone_number: str, data: dict):
    """Saves a newly created ticket to the database."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO tickets
            (ticket_id, phone_number, vehicle_location, service_date, driver_phone,
             engineer_id, engineer_name, engineer_phone, assignment_status,
             status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
        """, (
            ticket_id,
            phone_number,
            data.get("vehicle_location"),
            data.get("service_date"),
            data.get("driver_phone"),
            data.get("engineer_id"),
            data.get("engineer_name"),
            data.get("engineer_phone"),
            data.get("assignment_status"),
            now,
            now
        ))
        conn.commit()
    print(f"[DB] Ticket {ticket_id} saved for {phone_number}")


def get_ticket(ticket_id: str) -> dict | None:
    """Retrieves a ticket by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_ticket_status(ticket_id: str, status: str):
    """Updates the status of a ticket (OPEN, CLOSED, REOPENED)."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            "UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?",
            (status, now, ticket_id)
        )
        conn.commit()