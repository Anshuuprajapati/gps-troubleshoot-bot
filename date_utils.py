"""
date_utils.py — Deterministic date normalization for GPS bot.

Converts raw date strings extracted by Groq into ISO-formatted dates (YYYY-MM-DD).

Rules:
  - Bare day numbers ("25", "25 ko", "25th") → that day in current month.
    If that day has already passed, roll to the next month.
  - "kal"  → tomorrow
  - "parso" → day after tomorrow
  - "aaj"  → today
  - Already well-formed dates (YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY, "25 June") → parse directly.
  - If nothing matches, return the raw string unchanged so we never silently drop data.
"""

import re
from datetime import date, timedelta
from calendar import monthrange


# ── month name → number ───────────────────────────────────────────────────────
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _next_valid_day(day: int, today: date) -> date:
    """
    Return a date object for 'day' in the current month.
    If that day has already passed (strictly before today), roll to next month.
    Clamps to the last valid day of the month (e.g. day=31 in April → April 30).
    """
    year, month = today.year, today.month
    max_day = monthrange(year, month)[1]
    clamped = min(day, max_day)
    candidate = date(year, month, clamped)
    if candidate < today:
        # Roll to next month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        max_day = monthrange(year, month)[1]
        clamped = min(day, max_day)
        candidate = date(year, month, clamped)
    return candidate


def normalize_date(raw: str | None, today: date | None = None) -> str | None:
    """
    Normalize a raw date string to YYYY-MM-DD.
    Returns None if raw is None/empty.
    Returns raw unchanged if it cannot be parsed (so data is never lost).
    """
    if not raw:
        return raw

    today = today or date.today()
    s = raw.strip().lower()

    # ── relative keywords ─────────────────────────────────────────────────────
    if s in ("aaj", "today"):
        return today.isoformat()
    if s in ("kal", "tomorrow", "kal tak"):
        return (today + timedelta(days=1)).isoformat()
    if s in ("parso", "day after tomorrow"):
        return (today + timedelta(days=2)).isoformat()

    # ── already ISO YYYY-MM-DD ────────────────────────────────────────────────
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s

    # ── DD-MM-YYYY or DD/MM/YYYY ──────────────────────────────────────────────
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        d_, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d_).isoformat()
        except ValueError:
            pass

    # ── "25 June", "25 june 2026", "25th June" ───────────────────────────────
    m = re.match(r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s+(\d{4}))?", s)
    if m:
        day_num = int(m.group(1))
        month_word = m.group(2)
        year_str = m.group(3)
        month_num = _MONTH_MAP.get(month_word)
        if month_num:
            year = int(year_str) if year_str else today.year
            try:
                candidate = date(year, month_num, day_num)
                # If no year was specified and date has passed, roll to next year
                if not year_str and candidate < today:
                    candidate = date(year + 1, month_num, day_num)
                return candidate.isoformat()
            except ValueError:
                pass

    # ── bare day number: "25", "25 ko", "25th", "25th ko" ────────────────────
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?(?:\s+ko)?$", s)
    if m:
        day_num = int(m.group(1))
        if 1 <= day_num <= 31:
            return _next_valid_day(day_num, today).isoformat()

    # ── nothing matched — return original so we don't silently lose it ─────────
    print(f"[DATE UTILS] Could not normalize '{raw}', returning as-is.")
    return raw