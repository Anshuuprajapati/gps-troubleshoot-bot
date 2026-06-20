"""
date_utils.py — Deterministic date normalization for GPS bot.

Converts raw date strings extracted by LLM into ISO-formatted dates (YYYY-MM-DD).

Rules:
  - Bare day numbers ("25", "25 ko", "25th") → that day in current month.
    If that day has already passed, roll to the next month.
  - "kal"  → tomorrow
  - "parso" → day after tomorrow
  - "aaj"  → today
  - "next monday/tuesday/..." → next occurrence of that weekday
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

_WEEKDAY_MAP = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _next_valid_day(day: int, today: date) -> date:
    year, month = today.year, today.month
    max_day = monthrange(year, month)[1]
    clamped = min(day, max_day)
    candidate = date(year, month, clamped)
    if candidate < today:
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        max_day = monthrange(year, month)[1]
        clamped = min(day, max_day)
        candidate = date(year, month, clamped)
    return candidate


def _next_weekday(weekday_num: int, today: date) -> date:
    """Return next occurrence of weekday_num (0=Mon, 6=Sun), always in the future."""
    days_ahead = weekday_num - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return today + timedelta(days=days_ahead)


def normalize_date(raw: str | None, today: date | None = None) -> str | None:
    if not raw:
        return raw

    today = today or date.today()
    s = raw.strip().lower()

    # ── relative keywords ─────────────────────────────────────────────────────
    if s in ("aaj", "today"):
        return today.isoformat()
    if s in ("kal", "tomorrow", "kal tak", "tmrw", "tmr"):
        return (today + timedelta(days=1)).isoformat()
    if s in ("parso", "day after tomorrow"):
        return (today + timedelta(days=2)).isoformat()

    # ── next <weekday> ────────────────────────────────────────────────────────
    m = re.match(r"next\s+(\w+)", s)
    if m:
        day_word = m.group(1)
        wd = _WEEKDAY_MAP.get(day_word)
        if wd is not None:
            return _next_weekday(wd, today).isoformat()

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

    print(f"[DATE UTILS] Could not normalize '{raw}', returning as-is.")
    return raw