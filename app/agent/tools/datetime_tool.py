from datetime import datetime, timedelta, timezone
import re
from langchain_core.tools import tool


@tool
def get_current_datetime(query: str = "now") -> str:
    """
    Get current date, time, day of the week, timezone information, or compute date offsets (e.g. '45 days from today').
    
    Args:
        query: Query parameter e.g. 'now', 'today', 'utc', '45 days from today', '3 weeks ago'.
        
    Returns:
        Formatted datetime string.
    """
    now = datetime.now()
    now_utc = datetime.now(timezone.utc)
    
    q_lower = query.lower().strip()
    
    # Check offset pattern: X days/weeks from today / ago
    offset_match = re.search(r'(\d+)\s+(day|week|month|year)s?\s+(from today|in the future|later|ago|before)', q_lower)
    if offset_match:
        amount = int(offset_match.group(1))
        unit = offset_match.group(2)
        direction = offset_match.group(3)

        days_offset = amount
        if unit == "week":
            days_offset = amount * 7
        elif unit == "month":
            days_offset = amount * 30
        elif unit == "year":
            days_offset = amount * 365

        if "ago" in direction or "before" in direction:
            target_date = now - timedelta(days=days_offset)
        else:
            target_date = now + timedelta(days=days_offset)

        return (
            f"Calculated Date: {target_date.strftime('%A, %B %d, %Y')} "
            f"(Offset: {amount} {unit}s {direction})"
        )

    return (
        f"Current Local Date & Time: {now.strftime('%A, %B %d, %Y - %H:%M:%S')}\n"
        f"Current UTC Date & Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Day of Week: {now.strftime('%A')}\n"
        f"ISO Format: {now.isoformat()}"
    )
