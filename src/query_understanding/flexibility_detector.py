"""
Booking Flexibility Detector — Component 1d of Query Understanding
==================================================================
Classifies whether a user query signals flexible or rigid booking dates.

Output: "YES" (flexible) | "NO" (rigid)

Decision logic
--------------
Three evidence streams are combined into a signed score:

  1. Flexibility signals (+1 each)
       lexical patterns indicating openness to date changes
       e.g. "flexible", "around", "sometime in", "any weekend"

  2. Rigidity signals (-1 each)
       lexical patterns indicating fixed, committed dates
       e.g. "exactly", "must be", "only on", "cannot change"

  3. Date specificity (structural signal)
       Duckling returned a *day-grain* interval (precise dates)  → -1
       Duckling returned a week/month-grain interval             → +1
       No date entity found at all                               → +1
         (user hasn't committed to dates → leans flexible)

Final rule:  score >= 0  →  "YES" (flexible)
             score <  0  →  "NO"  (rigid)

A dedicated ML model can replace `detect()` once labeled data is available.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_FLEXIBLE_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bflexib\w*\b",                      # flexible, flexibility
    r"\baround\b",
    r"\bapprox(?:imately)?\b",
    r"\bsometime(?:s)?\b",
    r"\bany\s*time\b",
    r"\banytime\b",
    r"\bopen\s+to\b",
    r"\broughly\b",
    r"\bor\s+so\b",
    r"\bgive\s+or\s+take\b",
    r"\bnot\s+fixed\b",
    r"\bno\s+fixed\s+dates?\b",
    r"\bearly\s+\w+\b",                    # early June, early next month
    r"\bmid[- ]\w+\b",                     # mid-July, mid July
    r"\blate\s+\w+\b",                     # late August
    r"\bnext\s+(?:month|week|weekend|few\s+weeks)\b",
    r"\b(?:this|coming)\s+(?:summer|spring|autumn|fall|winter)\b",
    r"\b(?:summer|spring|autumn|fall|winter)\s+(?:trip|holiday|vacation|break|season)\b",
    r"\bany\s+(?:day|weekend|week)\b",
    r"\bwhen(?:ever)?\s+(?:it.s|is|its|you.re|there.s)\s+(?:cheaper|available|best|good)\b",
    r"\bcheapest\s+time\b",
    r"\bbest\s+time\s+to\b",
    r"\bdoesn.?t\s+matter\b",
    r"\bdon.?t\s+mind\s+when\b",
]]

_RIGID_PATTERNS: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in [
    r"\bexactly\b",
    r"\bspecifically\b",
    r"\bmust\s+be\b",
    r"\bonly\s+(?:on|from|between|during)\b",
    r"\bneed\s+to\s+(?:be|check|arrive|leave)\b",
    r"\bhave\s+to\s+(?:be|check|arrive|leave)\b",
    r"\bcannot\s+change\b",
    r"\bcan.?t\s+change\b",
    r"\bfixed\s+dates?\b",
    r"\bconfirmed\s+dates?\b",
    r"\balready\s+booked\b",
    r"\bno\s+flexibility\b",
]]

# Duckling grains coarser than "day" suggest vague / flexible date references
_COARSE_GRAINS = {"week", "month", "quarter", "year", "decade"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(text: str, duckling_entities: list[dict] | None = None) -> str:
    """Classify booking flexibility for *text*.

    Parameters
    ----------
    text :
        Normalised query string.
    duckling_entities :
        Raw JSON list returned by the Duckling REST API (``/parse``).
        Pass ``None`` (or omit) to skip structural date analysis.

    Returns
    -------
    "YES" if the user appears flexible, "NO" if rigid.
    """
    score = 0

    # ── Stream 1: flexibility keywords ────────────────────────────────────
    for pat in _FLEXIBLE_PATTERNS:
        if pat.search(text):
            score += 1

    # ── Stream 2: rigidity keywords ────────────────────────────────────────
    for pat in _RIGID_PATTERNS:
        if pat.search(text):
            score -= 1

    # ── Stream 3: Duckling date grain ──────────────────────────────────────
    if duckling_entities is None:
        # No structural signal — no dates committed to
        score += 1
    else:
        time_entities = [e for e in duckling_entities if e.get("dim") == "time"]
        if not time_entities:
            # Duckling found nothing → no dates mentioned
            score += 1
        else:
            for entity in time_entities:
                value = entity.get("value", {})
                # Check grain of from/to or the value itself
                grains = {
                    value.get("grain"),
                    value.get("from", {}).get("grain"),
                    value.get("to", {}).get("grain"),
                } - {None}
                if grains & _COARSE_GRAINS:
                    score += 1   # vague grain → flexible
                elif grains:
                    score -= 1   # day-level grain → rigid

    return "YES" if score >= 0 else "NO"
