from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown as md
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.query_understanding.preprocessing import PreprocessedQuery, preprocess
from src.query_understanding.budget_classifier import BudgetClassifier, BudgetTier
from src.query_understanding.flexibility_detector import detect as detect_flexibility

app = FastAPI(title="Hotel Search UI", version="0.1.0")

app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
templates = Jinja2Templates(directory="src/app/templates")

# Budget classifier — loaded once at startup; None if model file missing
try:
    _budget_clf = BudgetClassifier()
except FileNotFoundError:
    _budget_clf = None


class SearchQuery(BaseModel):
    raw: str
    destination: str | None = None
    checkin: datetime | None = None
    checkout: datetime | None = None
    guests: int | None = None
    max_price: int | None = None
    amenities: list[str] | None = None
    budget: str | None = None      # LUXURY | MID_RANGE | BUDGET | UNSPECIFIED
    flexible: str | None = None   # YES | NO


_RATING_TO_STARS: dict[str, int] = {
    "fivestar": 5,
    "fourstar": 4,
    "threestar": 3,
    "twostar": 2,
    "onestar": 1,
    "all": 3,
}

_RATING_TO_SCORE: dict[str, float] = {
    "fivestar": 9.5,
    "fourstar": 8.5,
    "threestar": 7.5,
    "twostar": 6.5,
    "onestar": 6.0,
    "all": 7.0,
}

_AMENITY_KEYWORDS: dict[str, list[str]] = {
    "pool": ["pool", "swimming"],
    "gym": ["gym", "fitness", "health club"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "spa": ["spa", "wellness"],
    "breakfast": ["breakfast"],
    "parking": ["parking"],
    "family rooms": ["family room", "family suite"],
    "restaurant": ["restaurant"],
    "bar": ["bar", "lounge"],
    "bike rental": ["bicycle", "bike rental"],
}


def _parse_amenities(facilities: str) -> list[str]:
    if not facilities or facilities.lower() in ("nan", "n/a", ""):
        return []
    fac_lower = facilities.lower()
    return [
        amenity
        for amenity, keywords in _AMENITY_KEYWORDS.items()
        if any(kw in fac_lower for kw in keywords)
    ]


def _clean_description(raw: str) -> str:
    """Strip HeadLine/Location labels and fix Windows smart-quote characters."""
    text = raw.strip()
    # Remove leading "HeadLine : <text> Location : " boilerplate
    text = re.sub(r"^HeadLine\s*:\s*.+?Location\s*:\s*", "", text, flags=re.IGNORECASE)
    # Replace Windows-1252 smart quotes and dashes with plain ASCII equivalents
    replacements = {
        "\x91": "'", "\x92": "'",
        "\x93": '"', "\x94": '"',
        "\x96": "-", "\x97": "-",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Collapse excessive whitespace / newlines into single spaces
    text = " ".join(text.split())
    return text


def _load_netherlands_properties() -> list[dict[str, Any]]:
    excel_path = Path("data/processed/netherlands_hotels.xlsx")
    df = pd.read_excel(excel_path)
    props: list[dict[str, Any]] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        rating_key = str(row.HotelRating).strip().lower()
        props.append(
            {
                "id": idx,
                "name": str(row.HotelName).strip(),
                "city": str(row.cityName).strip(),
                "country": str(row.countyName).strip(),
                "stars": _RATING_TO_STARS.get(rating_key, 3),
                "rating": _RATING_TO_SCORE.get(rating_key, 7.0),
                "address": str(row.Address).strip(),
                "attractions": str(row.Attractions).strip(),
                "description": md.markdown(_clean_description(str(row.Description))),
                "fax": str(row.FaxNumber).strip(),
                "phone": str(row.PhoneNumber).strip(),
                "amenities": _parse_amenities(str(row.HotelFacilities)),
            }
        )
    return props


PROPERTIES: list[dict[str, Any]] = _load_netherlands_properties()

# Build destination regex from all unique city names in the dataset
_city_patterns = sorted(
    {p["city"].lower() for p in PROPERTIES}, key=len, reverse=True
)
DESTINATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in _city_patterns) + r")\b",
    re.IGNORECASE,
)
DUCKLING_URL = "http://localhost:8000/parse"
DUCKLING_TIMEOUT = 2  # seconds — don't stall the UI if the server is down
GUESTS_RE = re.compile(r"(\d+)\s*(adults|guests|people)")
PRICE_RE = re.compile(r"<\s*\$?(\d+)|under\s*\$?(\d+)")
# Canonical amenity name → all surface forms the user might type.
# Match order within each group doesn't matter; longer phrases should appear
# before shorter ones so the combined regex greedily picks them up first.
AMENITY_SYNONYMS: dict[str, list[str]] = {
    "pool": [
        "swimming pool", "indoor pool", "outdoor pool", "rooftop pool",
        "infinity pool", "baby pool", "heated pool", "pool",
    ],
    "wifi": [
        "wi-fi", "wi fi", "wireless internet", "wireless", "internet", "wifi",
    ],
    "gym": [
        "fitness center", "fitness centre", "fitness room", "fitness club",
        "workout room", "health club", "gym",
    ],
    "spa": [
        "wellness center", "wellness centre", "thermal bath", "hot tub",
        "jacuzzi", "sauna", "steam room", "spa",
    ],
    "breakfast": [
        "breakfast included", "full board", "half board", "all inclusive",
        "bed and breakfast", "buffet breakfast", "breakfast", "free breakfast"
    ],
    "parking": [
        "free parking", "private parking", "car park", "garage", "parking",
    ],
    "family rooms": [
        "family suite", "family apartment", "family friendly",
        "kids friendly", "child friendly", "family rooms",
    ],
    "restaurant": [
        "on-site restaurant", "dining", "restaurant",
    ],
    "bar": [
        "cocktail bar", "rooftop bar", "lounge bar", "bar",
    ],
    "bike rental": [
        "bicycle rental", "bike hire", "bicycle hire", "cycling", "bike rental",
    ],
    "air conditioning": [
        "air con", "ac", "a/c", "climate control", "air conditioning",
    ],
    "pet friendly": [
        "pets allowed", "dogs allowed", "dog friendly", "pet friendly",
    ],
    "airport shuttle": [
        "airport transfer", "airport pickup", "shuttle service", "airport shuttle",
    ],
}

# Build a single compiled regex: longest phrases first (avoids partial matches).
# Each alternative is wrapped in a named group so we can map match → canonical.
_amenity_alternatives: list[tuple[str, str]] = []
for _canonical, _synonyms in AMENITY_SYNONYMS.items():
    for _syn in sorted(_synonyms, key=len, reverse=True):
        _amenity_alternatives.append((_canonical, _syn))

_amenity_alternatives.sort(key=lambda t: len(t[1]), reverse=True)

AMENITY_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(s) for _, s in _amenity_alternatives) + r")\b"
)

# Reverse lookup: normalised surface form → canonical name
_AMENITY_SURFACE_TO_CANONICAL: dict[str, str] = {
    s.lower(): canon
    for canon, synonyms in AMENITY_SYNONYMS.items()
    for s in synonyms
}


def _call_duckling(text: str) -> list[dict]:
    """POST to the Duckling REST server and return raw entity list.

    Returns an empty list if the server is unreachable.
    """
    try:
        resp = requests.post(
            DUCKLING_URL,
            data={"text": text, "locale": "en_US", "dims": '["time"]'},
            timeout=DUCKLING_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


def _parse_dates_duckling(
    entities: list[dict],
) -> tuple[datetime | None, datetime | None]:
    """Extract check-in / check-out datetimes from pre-fetched Duckling entities.

    Returns (None, None) if no day-level interval is found.
    """
    for entity in entities:
        if entity.get("dim") != "time":
            continue
        value = entity.get("value", {})
        if value.get("type") != "interval":
            continue
        from_str = value.get("from", {}).get("value")
        to_str = value.get("to", {}).get("value")
        if from_str and to_str:
            # Duckling returns ISO-8601; slice to YYYY-MM-DD to ignore timezone
            checkin = datetime.fromisoformat(from_str[:10])
            checkout = datetime.fromisoformat(to_str[:10])
            return checkin, checkout

    return None, None


def parse_query(raw: str) -> SearchQuery:
    destination_match = DESTINATION_RE.search(raw)
    destination = destination_match.group(1).title() if destination_match else None

    duckling_entities = _call_duckling(raw)
    checkin, checkout = _parse_dates_duckling(duckling_entities)

    guests = None
    guests_match = GUESTS_RE.search(raw)
    if guests_match:
        guests = int(guests_match.group(1))

    max_price = None
    price_match = PRICE_RE.search(raw)
    if price_match:
        max_price = int(price_match.group(1) or price_match.group(2))

    seen: set[str] = set()
    amenities = []
    for m in AMENITY_RE.finditer(raw):
        canonical = _AMENITY_SURFACE_TO_CANONICAL[m.group(1).lower()]
        if canonical not in seen:
            seen.add(canonical)
            amenities.append(canonical)
    if not amenities:
        amenities = None

    return SearchQuery(
        raw=raw,
        destination=destination,
        checkin=checkin,
        checkout=checkout,
        guests=guests,
        max_price=max_price,
        amenities=amenities,
        flexible=detect_flexibility(raw, duckling_entities),
    )



def score_property(query: SearchQuery, prop: dict[str, Any]) -> float:
    score = 0.0
    if query.destination and prop["city"].lower() == query.destination.lower():
        score += 3.0
    if query.amenities:
        overlap = len(set(query.amenities) & set(prop["amenities"]))
        score += overlap * 0.7
    score += (prop["rating"] - 7.0) * 0.4
    return score


def retrieve_candidates(query: SearchQuery) -> list[dict[str, Any]]:
    candidates = []
    for prop in PROPERTIES:
        if query.destination and prop["city"].lower() != query.destination.lower():
            continue
        candidates.append(prop)
    if not candidates:
        candidates = PROPERTIES
    return candidates


def rank_candidates(
    query: SearchQuery, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scored = [
        {**prop, "score": round(score_property(query, prop), 3)}
        for prop in candidates
    ]
    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": None,
            "parsed": None,
            "results": [],
        },
    )


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "") -> HTMLResponse:
    raw_query = q.strip()
    preprocessed: PreprocessedQuery | None = None
    parsed = None
    results: list[dict[str, Any]] = []

    if raw_query:
        # Stage 0 — Component 1: Query Preprocessing
        preprocessed = preprocess(raw_query)

        # Stage 0 — Component 2+: Entity Extraction on the normalized text
        parsed = parse_query(preprocessed.normalized)

        # Stage 0 — Budget tier classification
        if _budget_clf is not None:
            parsed.budget = _budget_clf.predict_label(preprocessed.normalized)

        # Enrich parsed amenities with synonym expansions from preprocessing
        if preprocessed.expanded_terms:
            extra = {
                syn
                for syns in preprocessed.expanded_terms.values()
                for syn in syns
                if syn in {k for keys in _AMENITY_KEYWORDS for k in keys}
            }
            if extra:
                current = set(parsed.amenities or [])
                parsed.amenities = list(current | extra)

        candidates = retrieve_candidates(parsed)
        results = rank_candidates(parsed, candidates)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "query": raw_query,
            "preprocessed": preprocessed,
            "parsed": parsed,
            "results": results,
        },
    )
