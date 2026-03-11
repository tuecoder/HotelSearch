"""
Component 1 — Query Preprocessing
==================================
Pipeline:
  raw text
    → 1. lowercase + unicode normalization (unidecode)
    → 2. spell correction  (pyspellchecker, domain-aware)
    → 3. tokenization      (NLTK word_tokenize)
    → 4. stop-word removal (NLTK English stopwords)
    → 5. lemmatization     (NLTK WordNetLemmatizer)
    → 6. synonym expansion (custom hotel/travel dictionary)

Output: PreprocessedQuery dataclass consumed by the Entity Extraction step.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from unidecode import unidecode

# ---------------------------------------------------------------------------
# Ensure required NLTK corpora are present (silent download on first run)
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


# ---------------------------------------------------------------------------
# Domain vocabulary — words that must NEVER be "spell-corrected" away
# ---------------------------------------------------------------------------
_DOMAIN_WHITELIST: frozenset[str] = frozenset(
    {
        # amenities
        "wifi", "spa", "gym", "sauna", "jacuzzi", "concierge", "valet",
        "beachfront", "rooftop", "minibar", "balcony",
        # property types
        "hotel", "hostel", "motel", "resort", "villa", "bungalow",
        "boutique", "apartment", "airbnb",
        # travel terms
        "checkin", "checkout", "b&b",
        # Dutch cities (Netherlands dataset)
        "amsterdam", "rotterdam", "utrecht", "delft", "haarlem", "leiden",
        "groningen", "eindhoven", "maastricht", "nijmegen", "arnhem",
        "hague", "zandvoort", "volendam", "ameland",
    }
)

# ---------------------------------------------------------------------------
# Synonym dictionary  →  term : list of expansion terms added to query context
# Used downstream by the Entity Extraction / BM25 query-expansion steps.
# ---------------------------------------------------------------------------
SYNONYM_MAP: dict[str, list[str]] = {
    # budget signals
    "cheap":      ["budget", "affordable", "economical"],
    "affordable": ["budget", "cheap", "economical"],
    "budget":     ["cheap", "affordable", "economical"],
    "inexpensive":["budget", "cheap", "affordable"],
    # luxury signals
    "luxury":     ["five-star", "premium", "upscale", "deluxe"],
    "luxurious":  ["luxury", "premium", "upscale"],
    "upscale":    ["luxury", "premium", "five-star"],
    "premium":    ["luxury", "upscale", "deluxe"],
    # amenities
    "pool":       ["swimming pool", "indoor pool"],
    "pools":      ["swimming pool", "pool"],
    "wifi":       ["wi-fi", "wireless internet", "internet"],
    "wi-fi":      ["wifi", "wireless internet"],
    "internet":   ["wifi", "wi-fi", "wireless"],
    "gym":        ["fitness center", "fitness room", "workout room"],
    "fitness":    ["gym", "fitness center"],
    "spa":        ["wellness", "sauna", "health club"],
    "wellness":   ["spa", "sauna"],
    "breakfast":  ["bed and breakfast", "b&b", "morning meal"],
    "parking":    ["garage", "car park", "self parking"],
    # traveller context
    "family":     ["kids", "children", "family-friendly"],
    "families":   ["family", "kids", "children"],
    "kids":       ["family", "children", "family-friendly"],
    "children":   ["family", "kids", "family-friendly"],
    "romantic":   ["couple", "honeymoon"],
    "honeymoon":  ["romantic", "couple"],
    "business":   ["corporate", "conference", "work trip"],
    "corporate":  ["business", "conference"],
    "solo":       ["single traveller", "alone"],
    # location qualifiers
    "central":    ["city center", "downtown", "centre"],
    "downtown":   ["city center", "central", "centre"],
    "beachfront": ["beach", "seaside", "oceanfront", "waterfront"],
    "beach":      ["beachfront", "seaside", "oceanfront"],
    "seaside":    ["beachfront", "beach", "oceanfront"],
    "near":       ["close to", "next to", "walking distance"],
}

# Stop words — use NLTK base set, then keep key travel terms
_STOP_WORDS: frozenset[str] = frozenset(stopwords.words("english")) - frozenset(
    {
        # preserve negations and travel-critical words
        "no", "not", "without", "under", "over",
        "near", "in", "at", "for",
    }
)

_lemmatizer = WordNetLemmatizer()
_spell = SpellChecker()
# Teach the spell-checker about domain words so it never "corrects" them
_spell.word_frequency.load_words(list(_DOMAIN_WHITELIST))


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class PreprocessedQuery:
    """Fully preprocessed query ready for Entity Extraction (Component 2)."""

    original: str
    """The raw unmodified user input."""

    normalized: str
    """Lowercased, unicode-normalized, spell-corrected string.
    This is passed directly into the regex / NER entity extractor."""

    tokens: list[str] = field(default_factory=list)
    """NLTK word tokens from the normalized text (punctuation stripped)."""

    lemmas: list[str] = field(default_factory=list)
    """Lemmatized tokens with stop words removed — used for BM25 / features."""

    expanded_terms: dict[str, list[str]] = field(default_factory=dict)
    """Synonym expansions found in the query: {matched_token: [synonyms]}.
    Downstream steps merge synonyms into the BM25 query for better recall."""

    def __str__(self) -> str:
        return (
            f"PreprocessedQuery(\n"
            f"  original  : {self.original!r}\n"
            f"  normalized: {self.normalized!r}\n"
            f"  tokens    : {self.tokens}\n"
            f"  lemmas    : {self.lemmas}\n"
            f"  expanded  : {self.expanded_terms}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step1_normalize(text: str) -> str:
    """Lowercase → unicode NFC normalization → unidecode transliteration.

    Examples
    --------
    'BARCELONA Hotels'  → 'barcelona hotels'
    'hôtel Île-de-France' → 'hotel ile-de-france'
    'Ámsterdam'        → 'amsterdam'
    """
    text = text.lower()
    text = unicodedata.normalize("NFC", text)   # compose accented chars first
    text = unidecode(text)                       # transliterate to ASCII
    # collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _step2_spell_correct(text: str) -> str:
    """Correct misspelled words while protecting domain vocabulary and proper nouns.

    Only single-word tokens that are entirely alphabetic and NOT in the
    domain whitelist are candidates for correction.  Numbers, proper nouns
    (detected by their presence in the original text as capitalized), and
    hyphenated compounds are left untouched.

    Examples
    --------
    'barselona hotel' → 'barcelona hotel'
    'swiming pool'    → 'swimming pool'
    'wifi'            → 'wifi'  (whitelisted)
    """
    words = text.split()
    corrected: list[str] = []
    for word in words:
        # Skip: non-alpha, domain whitelist, or short words (≤2 chars avoid false positives)
        if not word.isalpha() or word in _DOMAIN_WHITELIST or len(word) <= 2:
            corrected.append(word)
            continue
        suggestion = _spell.correction(word)
        corrected.append(suggestion if suggestion else word)
    return " ".join(corrected)


def _step3_tokenize(text: str) -> list[str]:
    """NLTK word tokenizer — returns alphabetic tokens only (no punctuation)."""
    return [tok for tok in word_tokenize(text) if (tok.isalpha() or tok.isnumeric() or tok.isalnum()) ]


def _step4_remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter out stop words; preserve travel-critical prepositions."""
    return [t for t in tokens if t not in _STOP_WORDS]


def _step5_lemmatize(tokens: list[str]) -> list[str]:
    """WordNet lemmatizer — noun form by default, sufficient for hotel queries.

    Examples
    --------
    ['swimming', 'pools'] → ['swimming', 'pool']
    ['families']          → ['family']
    """
    return [_lemmatizer.lemmatize(t) for t in tokens]


def _step6_expand_synonyms(tokens: list[str]) -> dict[str, list[str]]:
    """Return synonym expansions for any token found in SYNONYM_MAP.

    The returned dict maps each matched token to its synonym list.
    The caller (Entity Extraction) merges synonyms into the structured query.
    """
    return {t: SYNONYM_MAP[t] for t in tokens if t in SYNONYM_MAP}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(raw: str) -> PreprocessedQuery:
    """Run the full preprocessing pipeline on a raw user query string.

    Parameters
    ----------
    raw : str
        The unmodified text typed by the user, e.g.:
        "Family-friendly beachfront hotel in Amsterdam, pool, under €120"

    Returns
    -------
    PreprocessedQuery
        Dataclass with normalized text, tokens, lemmas, and synonym expansions.

    Examples
    --------
    >>> result = preprocess("Family-friendly hotel in Ámsterdam with POOL")
    >>> result.normalized
    'family-friendly hotel in amsterdam with pool'
    >>> 'pool' in result.expanded_terms
    True
    """
    if not raw or not raw.strip():
        return PreprocessedQuery(original=raw, normalized="")

    # Step 1 — normalize
    normalized = _step1_normalize(raw)

    # Step 2 — spell correction
    normalized = _step2_spell_correct(normalized)

    # Step 3 — tokenize
    tokens = _step3_tokenize(normalized)

    # Step 4 — remove stop words (on a copy; tokens keeps all words)
    content_tokens = _step4_remove_stopwords(tokens)

    # Step 5 — lemmatize the content tokens
    lemmas = _step5_lemmatize(content_tokens)

    # Step 6 — synonym expansion on lemmas
    expanded = _step6_expand_synonyms(lemmas)

    return PreprocessedQuery(
        original=raw,
        normalized=normalized,
        tokens=tokens,
        lemmas=lemmas,
        expanded_terms=expanded,
    )
