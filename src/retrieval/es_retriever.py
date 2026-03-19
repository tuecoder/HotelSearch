"""
Elasticsearch Hotel Retriever
==============================
Retrieves hotel candidates from the `hotels` index using an exact
keyword search on the `city` field (ES `term` query).

Because `city` is mapped as `keyword`, the `term` query matches the
stored value byte-for-byte — no tokenisation, no stemming.  City names
are stored in title-case (e.g. "Amsterdam"), so callers must pass the
destination in the same casing (which `parse_query` already does via
`.title()`).

Usage
-----
    from src.retrieval.es_retriever import retrieve_candidates
    hotels = retrieve_candidates("Amsterdam")         # exact-match
    hotels = retrieve_candidates(None)                # match_all, up to 200
    hotels = retrieve_candidates("Rotterdam", size=50)
"""
from __future__ import annotations

from typing import Any

ES_HOST       = "http://localhost:9200"
INDEX         = "hotels"
DEFAULT_SIZE  = 200


def retrieve_candidates(
    destination: str | None,
    *,
    size: int = DEFAULT_SIZE,
    es_host: str = ES_HOST,
    index: str = INDEX,
) -> list[dict[str, Any]]:
    """Return hotels matching *destination* via Elasticsearch.

    Parameters
    ----------
    destination:
        City name to search for (exact, case-sensitive keyword match).
        Pass ``None`` to return up to *size* hotels across all cities.
    size:
        Maximum number of results to return (default 200).
    es_host:
        Elasticsearch base URL (override for testing).
    index:
        Index name (override for testing).

    Returns
    -------
    list[dict]
        Each dict has the same keys as the PROPERTIES entries in the app:
        id, name, city, country, stars, rating, address, attractions,
        description, amenities, phone, fax.
    """
    from elasticsearch import Elasticsearch

    es = Elasticsearch(es_host)

    query: dict[str, Any] = (
        {"term": {"city": destination}}
        if destination
        else {"match_all": {}}
    )

    resp = es.search(index=index, query=query, size=size)
    return [hit["_source"] for hit in resp["hits"]["hits"]]
