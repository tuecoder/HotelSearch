from unittest.mock import MagicMock, patch


def test_retrieve_by_vector_builds_knn_query():
    fake_hits = [{"_source": {"id": 1, "name": "Hotel A"}}]
    fake_resp = {"hits": {"hits": fake_hits}}

    mock_es = MagicMock()
    mock_es.search.return_value = fake_resp

    with patch("elasticsearch.Elasticsearch", return_value=mock_es):
        from src.retrieval.es_retriever import retrieve_by_vector
        result = retrieve_by_vector([0.1] * 24, size=10)

    assert result == [{"id": 1, "name": "Hotel A"}]

    call_kwargs = mock_es.search.call_args.kwargs
    assert call_kwargs["index"] == "hotels"
    knn = call_kwargs["knn"]
    assert knn["field"] == "tower_embedding"
    assert knn["query_vector"] == [0.1] * 24
    assert knn["k"] == 10
    assert knn["num_candidates"] == 500


def test_retrieve_by_vector_default_size_is_100():
    fake_resp = {"hits": {"hits": []}}
    mock_es = MagicMock()
    mock_es.search.return_value = fake_resp

    with patch("elasticsearch.Elasticsearch", return_value=mock_es):
        from src.retrieval.es_retriever import retrieve_by_vector
        retrieve_by_vector([0.0] * 24)

    knn = mock_es.search.call_args.kwargs["knn"]
    assert knn["k"] == 100


def test_retrieve_candidates_unchanged():
    """Ensure retrieve_candidates still exists and has its original signature."""
    from src.retrieval.es_retriever import retrieve_candidates
    import inspect
    sig = inspect.signature(retrieve_candidates)
    params = list(sig.parameters)
    assert "destination" in params
