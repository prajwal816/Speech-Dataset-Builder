"""Tests for the Metadata Store."""

import os
import tempfile
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metadata.store import MetadataStore


class TestMetadataStore:
    """Unit tests for MetadataStore."""

    def _get_store(self, tmp_path):
        db_path = os.path.join(tmp_path, "test_meta.json")
        return MetadataStore(db_path)

    def test_insert_and_count(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"speaker_id": "spk_001", "language": "en", "duration": 5.0})
        store.insert({"speaker_id": "spk_002", "language": "fr", "duration": 3.2})
        assert store.count() == 2
        store.close()

    def test_query_exact_match(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"speaker_id": "spk_001", "language": "en", "duration": 5.0})
        store.insert({"speaker_id": "spk_002", "language": "en", "duration": 3.0})
        results = store.query("speaker_id", "spk_001")
        assert len(results) == 1
        assert results[0]["speaker_id"] == "spk_001"
        store.close()

    def test_query_range(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"speaker_id": "a", "duration": 1.0})
        store.insert({"speaker_id": "b", "duration": 5.0})
        store.insert({"speaker_id": "c", "duration": 10.0})
        results = store.query_range("duration", 2.0, 8.0)
        assert len(results) == 1
        assert results[0]["duration"] == 5.0
        store.close()

    def test_filter_by_multi_field(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"speaker_id": "spk_001", "language": "en", "duration": 5.0})
        store.insert({"speaker_id": "spk_001", "language": "fr", "duration": 3.0})
        store.insert({"speaker_id": "spk_002", "language": "en", "duration": 4.0})
        results = store.filter_by(speaker_id="spk_001", language="en")
        assert len(results) == 1
        store.close()

    def test_unique_values(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"language": "en"})
        store.insert({"language": "fr"})
        store.insert({"language": "en"})
        assert set(store.unique_values("language")) == {"en", "fr"}
        store.close()

    def test_total_duration(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"duration": 5.0, "language": "en"})
        store.insert({"duration": 3.0, "language": "en"})
        store.insert({"duration": 7.0, "language": "fr"})
        assert store.total_duration(language="en") == 8.0
        store.close()

    def test_clear(self, tmp_path):
        store = self._get_store(tmp_path)
        store.insert({"x": 1})
        store.clear()
        assert store.count() == 0
        store.close()

    def test_insert_many(self, tmp_path):
        store = self._get_store(tmp_path)
        ids = store.insert_many([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ids) == 3
        assert store.count() == 3
        store.close()
