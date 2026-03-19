"""
Metadata Store.

TinyDB-backed JSON store for per-sample metadata with query,
filter, and aggregation operations.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tinydb import Query, TinyDB

logger = logging.getLogger(__name__)


class MetadataStore:
    """
    Lightweight metadata store backed by TinyDB (JSON file).

    Schema per record
    -----------------
    - ``file_path``: str — path to the audio file
    - ``speaker_id``: str
    - ``language``: str
    - ``duration``: float (seconds)
    - ``sample_rate``: int
    - ``noise_level``: float | None (dB SNR if augmented)
    - ``augmentation``: dict — applied augmentation info
    - ``segment_index``: int — index within the original file
    - ``created_at``: str — ISO-8601 timestamp

    Parameters
    ----------
    db_path : str
        Path to the TinyDB JSON file.
    """

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.db_path = os.path.abspath(db_path)
        self.db = TinyDB(self.db_path, indent=2)
        logger.info("MetadataStore opened at %s  (%d records)", self.db_path, len(self.db))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def insert(self, record: Dict[str, Any]) -> int:
        """Insert a single metadata record. Returns the document ID."""
        record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        doc_id = self.db.insert(record)
        logger.debug("Inserted doc %d", doc_id)
        return doc_id

    def insert_many(self, records: List[Dict[str, Any]]) -> List[int]:
        """Bulk insert. Returns list of document IDs."""
        now = datetime.now(timezone.utc).isoformat()
        for r in records:
            r.setdefault("created_at", now)
        ids = self.db.insert_multiple(records)
        logger.info("Bulk inserted %d records", len(ids))
        return ids

    # ------------------------------------------------------------------
    # Read / query
    # ------------------------------------------------------------------
    def all(self) -> List[Dict[str, Any]]:
        """Return every record."""
        return self.db.all()

    def count(self) -> int:
        """Number of records in the store."""
        return len(self.db)

    def query(self, field: str, value: Any) -> List[Dict[str, Any]]:
        """Exact-match query on a single field."""
        q = Query()
        return self.db.search(q[field] == value)

    def query_range(self, field: str, min_val: float, max_val: float) -> List[Dict[str, Any]]:
        """Range query on a numeric field (inclusive)."""
        q = Query()
        return self.db.search((q[field] >= min_val) & (q[field] <= max_val))

    def filter_by(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Multi-field exact-match filter.

        Example::

            store.filter_by(language="en", speaker_id="spk_001")
        """
        q = Query()
        conditions = [(q[k] == v) for k, v in kwargs.items()]
        combined = conditions[0]
        for c in conditions[1:]:
            combined = combined & c
        return self.db.search(combined)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def unique_values(self, field: str) -> List[Any]:
        """Return unique values for a given field."""
        return list({r.get(field) for r in self.db.all() if field in r})

    def total_duration(self, **filter_kwargs) -> float:
        """Sum of ``duration`` field, optionally filtered."""
        records = self.filter_by(**filter_kwargs) if filter_kwargs else self.all()
        return sum(r.get("duration", 0.0) for r in records)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Drop all records."""
        self.db.truncate()
        logger.warning("Metadata store cleared.")

    def close(self) -> None:
        """Close the underlying database."""
        self.db.close()
