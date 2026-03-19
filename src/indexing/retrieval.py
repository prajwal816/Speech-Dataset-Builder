"""
Subset Retrieval System.

Fast subset retrieval by metadata filters, vector similarity,
train/test splitting, and custom subset sampling.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SubsetRetriever:
    """
    Retrieve and split dataset subsets using metadata and vector similarity.

    Parameters
    ----------
    metadata_records : list[dict]
        Full metadata list from ``MetadataStore.all()``.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, metadata_records: List[Dict[str, Any]], seed: int = 42):
        self.records = metadata_records
        self.seed = seed

    # ------------------------------------------------------------------
    # Metadata-based filtering
    # ------------------------------------------------------------------
    def filter_by_language(self, language: str) -> List[Dict[str, Any]]:
        """Return records matching a specific language."""
        return [r for r in self.records if r.get("language") == language]

    def filter_by_speaker(self, speaker_id: str) -> List[Dict[str, Any]]:
        """Return records for a specific speaker."""
        return [r for r in self.records if r.get("speaker_id") == speaker_id]

    def filter_by_duration(self, min_sec: float, max_sec: float) -> List[Dict[str, Any]]:
        """Return records within a duration range (inclusive)."""
        return [
            r for r in self.records
            if min_sec <= r.get("duration", 0.0) <= max_sec
        ]

    def filter_by(self, **kwargs) -> List[Dict[str, Any]]:
        """Generic multi-field exact-match filter."""
        result = self.records
        for key, val in kwargs.items():
            result = [r for r in result if r.get(key) == val]
        return result

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    def train_test_split(
        self,
        records: Optional[List[Dict[str, Any]]] = None,
        train_ratio: float = 0.8,
        stratify_by: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split records into train and test sets.

        Parameters
        ----------
        records : list[dict] | None
            Records to split. Defaults to all records.
        train_ratio : float
            Fraction for training set.
        stratify_by : str | None
            Metadata field name for stratified splitting.

        Returns
        -------
        tuple[list[dict], list[dict]]
            (train_records, test_records)
        """
        if records is None:
            records = self.records
        if not records:
            return [], []

        stratify = None
        if stratify_by:
            labels = [r.get(stratify_by, "unknown") for r in records]
            # Only stratify if there are at least 2 unique classes and each class has ≥2 samples
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) >= 2 and all(c >= 2 for c in counts):
                stratify = labels
            else:
                logger.warning(
                    "Cannot stratify by '%s' (unique=%d, min_count=%d). Falling back to random split.",
                    stratify_by, len(unique), min(counts) if len(counts) else 0,
                )

        train, test = train_test_split(
            records,
            train_size=train_ratio,
            random_state=self.seed,
            stratify=stratify,
        )
        logger.info("Split: train=%d  test=%d", len(train), len(test))
        return train, test

    # ------------------------------------------------------------------
    # Custom subset sampling
    # ------------------------------------------------------------------
    def sample_subset(
        self,
        n: int,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Random sample of *n* records (without replacement)."""
        if records is None:
            records = self.records
        n = min(n, len(records))
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(records), size=n, replace=False)
        return [records[i] for i in indices]

    # ------------------------------------------------------------------
    # Vector-similarity subset
    # ------------------------------------------------------------------
    @staticmethod
    def subset_by_similarity(
        distances: np.ndarray,
        indices: np.ndarray,
        metadata_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Given FAISS search results, return the matching metadata records.

        Parameters
        ----------
        distances : np.ndarray
            Distance matrix from ``FAISSIndex.search()``.
        indices : np.ndarray
            Index matrix from ``FAISSIndex.search()``.
        metadata_records : list[dict]
            Full metadata list (order must match FAISS index order).

        Returns
        -------
        list[dict]
            Metadata records for the neighbours, with ``_distance`` appended.
        """
        results = []
        for dist, idx in zip(distances.flatten(), indices.flatten()):
            if 0 <= idx < len(metadata_records):
                rec = dict(metadata_records[idx])
                rec["_distance"] = float(dist)
                results.append(rec)
        return results
