"""
FAISS Vector Index for audio embeddings.

Builds a flat L2 index over MelSpectrogram-derived embeddings
for fast nearest-neighbour retrieval.
"""

import logging
import os
from typing import List, Optional, Tuple

import faiss
import librosa
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    Build, save, load, and query a FAISS flat L2 index.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of each embedding vector.
    n_mels : int
        Number of mel bands for the embedding extractor.
    max_frames : int
        Max spectrogram frames to keep (pad/truncate).
    sr : int
        Expected sample rate for embedding extraction.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_mels: int = 128,
        max_frames: int = 128,
        sr: int = 16000,
    ):
        self.embedding_dim = embedding_dim
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.sr = sr
        self.index: Optional[faiss.IndexFlatL2] = None
        self._embeddings: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Embedding extractor
    # ------------------------------------------------------------------
    def extract_embedding(self, y: np.ndarray) -> np.ndarray:
        """
        Derive a fixed-size embedding from an audio signal using
        a mean-pooled MelSpectrogram.

        Returns a 1-D vector of shape ``(embedding_dim,)``.
        """
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels, n_fft=1024, hop_length=256,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Truncate or pad along time axis
        if mel_db.shape[1] > self.max_frames:
            mel_db = mel_db[:, :self.max_frames]
        elif mel_db.shape[1] < self.max_frames:
            pad_width = self.max_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")

        # Mean-pool across time to get (n_mels,) then truncate/pad to embedding_dim
        embedding = mel_db.mean(axis=1)
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]
        elif len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))

        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def build(self, signals: List[np.ndarray]) -> None:
        """
        Extract embeddings and build the FAISS index.

        Parameters
        ----------
        signals : list[np.ndarray]
            List of audio signals.
        """
        self._embeddings = [self.extract_embedding(s) for s in signals]
        matrix = np.vstack(self._embeddings).astype(np.float32)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(matrix)
        logger.info("Built FAISS index with %d vectors (dim=%d)", self.index.ntotal, self.embedding_dim)

    def add(self, signals: List[np.ndarray]) -> None:
        """Add more vectors to an existing index."""
        if self.index is None:
            self.build(signals)
            return
        new = [self.extract_embedding(s) for s in signals]
        self._embeddings.extend(new)
        matrix = np.vstack(new).astype(np.float32)
        self.index.add(matrix)
        logger.info("Added %d vectors → total %d", len(new), self.index.ntotal)

    def search(self, query_signal: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbours for a query audio signal.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (distances, indices) — both of shape ``(1, k)``.
        """
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty. Call build() first.")
        q = self.extract_embedding(query_signal).reshape(1, -1)
        distances, indices = self.index.search(q, k)
        return distances, indices

    def save(self, index_path: str, embeddings_path: str) -> None:
        """Save FAISS index and raw embeddings to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, index_path)
        if self._embeddings:
            np.save(embeddings_path, np.vstack(self._embeddings))
        logger.info("Saved index to %s, embeddings to %s", index_path, embeddings_path)

    def load(self, index_path: str, embeddings_path: Optional[str] = None) -> None:
        """Load a previously saved FAISS index (and optional embeddings)."""
        self.index = faiss.read_index(index_path)
        if embeddings_path and os.path.exists(embeddings_path):
            self._embeddings = list(np.load(embeddings_path))
        logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)

    @property
    def embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a 2-D array."""
        if not self._embeddings:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.vstack(self._embeddings)
