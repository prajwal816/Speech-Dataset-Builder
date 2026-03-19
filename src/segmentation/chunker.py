"""
Audio Segmentation Module.

Chunk long audio files into training-ready segments with optional
speaker-aware segmentation (simulated via energy envelope).
"""

import logging
from typing import List, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioChunker:
    """
    Segment audio into fixed-length or energy-based chunks.

    Parameters
    ----------
    chunk_duration : float
        Target chunk length in seconds.
    overlap : float
        Overlap between consecutive chunks in seconds.
    min_duration : float
        Minimum acceptable chunk length in seconds.
    method : str
        ``"fixed"`` for uniform slicing, ``"energy"`` for energy-based splitting.
    speaker_aware : bool
        If True, attempt to detect speaker turns via energy dips.
    energy_threshold_db : float
        Threshold for energy-based silence/speaker-turn detection.
    sr : int
        Sample rate of the audio.
    """

    def __init__(
        self,
        chunk_duration: float = 10.0,
        overlap: float = 1.0,
        min_duration: float = 2.0,
        method: str = "fixed",
        speaker_aware: bool = False,
        energy_threshold_db: float = -35.0,
        sr: int = 16000,
    ):
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.min_duration = min_duration
        self.method = method
        self.speaker_aware = speaker_aware
        self.energy_threshold_db = energy_threshold_db
        self.sr = sr

    # ------------------------------------------------------------------
    # Fixed-length chunking
    # ------------------------------------------------------------------
    def _fixed_chunks(self, y: np.ndarray) -> List[np.ndarray]:
        """Split audio into overlapping fixed-length chunks."""
        chunk_samples = int(self.chunk_duration * self.sr)
        step = int((self.chunk_duration - self.overlap) * self.sr)
        min_samples = int(self.min_duration * self.sr)

        chunks: List[np.ndarray] = []
        start = 0
        while start < len(y):
            end = start + chunk_samples
            segment = y[start:end]
            if len(segment) >= min_samples:
                chunks.append(segment)
            start += step
        return chunks

    # ------------------------------------------------------------------
    # Energy-based chunking
    # ------------------------------------------------------------------
    def _energy_chunks(self, y: np.ndarray) -> List[np.ndarray]:
        """
        Split audio at low-energy boundaries (silence gaps).

        Uses short-time RMS energy to find natural break points.
        """
        frame_length = int(0.025 * self.sr)   # 25 ms frames
        hop_length = int(0.010 * self.sr)      # 10 ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Find frames below threshold → potential split points
        silence_mask = rms_db < self.energy_threshold_db
        min_samples = int(self.min_duration * self.sr)

        chunks: List[np.ndarray] = []
        seg_start = 0

        for i, is_silent in enumerate(silence_mask):
            if is_silent:
                sample_idx = i * hop_length
                segment = y[seg_start:sample_idx]
                if len(segment) >= min_samples:
                    chunks.append(segment)
                    seg_start = sample_idx

        # Remaining tail
        tail = y[seg_start:]
        if len(tail) >= min_samples:
            chunks.append(tail)

        return chunks

    # ------------------------------------------------------------------
    # Speaker-aware segmentation (simulate)
    # ------------------------------------------------------------------
    def _speaker_aware_chunks(self, y: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        Simulate speaker-aware segmentation using energy envelope.

        Detects energy dips as likely speaker boundaries and assigns
        alternating speaker IDs.

        Returns
        -------
        list[tuple[np.ndarray, int]]
            (segment, simulated_speaker_id)
        """
        raw_chunks = self._energy_chunks(y)
        labeled: List[Tuple[np.ndarray, int]] = []
        for idx, chunk in enumerate(raw_chunks):
            speaker_id = idx % 2   # alternate speakers
            labeled.append((chunk, speaker_id))
        return labeled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def segment(self, y: np.ndarray) -> List[dict]:
        """
        Segment audio signal into chunks.

        Returns
        -------
        list[dict]
            Each dict has keys ``"audio"`` (np.ndarray), ``"speaker_id"`` (int),
            and ``"duration"`` (float, seconds).
        """
        if self.speaker_aware:
            labeled = self._speaker_aware_chunks(y)
            results = []
            for audio, spk in labeled:
                results.append({
                    "audio": audio,
                    "speaker_id": spk,
                    "duration": len(audio) / self.sr,
                })
            return results

        if self.method == "energy":
            chunks = self._energy_chunks(y)
        else:
            chunks = self._fixed_chunks(y)

        results = []
        for chunk in chunks:
            results.append({
                "audio": chunk,
                "speaker_id": -1,
                "duration": len(chunk) / self.sr,
            })
        logger.info("Segmented audio into %d chunks (method=%s)", len(results), self.method)
        return results
