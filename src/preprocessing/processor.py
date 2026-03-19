"""
Audio Preprocessing Module.

Resampling, silence trimming, and normalization for raw audio signals.
"""

import logging
from typing import Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Stateless audio processor driven by config parameters.

    Parameters
    ----------
    target_sr : int
        Target sample rate.
    normalize : bool
        Apply normalization after processing.
    normalization_method : str
        ``"peak"`` or ``"rms"``.
    trim_silence : bool
        Trim leading/trailing silence.
    silence_threshold_db : float
        Threshold in dBFS for silence detection.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        normalize: bool = True,
        normalization_method: str = "peak",
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
    ):
        self.target_sr = target_sr
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db

    # ------------------------------------------------------------------
    # Core transforms
    # ------------------------------------------------------------------
    def resample(self, y: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample signal to target sample rate."""
        if orig_sr == self.target_sr:
            return y
        resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=self.target_sr)
        logger.debug("Resampled %d→%d Hz  samples %d→%d", orig_sr, self.target_sr, len(y), len(resampled))
        return resampled

    def trim(self, y: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence using an energy threshold."""
        trimmed, _ = librosa.effects.trim(y, top_db=abs(self.silence_threshold_db))
        removed = len(y) - len(trimmed)
        if removed > 0:
            logger.debug("Trimmed %d silent samples (%.2f%%)", removed, 100.0 * removed / len(y))
        return trimmed

    def normalize_signal(self, y: np.ndarray) -> np.ndarray:
        """
        Normalize signal amplitude.

        - **peak**: scale so max |sample| = 1.0
        - **rms**: scale to target RMS of 0.1
        """
        if len(y) == 0:
            return y

        if self.normalization_method == "peak":
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak
        elif self.normalization_method == "rms":
            target_rms = 0.1
            current_rms = np.sqrt(np.mean(y ** 2))
            if current_rms > 0:
                y = y * (target_rms / current_rms)
                # Clip to avoid overflow
                y = np.clip(y, -1.0, 1.0)
        else:
            logger.warning("Unknown normalization method '%s', skipping.", self.normalization_method)
        return y

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def process(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """
        Apply the full preprocessing chain.

        Parameters
        ----------
        y : np.ndarray
            Input audio signal.
        sr : int
            Original sample rate.

        Returns
        -------
        tuple[np.ndarray, int]
            (processed_signal, target_sample_rate)
        """
        y = self.resample(y, sr)
        if self.trim_silence:
            y = self.trim(y)
        if self.normalize:
            y = self.normalize_signal(y)
        return y, self.target_sr
