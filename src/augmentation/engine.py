"""
Audio Augmentation Engine.

Applies noise injection, speed perturbation, and pitch shift
with configurable ranges and reproducible randomness.
"""

import logging
from typing import List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)


class AugmentationEngine:
    """
    Produce augmented copies of audio signals.

    Parameters
    ----------
    snr_range_db : tuple[float, float]
        (min_snr, max_snr) in dB for additive Gaussian noise.
    speed_range : tuple[float, float]
        (min_factor, max_factor) for time-stretch speed perturbation.
    semitone_range : tuple[int, int]
        (min_semitones, max_semitones) for pitch shifting.
    enable_noise : bool
        Toggle noise injection.
    enable_speed : bool
        Toggle speed perturbation.
    enable_pitch : bool
        Toggle pitch shift.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        snr_range_db: Tuple[float, float] = (5.0, 15.0),
        speed_range: Tuple[float, float] = (0.85, 1.15),
        semitone_range: Tuple[int, int] = (-4, 4),
        enable_noise: bool = True,
        enable_speed: bool = True,
        enable_pitch: bool = True,
        seed: Optional[int] = 42,
    ):
        self.snr_range = snr_range_db
        self.speed_range = speed_range
        self.semitone_range = semitone_range
        self.enable_noise = enable_noise
        self.enable_speed = enable_speed
        self.enable_pitch = enable_pitch
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Individual augmentations
    # ------------------------------------------------------------------
    def add_noise(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Inject additive white Gaussian noise at a random SNR.

        Returns
        -------
        tuple[np.ndarray, float]
            (augmented_signal, applied_snr_db)
        """
        snr_db = self.rng.uniform(*self.snr_range)
        signal_power = np.mean(y ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), size=y.shape).astype(y.dtype)
        augmented = y + noise
        logger.debug("Added noise at SNR=%.1f dB", snr_db)
        return augmented, float(snr_db)

    def change_speed(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """
        Apply speed perturbation via time-stretching + resampling.

        Returns
        -------
        tuple[np.ndarray, float]
            (augmented_signal, speed_factor)
        """
        factor = self.rng.uniform(*self.speed_range)
        stretched = librosa.effects.time_stretch(y, rate=factor)
        logger.debug("Speed perturbation factor=%.2f", factor)
        return stretched, float(factor)

    def shift_pitch(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
        """
        Shift pitch by a random number of semitones.

        Returns
        -------
        tuple[np.ndarray, float]
            (augmented_signal, semitones_shifted)
        """
        n_steps = self.rng.integers(self.semitone_range[0], self.semitone_range[1] + 1)
        shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=int(n_steps))
        logger.debug("Pitch shifted by %d semitones", n_steps)
        return shifted, float(n_steps)

    # ------------------------------------------------------------------
    # Combined chain
    # ------------------------------------------------------------------
    def augment(
        self,
        y: np.ndarray,
        sr: int,
        augmentations: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply a chain of augmentations.

        Parameters
        ----------
        y : np.ndarray
            Input signal.
        sr : int
            Sample rate.
        augmentations : list[str] | None
            Subset of ``["noise", "speed", "pitch"]``.
            If None, apply all enabled augmentations.

        Returns
        -------
        tuple[np.ndarray, dict]
            (augmented_signal, metadata_dict)
        """
        if augmentations is None:
            augmentations = []
            if self.enable_noise:
                augmentations.append("noise")
            if self.enable_speed:
                augmentations.append("speed")
            if self.enable_pitch:
                augmentations.append("pitch")

        meta: dict = {}
        result = y.copy()

        for aug in augmentations:
            if aug == "noise":
                result, snr = self.add_noise(result)
                meta["noise_snr_db"] = snr
            elif aug == "speed":
                result, factor = self.change_speed(result, sr)
                meta["speed_factor"] = factor
            elif aug == "pitch":
                result, steps = self.shift_pitch(result, sr)
                meta["pitch_semitones"] = steps
            else:
                logger.warning("Unknown augmentation '%s', skipping.", aug)

        return result, meta
