"""Tests for the Augmentation Engine."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.augmentation.engine import AugmentationEngine


class TestAugmentationEngine:
    """Unit tests for AugmentationEngine."""

    def _make_sine(self, freq=440, duration=2.0, sr=16000, amplitude=0.5):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr

    # ── Noise Injection ──────────────────────────────────────
    def test_noise_modifies_signal(self):
        y, sr = self._make_sine()
        engine = AugmentationEngine(seed=0)
        y_noisy, snr = engine.add_noise(y)
        assert not np.allclose(y, y_noisy)
        assert 0 < snr <= 20

    def test_noise_snr_within_range(self):
        y, sr = self._make_sine()
        engine = AugmentationEngine(snr_range_db=(8.0, 12.0), seed=42)
        _, snr = engine.add_noise(y)
        assert 8.0 <= snr <= 12.0

    # ── Speed Perturbation ───────────────────────────────────
    def test_speed_changes_length(self):
        y, sr = self._make_sine(duration=3.0)
        engine = AugmentationEngine(speed_range=(0.5, 0.6), seed=42)
        y_fast, factor = engine.change_speed(y, sr)
        # Speeding up → shorter; slowing down → longer
        assert len(y_fast) != len(y)

    # ── Pitch Shift ──────────────────────────────────────────
    def test_pitch_preserves_duration(self):
        y, sr = self._make_sine(duration=1.0)
        engine = AugmentationEngine(semitone_range=(-2, 2), seed=42)
        y_shifted, steps = engine.shift_pitch(y, sr)
        # librosa pitch_shift preserves length
        assert len(y_shifted) == len(y)

    # ── Chain ────────────────────────────────────────────────
    def test_augment_chain_returns_meta(self):
        y, sr = self._make_sine()
        engine = AugmentationEngine(seed=42)
        y_aug, meta = engine.augment(y, sr)
        assert "noise_snr_db" in meta
        assert "speed_factor" in meta
        assert "pitch_semitones" in meta
        assert len(y_aug) > 0
