"""Tests for the Audio Preprocessing module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.processor import AudioProcessor


class TestAudioProcessor:
    """Unit tests for AudioProcessor."""

    def _make_sine(self, freq=440, duration=2.0, sr=44100, amplitude=0.5):
        """Generate a sine-wave test signal."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr

    # ── Resampling ───────────────────────────────────────────
    def test_resample_changes_length(self):
        y, sr = self._make_sine(sr=44100, duration=1.0)
        proc = AudioProcessor(target_sr=16000)
        y_out = proc.resample(y, orig_sr=sr)
        expected_len = int(len(y) * 16000 / 44100)
        assert abs(len(y_out) - expected_len) < 10

    def test_resample_noop_when_same_sr(self):
        y, sr = self._make_sine(sr=16000, duration=1.0)
        proc = AudioProcessor(target_sr=16000)
        y_out = proc.resample(y, orig_sr=sr)
        assert len(y_out) == len(y)

    # ── Silence Trimming ─────────────────────────────────────
    def test_trim_removes_silence(self):
        sr = 16000
        silence = np.zeros(sr, dtype=np.float32)  # 1 s silence
        y, _ = self._make_sine(sr=sr, duration=1.0)
        signal = np.concatenate([silence, y, silence])
        proc = AudioProcessor(target_sr=sr, trim_silence=True, silence_threshold_db=-40)
        trimmed = proc.trim(signal)
        assert len(trimmed) < len(signal)

    # ── Normalization ────────────────────────────────────────
    def test_peak_normalization(self):
        y, sr = self._make_sine(sr=16000, amplitude=0.3)
        proc = AudioProcessor(target_sr=sr, normalization_method="peak")
        y_norm = proc.normalize_signal(y)
        assert abs(np.max(np.abs(y_norm)) - 1.0) < 1e-5

    def test_rms_normalization_clips(self):
        y, sr = self._make_sine(sr=16000, amplitude=0.5)
        proc = AudioProcessor(target_sr=sr, normalization_method="rms")
        y_norm = proc.normalize_signal(y)
        assert np.max(np.abs(y_norm)) <= 1.0 + 1e-6

    # ── Full pipeline ────────────────────────────────────────
    def test_process_returns_target_sr(self):
        y, sr = self._make_sine(sr=44100, duration=1.5)
        proc = AudioProcessor(target_sr=16000)
        y_out, sr_out = proc.process(y, sr)
        assert sr_out == 16000
        assert len(y_out) > 0
