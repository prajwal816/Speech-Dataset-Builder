"""
End-to-end pipeline smoke test.

Generates synthetic audio, runs the full pipeline, and verifies outputs exist.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.pipeline import DatasetPipeline


def _generate_synthetic_audio(output_dir: str, n_files: int = 4, sr: int = 16000):
    """Create simple sine-wave WAV files for testing."""
    os.makedirs(output_dir, exist_ok=True)
    freqs = [220, 440, 660, 880]
    for i in range(n_files):
        freq = freqs[i % len(freqs)]
        duration = 5.0 + i * 2  # varying lengths
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = os.path.join(output_dir, f"synth_{i:03d}.wav")
        sf.write(path, y, sr)


class TestPipeline:
    """Smoke tests for the full DatasetPipeline."""

    def test_full_pipeline(self, tmp_path):
        """Run build → augment → index → export and check files exist."""
        # Create a temporary config with paths pointing to tmp_path
        raw_dir = os.path.join(str(tmp_path), "raw")
        _generate_synthetic_audio(raw_dir, n_files=4)

        # Write a test config
        config_content = f"""
paths:
  raw_audio_dir: "{raw_dir.replace(os.sep, '/')}"
  processed_dir: "{os.path.join(str(tmp_path), 'processed').replace(os.sep, '/')}"
  augmented_dir: "{os.path.join(str(tmp_path), 'augmented').replace(os.sep, '/')}"
  segments_dir: "{os.path.join(str(tmp_path), 'segments').replace(os.sep, '/')}"
  export_dir: "{os.path.join(str(tmp_path), 'export').replace(os.sep, '/')}"
  metadata_db: "{os.path.join(str(tmp_path), 'storage', 'metadata.json').replace(os.sep, '/')}"
  faiss_index: "{os.path.join(str(tmp_path), 'storage', 'faiss.index').replace(os.sep, '/')}"
  embeddings_store: "{os.path.join(str(tmp_path), 'storage', 'embeddings.npy').replace(os.sep, '/')}"
  log_dir: "{os.path.join(str(tmp_path), 'logs').replace(os.sep, '/')}"

audio:
  target_sample_rate: 16000
  mono: true
  supported_formats: [".wav"]

preprocessing:
  resample: true
  normalize: true
  normalization_method: "peak"
  trim_silence: true
  silence_threshold_db: -40.0
  silence_min_length_ms: 100

augmentation:
  enabled: true
  num_augmented_copies: 1
  seed: 42
  noise_injection:
    enabled: true
    snr_range_db: [5, 15]
  speed_perturbation:
    enabled: false
    factor_range: [0.9, 1.1]
  pitch_shift:
    enabled: false
    semitone_range: [-2, 2]

segmentation:
  enabled: true
  method: "fixed"
  chunk_duration_sec: 5.0
  overlap_sec: 0.5
  min_chunk_duration_sec: 1.0
  speaker_aware: false
  energy_threshold_db: -35.0

metadata:
  default_language: "en"
  default_speaker_id: "test_speaker"

indexing:
  enabled: false
  embedding_dim: 64
  n_mels: 64
  max_frames: 64

export:
  train_ratio: 0.8
  stratify_by: "speaker_id"
  shuffle: true
  seed: 42

logging:
  level: "WARNING"
  format: "%(asctime)s | %(levelname)s | %(message)s"
  file: "{os.path.join(str(tmp_path), 'logs', 'test.log').replace(os.sep, '/')}"
"""
        config_path = os.path.join(str(tmp_path), "test_config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)

        cfg = load_config(config_path)
        pipeline = DatasetPipeline(cfg)
        pipeline.run_all()

        # Verify outputs
        segments_dir = os.path.join(str(tmp_path), "segments")
        augmented_dir = os.path.join(str(tmp_path), "augmented")
        export_dir = os.path.join(str(tmp_path), "export")
        storage_dir = os.path.join(str(tmp_path), "storage")

        assert os.path.isdir(segments_dir)
        assert len(os.listdir(segments_dir)) > 0, "No segments produced"

        assert os.path.isdir(augmented_dir)
        assert len(os.listdir(augmented_dir)) > 0, "No augmented files produced"

        # Note: FAISS indexing is disabled in this test config to prevent faiss-cpu from hanging on Windows.

        assert os.path.isdir(os.path.join(export_dir, "train")), "Train dir missing"
        assert os.path.isdir(os.path.join(export_dir, "test")), "Test dir missing"
        assert os.path.isfile(os.path.join(export_dir, "train", "manifest.json"))
        assert os.path.isfile(os.path.join(export_dir, "test", "manifest.json"))
