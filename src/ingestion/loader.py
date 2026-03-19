"""
Audio Ingestion Module.

Loads raw audio files (WAV, FLAC, MP3, OGG) with support for
multi-speaker channel separation and batch loading.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Load and iterate over raw audio files from a directory.

    Parameters
    ----------
    source_dir : str
        Directory containing raw audio files.
    target_sr : int
        Target sample rate for loading (resampling happens here for consistency).
    mono : bool
        If True, force mono by averaging channels.
    supported_formats : list[str]
        File extensions to include (e.g. ['.wav', '.flac']).
    """

    def __init__(
        self,
        source_dir: str,
        target_sr: int = 16000,
        mono: bool = True,
        supported_formats: Optional[List[str]] = None,
    ):
        self.source_dir = os.path.abspath(source_dir)
        self.target_sr = target_sr
        self.mono = mono
        self.supported_formats = supported_formats or [".wav", ".flac", ".mp3", ".ogg"]
        self._file_list: List[str] = []
        self._scan()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _scan(self) -> None:
        """Walk source directory and collect matching audio files."""
        if not os.path.isdir(self.source_dir):
            logger.warning("Source directory does not exist: %s", self.source_dir)
            return
        for root, _, files in os.walk(self.source_dir):
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() in self.supported_formats:
                    self._file_list.append(os.path.join(root, fname))
        logger.info("Found %d audio files in %s", len(self._file_list), self.source_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def files(self) -> List[str]:
        """Return list of discovered audio file paths."""
        return list(self._file_list)

    def load(self, path: str) -> Tuple[np.ndarray, int]:
        """
        Load a single audio file.

        Returns
        -------
        tuple[np.ndarray, int]
            (audio_signal, sample_rate)
        """
        y, sr = librosa.load(path, sr=self.target_sr, mono=self.mono)
        logger.debug("Loaded %s  duration=%.2fs  sr=%d", path, len(y) / sr, sr)
        return y, sr

    def load_multichannel(self, path: str) -> Dict[int, np.ndarray]:
        """
        Load a multi-channel file and return per-channel arrays.

        Useful for multi-speaker conversations recorded on separate channels.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping of channel index → mono signal.
        """
        data, sr = sf.read(path, always_2d=True)
        # Resample each channel independently
        channels: Dict[int, np.ndarray] = {}
        for ch in range(data.shape[1]):
            signal = data[:, ch].astype(np.float32)
            if sr != self.target_sr:
                signal = librosa.resample(signal, orig_sr=sr, target_sr=self.target_sr)
            channels[ch] = signal
        logger.info("Loaded %d channels from %s", len(channels), path)
        return channels

    def iter_batch(self, batch_size: int = 32, show_progress: bool = True):
        """
        Yield batches of (path, audio, sr) tuples.

        Parameters
        ----------
        batch_size : int
            Number of files per batch.
        show_progress : bool
            Show a tqdm progress bar.

        Yields
        ------
        list[tuple[str, np.ndarray, int]]
        """
        batch: List[Tuple[str, np.ndarray, int]] = []
        iterator = tqdm(self._file_list, desc="Loading audio", disable=not show_progress)
        for fpath in iterator:
            try:
                y, sr = self.load(fpath)
                batch.append((fpath, y, sr))
            except Exception as exc:
                logger.error("Failed to load %s: %s", fpath, exc)
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
