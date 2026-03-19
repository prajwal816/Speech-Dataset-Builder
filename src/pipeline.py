"""
Dataset Pipeline Orchestrator.

Drives the full flow:  ingest → preprocess → augment → segment → metadata → index → export
All steps are config-driven with structured logging.
"""

import logging
import os
import shutil
from typing import List, Optional

import numpy as np
import soundfile as sf

from src.config import Config, load_config
from src.ingestion.loader import AudioLoader
from src.preprocessing.processor import AudioProcessor
from src.augmentation.engine import AugmentationEngine
from src.segmentation.chunker import AudioChunker
from src.metadata.store import MetadataStore
from src.indexing.faiss_index import FAISSIndex
from src.indexing.retrieval import SubsetRetriever

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    End-to-end dataset building pipeline.

    Parameters
    ----------
    config : Config
        Loaded pipeline configuration.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self._setup_logging()
        self._ensure_dirs()

        # Instantiate components
        self.loader = AudioLoader(
            source_dir=self.cfg.paths.raw_audio_dir,
            target_sr=self.cfg.audio.target_sample_rate,
            mono=self.cfg.audio.mono,
            supported_formats=self.cfg.audio.supported_formats,
        )

        self.processor = AudioProcessor(
            target_sr=self.cfg.audio.target_sample_rate,
            normalize=self.cfg.preprocessing.normalize,
            normalization_method=self.cfg.preprocessing.normalization_method,
            trim_silence=self.cfg.preprocessing.trim_silence,
            silence_threshold_db=self.cfg.preprocessing.silence_threshold_db,
        )

        aug_cfg = self.cfg.augmentation
        self.augmentor = AugmentationEngine(
            snr_range_db=tuple(aug_cfg.noise_injection.snr_range_db),
            speed_range=tuple(aug_cfg.speed_perturbation.factor_range),
            semitone_range=tuple(aug_cfg.pitch_shift.semitone_range),
            enable_noise=aug_cfg.noise_injection.enabled,
            enable_speed=aug_cfg.speed_perturbation.enabled,
            enable_pitch=aug_cfg.pitch_shift.enabled,
            seed=aug_cfg.seed,
        )

        seg_cfg = self.cfg.segmentation
        self.chunker = AudioChunker(
            chunk_duration=seg_cfg.chunk_duration_sec,
            overlap=seg_cfg.overlap_sec,
            min_duration=seg_cfg.min_chunk_duration_sec,
            method=seg_cfg.method,
            speaker_aware=seg_cfg.speaker_aware,
            energy_threshold_db=seg_cfg.energy_threshold_db,
            sr=self.cfg.audio.target_sample_rate,
        )

        self.meta_store = MetadataStore(db_path=self.cfg.paths.metadata_db)

        idx_cfg = self.cfg.indexing
        self.faiss_index = FAISSIndex(
            embedding_dim=idx_cfg.embedding_dim,
            n_mels=idx_cfg.n_mels,
            max_frames=idx_cfg.max_frames,
            sr=self.cfg.audio.target_sample_rate,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_logging(self) -> None:
        log_cfg = self.cfg.logging
        os.makedirs(os.path.dirname(log_cfg.file), exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, log_cfg.level, logging.INFO),
            format=log_cfg.format,
            handlers=[
                logging.FileHandler(log_cfg.file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        logger.info("Logging initialised — level=%s", log_cfg.level)

    def _ensure_dirs(self) -> None:
        for attr in ["raw_audio_dir", "processed_dir", "augmented_dir", "segments_dir", "export_dir"]:
            path = getattr(self.cfg.paths, attr)
            os.makedirs(path, exist_ok=True)
        os.makedirs(self.cfg.paths.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.paths.metadata_db), exist_ok=True)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def build(self) -> None:
        """
        Run ingestion → preprocessing → segmentation → metadata.

        Produces clean, segmented audio and populates the metadata store.
        """
        logger.info("═══ STAGE: BUILD ═══")
        sr = self.cfg.audio.target_sample_rate
        segment_counter = 0

        for batch in self.loader.iter_batch(batch_size=16):
            for fpath, y, _ in batch:
                # Preprocess
                y_clean, _ = self.processor.process(y, sr)

                # Save processed (full) file
                basename = os.path.splitext(os.path.basename(fpath))[0]
                processed_path = os.path.join(self.cfg.paths.processed_dir, f"{basename}.wav")
                sf.write(processed_path, y_clean, sr)

                # Segment
                segments = self.chunker.segment(y_clean)
                for idx, seg in enumerate(segments):
                    seg_name = f"{basename}_seg{idx:04d}.wav"
                    seg_path = os.path.join(self.cfg.paths.segments_dir, seg_name)
                    sf.write(seg_path, seg["audio"], sr)

                    # Metadata
                    self.meta_store.insert({
                        "file_path": seg_path,
                        "source_file": fpath,
                        "speaker_id": str(seg.get("speaker_id", self.cfg.metadata.default_speaker_id)),
                        "language": self.cfg.metadata.default_language,
                        "duration": seg["duration"],
                        "sample_rate": sr,
                        "noise_level": None,
                        "augmentation": {},
                        "segment_index": idx,
                    })
                    segment_counter += 1

        logger.info("BUILD complete — %d segments, %d metadata records",
                     segment_counter, self.meta_store.count())

    def augment(self) -> None:
        """
        Generate augmented copies of every segment.

        Reads segments from ``segments_dir``, writes augmented versions to
        ``augmented_dir``, and appends metadata.
        """
        logger.info("═══ STAGE: AUGMENT ═══")
        if not self.cfg.augmentation.enabled:
            logger.info("Augmentation disabled — skipping.")
            return

        sr = self.cfg.audio.target_sample_rate
        n_copies = self.cfg.augmentation.num_augmented_copies
        aug_count = 0

        seg_dir = self.cfg.paths.segments_dir
        for fname in sorted(os.listdir(seg_dir)):
            if not fname.endswith(".wav"):
                continue
            seg_path = os.path.join(seg_dir, fname)
            y, _ = sf.read(seg_path, dtype="float32")

            for copy_idx in range(n_copies):
                y_aug, aug_meta = self.augmentor.augment(y, sr)
                aug_name = f"{os.path.splitext(fname)[0]}_aug{copy_idx:02d}.wav"
                aug_path = os.path.join(self.cfg.paths.augmented_dir, aug_name)
                sf.write(aug_path, y_aug, sr)

                self.meta_store.insert({
                    "file_path": aug_path,
                    "source_file": seg_path,
                    "speaker_id": self.cfg.metadata.default_speaker_id,
                    "language": self.cfg.metadata.default_language,
                    "duration": len(y_aug) / sr,
                    "sample_rate": sr,
                    "noise_level": aug_meta.get("noise_snr_db"),
                    "augmentation": aug_meta,
                    "segment_index": -1,
                })
                aug_count += 1

        logger.info("AUGMENT complete — %d augmented files", aug_count)

    def index(self) -> None:
        """
        Build FAISS index from all segments + augmented audio.
        """
        logger.info("═══ STAGE: INDEX ═══")
        if not self.cfg.indexing.enabled:
            logger.info("Indexing disabled — skipping.")
            return

        signals: List[np.ndarray] = []
        for record in self.meta_store.all():
            fpath = record.get("file_path", "")
            if os.path.isfile(fpath):
                y, _ = sf.read(fpath, dtype="float32")
                signals.append(y)

        if not signals:
            logger.warning("No audio files to index.")
            return

        self.faiss_index.build(signals)
        self.faiss_index.save(
            self.cfg.paths.faiss_index,
            self.cfg.paths.embeddings_store,
        )
        logger.info("INDEX complete — %d vectors", self.faiss_index.index.ntotal)

    def export(self, output_dir: Optional[str] = None) -> None:
        """
        Export train/test split to disk.

        Creates ``train/`` and ``test/`` subdirectories with audio copies
        and a manifest JSON per split.
        """
        import json

        logger.info("═══ STAGE: EXPORT ═══")
        export_dir = output_dir or self.cfg.paths.export_dir

        retriever = SubsetRetriever(self.meta_store.all(), seed=self.cfg.export.seed)
        train, test = retriever.train_test_split(
            train_ratio=self.cfg.export.train_ratio,
            stratify_by=self.cfg.export.stratify_by,
        )

        for split_name, split_records in [("train", train), ("test", test)]:
            split_dir = os.path.join(export_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            manifest = []
            for rec in split_records:
                src = rec.get("file_path", "")
                if os.path.isfile(src):
                    dst = os.path.join(split_dir, os.path.basename(src))
                    shutil.copy2(src, dst)
                    entry = dict(rec)
                    entry["file_path"] = dst
                    manifest.append(entry)

            manifest_path = os.path.join(split_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, default=str)
            logger.info("Exported %d files to %s", len(manifest), split_dir)

        logger.info("EXPORT complete.")

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------
    def run_all(self) -> None:
        """Execute the complete pipeline: build → augment → index → export."""
        self.build()
        self.augment()
        self.index()
        self.export()
        logger.info("════════ PIPELINE COMPLETE ════════")
