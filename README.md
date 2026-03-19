# 🎙️ Next-Generation Speech AI Dataset Builder

A **scalable, config-driven pipeline** for generating, augmenting, indexing, and managing large-scale speech datasets — designed for training production speech AI models.

---

## 📑 Table of Contents

- [Pipeline Overview](#-pipeline-overview)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [CLI Reference](#-cli-reference)
- [Augmentation Strategy](#-augmentation-strategy)
- [Metadata Schema](#-metadata-schema)
- [Indexing Strategy](#-indexing-strategy)
- [Example Queries](#-example-queries)
- [Scaling Strategy](#-scaling-strategy)
- [Configuration](#-configuration)
- [Testing](#-testing)

---

## 🔄 Pipeline Overview

```
┌─────────────┐    ┌───────────────┐    ┌──────────────┐    ┌─────────────┐
│  Raw Audio   │───▶│ Preprocessing │───▶│ Segmentation │───▶│  Metadata   │
│  Ingestion   │    │  Resample     │    │  Fixed/Energy│    │   Store     │
│  WAV/FLAC/   │    │  Trim         │    │  Speaker-    │    │  TinyDB     │
│  MP3/OGG     │    │  Normalize    │    │  aware       │    │             │
└─────────────┘    └───────────────┘    └──────┬───────┘    └──────┬──────┘
                                               │                   │
                                               ▼                   ▼
                                        ┌──────────────┐    ┌─────────────┐
                                        │ Augmentation │    │   FAISS     │
                                        │  Noise       │    │  Vector     │
                                        │  Speed       │    │  Index      │
                                        │  Pitch       │    │             │
                                        └──────┬───────┘    └──────┬──────┘
                                               │                   │
                                               ▼                   ▼
                                        ┌──────────────────────────────────┐
                                        │         Dataset Export           │
                                        │   Train/Test Split + Manifest    │
                                        └──────────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Ingest** | `src/ingestion/loader.py` | Load WAV/FLAC/MP3/OGG, multi-speaker channel separation, batch iteration |
| **Preprocess** | `src/preprocessing/processor.py` | Resample to target SR, energy-based silence trimming, peak/RMS normalization |
| **Segment** | `src/segmentation/chunker.py` | Fixed-length or energy-based chunking, speaker-aware segmentation |
| **Augment** | `src/augmentation/engine.py` | Noise injection (0–15 dB SNR), speed perturbation, pitch shift |
| **Metadata** | `src/metadata/store.py` | TinyDB-backed per-sample metadata with query/filter/aggregate |
| **Index** | `src/indexing/faiss_index.py` | FAISS flat L2 index on MelSpectrogram embeddings for kNN retrieval |
| **Export** | `src/indexing/retrieval.py` | Stratified train/test split, custom subset sampling, vector similarity retrieval |

---

## 📁 Project Structure

```
Speech-Dataset-Builder/
├── configs/
│   └── default.yaml              # Full pipeline configuration
├── data/
│   ├── raw/                      # Input audio files
│   ├── processed/                # Cleaned audio
│   ├── augmented/                # Augmented copies
│   ├── segments/                 # Training-ready chunks
│   └── export/                   # Final train/test splits
│       ├── train/
│       │   └── manifest.json
│       └── test/
│           └── manifest.json
├── storage/
│   ├── metadata.json             # TinyDB metadata store
│   ├── faiss.index               # FAISS vector index
│   └── embeddings.npy            # Raw embedding vectors
├── src/
│   ├── config.py                 # YAML config loader
│   ├── pipeline.py               # Pipeline orchestrator
│   ├── ingestion/
│   │   └── loader.py             # Audio file loading
│   ├── preprocessing/
│   │   └── processor.py          # Resample, trim, normalize
│   ├── augmentation/
│   │   └── engine.py             # Noise, speed, pitch augmentation
│   ├── segmentation/
│   │   └── chunker.py            # Audio chunking
│   ├── metadata/
│   │   └── store.py              # TinyDB metadata store
│   └── indexing/
│       ├── faiss_index.py        # FAISS index builder
│       └── retrieval.py          # Subset retrieval system
├── scripts/
│   └── cli.py                    # Click CLI
├── tests/
│   ├── test_preprocessing.py
│   ├── test_augmentation.py
│   ├── test_metadata.py
│   └── test_pipeline.py          # End-to-end smoke test
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Raw Audio

Place your audio files (`.wav`, `.flac`, `.mp3`, `.ogg`) into `data/raw/`.

### 3. Run the Full Pipeline

```bash
python scripts/cli.py run --config configs/default.yaml
```

This executes: **ingest → preprocess → segment → augment → index → export**.

### 4. Check Outputs

```
data/segments/      ← Clean, segmented audio
data/augmented/     ← Augmented copies
data/export/train/  ← Training split + manifest.json
data/export/test/   ← Test split + manifest.json
storage/            ← Metadata DB + FAISS index
```

---

## ⌨️ CLI Reference

| Command | Description |
|---------|-------------|
| `python scripts/cli.py run -c <config>` | Full pipeline end-to-end |
| `python scripts/cli.py build -c <config>` | Ingest → preprocess → segment |
| `python scripts/cli.py augment -c <config>` | Generate augmented copies |
| `python scripts/cli.py index -c <config>` | Build FAISS vector index |
| `python scripts/cli.py export -c <config>` | Export train/test split |
| `python scripts/cli.py query -s <speaker> -l <lang>` | Query metadata store |

### Query Examples

```bash
# All English segments
python scripts/cli.py query -c configs/default.yaml --language en

# Segments from speaker spk_001, min 5 seconds
python scripts/cli.py query -c configs/default.yaml --speaker spk_001 --min-duration 5.0

# Show top 20 results
python scripts/cli.py query -c configs/default.yaml --limit 20
```

---

## 🎛️ Augmentation Strategy

The engine applies **three augmentation types** in a configurable chain:

### Noise Injection
- **Method**: Additive white Gaussian noise
- **SNR range**: 5–15 dB (configurable)
- **Purpose**: Simulate real-world recording conditions (background noise, microphone hiss)

### Speed Perturbation
- **Method**: Time-stretching via `librosa.effects.time_stretch`
- **Factor range**: 0.85×–1.15× (configurable)
- **Purpose**: Simulate natural speaking rate variation

### Pitch Shift
- **Method**: Pitch shifting via `librosa.effects.pitch_shift`
- **Range**: ±4 semitones (configurable)
- **Purpose**: Simulate speaker variation (F0 differences)

**Reproducibility**: All augmentations use a seeded random number generator (`numpy.random.default_rng`).

```yaml
augmentation:
  num_augmented_copies: 2
  seed: 42
  noise_injection:
    snr_range_db: [5, 15]
  speed_perturbation:
    factor_range: [0.85, 1.15]
  pitch_shift:
    semitone_range: [-4, 4]
```

---

## 📊 Metadata Schema

Each segment has a JSON metadata record stored in **TinyDB**:

| Field | Type | Description |
|-------|------|-------------|
| `file_path` | `str` | Absolute path to the audio file |
| `source_file` | `str` | Original raw audio file path |
| `speaker_id` | `str` | Speaker identifier |
| `language` | `str` | ISO language code (e.g. `en`, `fr`) |
| `duration` | `float` | Duration in seconds |
| `sample_rate` | `int` | Sample rate (Hz) |
| `noise_level` | `float\|null` | Applied noise SNR in dB (null if clean) |
| `augmentation` | `dict` | Applied augmentation parameters |
| `segment_index` | `int` | Index within the source file |
| `created_at` | `str` | ISO-8601 creation timestamp |

### Example Record

```json
{
  "file_path": "data/segments/interview_seg0003.wav",
  "source_file": "data/raw/interview.wav",
  "speaker_id": "spk_001",
  "language": "en",
  "duration": 9.87,
  "sample_rate": 16000,
  "noise_level": null,
  "augmentation": {},
  "segment_index": 3,
  "created_at": "2026-03-19T07:48:00+00:00"
}
```

---

## 🔍 Indexing Strategy

### FAISS Vector Index

Audio segments are indexed using **FAISS (Facebook AI Similarity Search)** for fast nearest-neighbour retrieval:

1. **Embedding extraction**: Each audio segment → Mel spectrogram → mean-pooled across time → fixed-size vector (default 128-d)
2. **Index type**: `IndexFlatL2` (exact brute-force L2 search)
3. **Operations**: `build()`, `add()`, `search(query, k)`, `save()`, `load()`

```python
from src.indexing.faiss_index import FAISSIndex

index = FAISSIndex(embedding_dim=128)
index.build(audio_signals)

# Find 5 nearest neighbours
distances, indices = index.search(query_audio, k=5)
```

### Subset Retrieval

The `SubsetRetriever` combines **metadata filtering + vector similarity**:

```python
from src.indexing.retrieval import SubsetRetriever

retriever = SubsetRetriever(metadata_records)

# Metadata-based filtering
english_set = retriever.filter_by_language("en")
long_clips = retriever.filter_by_duration(min_sec=5.0, max_sec=30.0)

# Stratified train/test split
train, test = retriever.train_test_split(train_ratio=0.8, stratify_by="speaker_id")

# Random subset sampling
sample = retriever.sample_subset(n=1000)

# Combine FAISS results with metadata
similar = SubsetRetriever.subset_by_similarity(distances, indices, metadata_records)
```

---

## 🔎 Example Queries (Dataset Retrieval)

### Python API

```python
from src.metadata.store import MetadataStore

store = MetadataStore("storage/metadata.json")

# 1. All segments for a specific speaker
spk_segments = store.query("speaker_id", "spk_001")

# 2. English segments between 3–10 seconds
store.filter_by(language="en")
short_en = store.query_range("duration", 3.0, 10.0)

# 3. Total hours of recorded data
total_hours = store.total_duration() / 3600
print(f"Total dataset: {total_hours:.1f} hours")

# 4. List all unique languages
languages = store.unique_values("language")

# 5. Find augmented samples (noise level is not None)
augmented = [r for r in store.all() if r.get("noise_level") is not None]
```

### CLI

```bash
# All English clips from speaker spk_002
python scripts/cli.py query -c configs/default.yaml --speaker spk_002 --language en

# Clips between 5–15 seconds
python scripts/cli.py query -c configs/default.yaml --min-duration 5 --max-duration 15
```

---

## 📈 Scaling Strategy

This pipeline is designed with scaling to **millions of samples** in mind:

### Data Layer
| Challenge | Solution |
|-----------|----------|
| Large file I/O | Batch loading with configurable `batch_size` and `tqdm` progress |
| Disk space | Generate augmentations lazily or limit `num_augmented_copies` |
| File enumeration | Recursive `os.walk` with extension filtering |

### Metadata Layer
| Challenge | Solution |
|-----------|----------|
| JSON at scale | Migrate from TinyDB → **MongoDB** (same query interface) |
| Schema evolution | Flexible document store — no migrations needed |
| Fast lookups | Create indexes on `speaker_id`, `language`, `duration` |

### Indexing Layer
| Challenge | Solution |
|-----------|----------|
| Millions of vectors | Switch from `IndexFlatL2` → **`IndexIVFFlat`** or **`IndexIVFPQ`** |
| Memory pressure | Use `faiss.index_cpu_to_gpu()` for GPU-accelerated search |
| Sharding | Partition FAISS indexes by language or speaker |

### Compute Layer
| Challenge | Solution |
|-----------|----------|
| CPU bottleneck | Parallelize augmentation with `multiprocessing.Pool` |
| I/O bound | Use `concurrent.futures.ThreadPoolExecutor` for file I/O |
| Distributed | Deploy on **Apache Spark** or **Dask** for cluster-scale |

### Recommended Architecture at Scale

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Object Store│     │   MongoDB    │     │  FAISS GPU   │
│  (S3/GCS)    │────▶│  Metadata    │────▶│  Index Shard │
│  Audio files │     │  Cluster     │     │  per Language │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
┌──────────────────────────────────────────────────────────┐
│            Spark / Dask Processing Cluster               │
│   Ingest → Preprocess → Augment → Segment → Index       │
└──────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

All pipeline behaviour is controlled from a **single YAML config** (`configs/default.yaml`).

Key sections:

| Section | Controls |
|---------|----------|
| `paths` | Input/output directories, metadata DB, FAISS index paths |
| `audio` | Target sample rate, mono/stereo, supported formats |
| `preprocessing` | Normalize method, silence threshold, trim settings |
| `augmentation` | Enable/disable each aug type, SNR/speed/pitch ranges, copies |
| `segmentation` | Chunk method (fixed/energy), duration, overlap, speaker-aware |
| `metadata` | Default language and speaker ID |
| `indexing` | Embedding dimension, mel bands, max frames |
| `export` | Train ratio, stratification field, shuffle seed |
| `logging` | Log level, format, output file |

Override for a custom run:

```bash
python scripts/cli.py run --config configs/my_custom.yaml
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Individual test suites
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_augmentation.py -v
python -m pytest tests/test_metadata.py -v
python -m pytest tests/test_pipeline.py -v    # end-to-end smoke test
```

---

## 📜 License

MIT

---

Built with ❤️ for speech AI researchers and ML engineers.
