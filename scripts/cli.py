#!/usr/bin/env python3
"""
CLI for the Next-Generation Speech AI Dataset Builder.

Commands
--------
    build     — Ingest, preprocess, and segment raw audio.
    augment   — Generate augmented copies of segments.
    index     — Build FAISS vector index.
    export    — Export train/test split.
    query     — Query metadata store.
    run       — Execute the full pipeline end-to-end.
"""

import json
import sys
import os

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click

from src.config import load_config
from src.pipeline import DatasetPipeline
from src.metadata.store import MetadataStore


@click.group()
@click.option(
    "--config", "-c",
    default="configs/default.yaml",
    help="Path to YAML config file.",
    type=click.Path(exists=True),
)
@click.pass_context
def cli(ctx, config):
    """Next-Generation Speech AI Dataset Builder CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)


# ── Build ────────────────────────────────────────────────────────────
@cli.command()
@click.pass_context
def build(ctx):
    """Ingest raw audio, preprocess, and segment into training-ready chunks."""
    pipeline = DatasetPipeline(ctx.obj["config"])
    pipeline.build()
    click.echo("✅  Build complete.")


# ── Augment ──────────────────────────────────────────────────────────
@cli.command()
@click.pass_context
def augment(ctx):
    """Generate augmented copies of segmented audio."""
    pipeline = DatasetPipeline(ctx.obj["config"])
    pipeline.augment()
    click.echo("✅  Augmentation complete.")


# ── Index ────────────────────────────────────────────────────────────
@cli.command()
@click.option("--rebuild", is_flag=True, help="Rebuild index from scratch.")
@click.pass_context
def index(ctx, rebuild):
    """Build or rebuild the FAISS vector index."""
    pipeline = DatasetPipeline(ctx.obj["config"])
    pipeline.index()
    click.echo("✅  Indexing complete.")


# ── Export ───────────────────────────────────────────────────────────
@cli.command()
@click.option("--output", "-o", default=None, help="Override export directory.")
@click.pass_context
def export(ctx, output):
    """Export train/test split with manifest files."""
    pipeline = DatasetPipeline(ctx.obj["config"])
    pipeline.export(output_dir=output)
    click.echo("✅  Export complete.")


# ── Query ────────────────────────────────────────────────────────────
@cli.command()
@click.option("--speaker", "-s", default=None, help="Filter by speaker ID.")
@click.option("--language", "-l", default=None, help="Filter by language.")
@click.option("--min-duration", type=float, default=None, help="Min duration (seconds).")
@click.option("--max-duration", type=float, default=None, help="Max duration (seconds).")
@click.option("--limit", "-n", type=int, default=10, help="Max results to show.")
@click.pass_context
def query(ctx, speaker, language, min_duration, max_duration, limit):
    """Query the metadata store with optional filters."""
    cfg = ctx.obj["config"]
    store = MetadataStore(db_path=cfg.paths.metadata_db)

    results = store.all()

    if speaker:
        results = [r for r in results if r.get("speaker_id") == speaker]
    if language:
        results = [r for r in results if r.get("language") == language]
    if min_duration is not None:
        results = [r for r in results if r.get("duration", 0) >= min_duration]
    if max_duration is not None:
        results = [r for r in results if r.get("duration", 0) <= max_duration]

    results = results[:limit]

    click.echo(f"Found {len(results)} result(s):\n")
    for rec in results:
        click.echo(json.dumps(rec, indent=2, default=str))
        click.echo("---")

    store.close()


# ── Run (full pipeline) ─────────────────────────────────────────────
@cli.command()
@click.pass_context
def run(ctx):
    """Run the full pipeline: build → augment → index → export."""
    pipeline = DatasetPipeline(ctx.obj["config"])
    pipeline.run_all()
    click.echo("✅  Full pipeline complete.")


if __name__ == "__main__":
    cli()
