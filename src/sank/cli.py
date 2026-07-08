"""Command-line entry point.

    sank run --watchlist config/watchlist.example.yaml \\
              --reference config/reference_corpus.example.yaml \\
              --domain config/domains/competitive_intelligence.yaml \\
              --vector-store local

This is intentionally the only place in the codebase that reads
environment variables or decides which concrete LLMClient/VectorStore to
instantiate — everything downstream (pipeline.py, agents.py) only ever
sees the abstract interfaces. That's what makes `sank run --vector-store
local` (zero API keys, for trying it out) and a real production run with
`--vector-store pinecone` the *same code path*.
"""

from __future__ import annotations

import logging
import os
import sys

import click
from dotenv import load_dotenv

from sank.config import load_domain_config, load_reference_corpus, load_watchlist
from sank.exceptions import SankError
from sank.llm_client import AnthropicLLMClient, GeminiLLMClient
from sank.pipeline import run_pipeline
from sank.vector_store import LocalVectorStore, PineconeVectorStore


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
def cli() -> None:
    """Sank — watch public signals, brief yourself daily against your own reference corpus."""
    load_dotenv()


@cli.command()
@click.option("--watchlist", required=True, type=click.Path(exists=True), help="Path to watchlist YAML.")
@click.option("--reference", required=True, type=click.Path(exists=True), help="Path to reference corpus YAML.")
@click.option("--domain", required=True, type=click.Path(exists=True), help="Path to domain config YAML.")
@click.option(
    "--provider",
    type=click.Choice(["gemini", "anthropic"]),
    default="gemini",
    show_default=True,
    help="'gemini' has a genuine free tier (no card needed, get a key at aistudio.google.com). "
    "'anthropic' needs a funded account.",
)
@click.option(
    "--vector-store",
    type=click.Choice(["local", "pinecone"]),
    default="local",
    show_default=True,
    help="'local' needs no API key (TF-IDF). 'pinecone' needs PINECONE_API_KEY + an embed function wired in.",
)
@click.option("--model", default=None, help="Override the default model for the chosen provider.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def run(
    watchlist: str,
    reference: str,
    domain: str,
    provider: str,
    vector_store: str,
    model: str | None,
    verbose: bool,
) -> None:
    """Run one full Sank cycle and print the digest."""
    _configure_logging(verbose)
    logger = logging.getLogger("sank.cli")

    try:
        domain_cfg = load_domain_config(domain)
        entities = load_watchlist(watchlist)
        reference_items = load_reference_corpus(reference)
    except SankError as exc:
        click.secho(f"Config error: {exc}", fg="red", err=True)
        sys.exit(1)

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            click.secho(
                "GEMINI_API_KEY is not set. Get a free key (no credit card) at "
                "https://aistudio.google.com, then add it to .env (see .env.example).",
                fg="red",
                err=True,
            )
            sys.exit(1)
        llm = GeminiLLMClient(api_key=api_key, **({"model": model} if model else {}))
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            click.secho(
                "ANTHROPIC_API_KEY is not set. Put it in a .env file (see .env.example) "
                "or export it before running.",
                fg="red",
                err=True,
            )
            sys.exit(1)
        llm = AnthropicLLMClient(api_key=api_key, **({"model": model} if model else {}))

    if vector_store == "local":
        vs = LocalVectorStore()
    else:
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            click.secho("PINECONE_API_KEY is not set.", fg="red", err=True)
            sys.exit(1)
        click.secho(
            "Note: --vector-store pinecone needs a real embed_fn wired in "
            "(see README 'Swapping in Pinecone'); using a placeholder will "
            "produce meaningless matches.",
            fg="yellow",
        )
        vs = PineconeVectorStore(
            api_key=pinecone_key,
            index_name="sank-reference",
            embed_fn=lambda text: [0.0] * 384,  # placeholder — replace per README
        )

    logger.info("Watching %d entities across %d domain(s): %s", len(entities), 1, domain_cfg.domain)

    try:
        result = run_pipeline(entities, reference_items, domain_cfg, llm, vs)
    except SankError as exc:
        click.secho(f"Pipeline failed: {exc}", fg="red", err=True)
        sys.exit(1)

    click.echo()
    click.secho("=== Sank Digest ===", bold=True)
    click.echo(result.digest.summary_text)
    click.echo()
    click.secho(
        f"Sources: {result.sources_succeeded}/{result.sources_attempted} succeeded.",
        fg="green" if not result.errors else "yellow",
    )
    for err in result.errors:
        click.secho(f"  - {err}", fg="yellow")


@cli.command()
@click.option("--watchlist", required=True, type=click.Path(exists=True))
def validate(watchlist: str) -> None:
    """Just load and validate a watchlist file without running the pipeline."""
    try:
        entities = load_watchlist(watchlist)
    except SankError as exc:
        click.secho(f"Invalid: {exc}", fg="red", err=True)
        sys.exit(1)
    click.secho(f"Valid — {len(entities)} entities, {sum(len(e.sources) for e in entities)} sources.", fg="green")


if __name__ == "__main__":
    cli()
