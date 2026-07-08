"""Shared fixtures for the Sank test suite.

Every fixture here builds an in-memory or offline stand-in for an external
dependency — no test in this suite needs an API key, a Pinecone account, or
network access. That's a deliberate design property of the package
(LLMClient and VectorStore are interfaces), not a workaround.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sank.config import load_domain_config
from sank.models import Entity, ReferenceItem, Source

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"
CONFIG_DIR = Path(__file__).parent.parent / "config"


@pytest.fixture
def domain_config():
    return load_domain_config(CONFIG_DIR / "domains" / "competitive_intelligence.yaml")


@pytest.fixture
def creator_domain_config():
    return load_domain_config(CONFIG_DIR / "domains" / "creator_comparison.yaml")


@pytest.fixture
def sample_reference_items() -> list[ReferenceItem]:
    return [
        ReferenceItem(id="rm-001", text="Reduce time-to-first-deploy from prompt to live URL"),
        ReferenceItem(id="rm-002", text="Let teams collaborate on the same project in real time"),
        ReferenceItem(id="rm-003", text="One-click custom domain connection with automatic SSL"),
    ]


@pytest.fixture
def sample_entity() -> Entity:
    return Entity(
        name="Cursor",
        category="AI coding IDE",
        sources=[Source(url="https://cursor.com/changelog", source_type="changelog")],
    )


@pytest.fixture
def sample_changelog_html() -> str:
    return (SAMPLE_DATA_DIR / "sample_changelog.html").read_text()
