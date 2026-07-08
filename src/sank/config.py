"""Config loading: turns YAML files into validated, typed Python objects.

Three kinds of config, loaded independently so they can be swapped
independently:

1. Domain config   — WHAT VOCABULARY/PROMPTS to use (competitive intel vs
                      creator comparison vs anything else).
2. Watchlist config — WHO/WHAT to watch (the list of Entities + Sources).
3. Reference corpus  — WHAT TO COMPARE AGAINST (your roadmap, your content
                      pillars, whatever "your own stuff" means in this domain).

Swapping domains means swapping these three files. The pipeline code never
changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from sank.exceptions import ConfigError
from sank.models import Entity, ReferenceItem


class DomainConfig(BaseModel):
    """Vocabulary + prompt templates for one domain.

    `entity_label` / `reference_label` are injected into prompts purely so
    the LLM's output reads naturally (e.g. "competitor" vs "creator"); they
    never affect control flow.
    """

    domain: str
    entity_label: str
    reference_label: str
    extraction_system_prompt: str
    scoring_system_prompt: str
    briefing_system_prompt: str


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in {p}: {exc}") from exc
    if data is None:
        raise ConfigError(f"Config file is empty: {p}")
    return data


def load_domain_config(path: str | Path) -> DomainConfig:
    data = _load_yaml(path)
    try:
        return DomainConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid domain config in {path}:\n{exc}") from exc


def load_watchlist(path: str | Path) -> list[Entity]:
    data = _load_yaml(path)
    entities_raw = data.get("entities") if isinstance(data, dict) else data
    if not entities_raw:
        raise ConfigError(f"No entities found in watchlist file: {path}")
    try:
        return [Entity.model_validate(e) for e in entities_raw]
    except ValidationError as exc:
        raise ConfigError(f"Invalid watchlist entry in {path}:\n{exc}") from exc


def load_reference_corpus(path: str | Path) -> list[ReferenceItem]:
    data = _load_yaml(path)
    items_raw = data.get("reference_items") if isinstance(data, dict) else data
    if not items_raw:
        raise ConfigError(f"No reference_items found in {path}")
    try:
        return [ReferenceItem.model_validate(i) for i in items_raw]
    except ValidationError as exc:
        raise ConfigError(f"Invalid reference item in {path}:\n{exc}") from exc
