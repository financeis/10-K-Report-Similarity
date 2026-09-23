from pathlib import Path

import pytest
from pydantic import ValidationError

from tenksim.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_repo_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.methods and cfg.filings.year


def minimal(**overrides) -> dict:
    base = {
        "name": "t",
        "filings": {"year": 2024},
        "methods": [{"name": "tfidf", "kind": "tfidf"}],
    }
    base.update(overrides)
    return base


def test_minimal_defaults():
    cfg = Config.model_validate(minimal())
    assert cfg.universe.source == "sp500_wikipedia"
    assert cfg.filings.sections == ["business"]
    assert cfg.sections_dir == Path("data/sections/2024")
    assert cfg.method_dir("tfidf") == Path("data/runs/t/methods/tfidf")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"methods": [{"name": "a", "kind": "tfidf"}, {"name": "a", "kind": "tfidf"}]}, "중복"),
        (
            {"methods": [{"name": "r", "kind": "tfidf", "section": "risk_factors"}]},
            "filings.sections",
        ),
        ({"methods": [{"name": "s", "kind": "sbert"}]}, "model"),
        ({"methods": [{"name": "bad name", "kind": "tfidf"}]}, "method 이름"),
        ({"methods": [{"name": "t", "kind": "tfidf", "doc_tokens": 512}]}, "doc_tokens"),
        ({"universe": {"source": "csv"}}, "universe.path"),
        ({"filings": {"year": 2024, "sectoins": ["business"]}}, "sectoins"),
    ],
)
def test_invalid_configs(overrides, message):
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(minimal(**overrides))
