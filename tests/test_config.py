from pathlib import Path

import pytest
from pydantic import ValidationError

from tenksim.config import Config, load_config
from tenksim.relations.names import load_aliases

ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [p for p in sorted((ROOT / "configs").glob("*.yaml")) if p.name != "aliases.yaml"]


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_repo_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.methods and cfg.filings.year
    if cfg.relations:
        load_aliases(ROOT / cfg.relations.aliases)


def test_relations_similarity_must_exist():
    with pytest.raises(ValidationError, match="relations.similarity"):
        Config.model_validate(minimal(relations={"similarity": "nope"}))


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


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ensembles": [{"name": "e", "members": ["tfidf"]}]}, "2개 이상"),
        ({"ensembles": [{"name": "e", "members": ["tfidf", "nope"]}]}, "없는 변형"),
        ({"ensembles": [{"name": "e", "members": ["tfidf", "tfidf+center"]}]}, "없는 변형"),
        ({"ensembles": [{"name": "tfidf", "members": ["tfidf", "s"]}]}, "겹칩니다"),
        ({"evaluation": {"k": [1, 3], "primary_k": 5}}, "primary_k"),
        ({"evaluation": {"baseline": "nope"}}, "baseline"),
    ],
)
def test_invalid_ensemble_and_evaluation(overrides, message):
    cfg = minimal(**overrides)
    cfg["methods"] = [
        {"name": "tfidf", "kind": "tfidf"},
        {"name": "s", "kind": "sbert", "model": "m"},
    ]
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(cfg)


def test_variant_names_and_main_k():
    cfg = Config.model_validate(
        minimal(
            methods=[
                {"name": "tfidf", "kind": "tfidf"},
                {"name": "s", "kind": "sbert", "model": "m"},
            ],
            ensembles=[{"name": "e", "members": ["tfidf", "s+center"]}],
            evaluation={"k": [1, 3]},
        )
    )
    assert cfg.variant_names == ["tfidf", "s", "s+center"]
    assert cfg.baseline_name == "tfidf" and cfg.evaluation.main_k == 3
    assert cfg.ensemble("e").members == ["tfidf", "s+center"]
