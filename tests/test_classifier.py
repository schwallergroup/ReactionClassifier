"""Self-contained tests (no external data paths) — CI friendly."""
import pytest

from reactionclassifier import (
    ReactionClassifier,
    load_granularity,
    load_taxonomy,
    name_for,
    tier_path,
)


@pytest.fixture(scope="module")
def clf():
    return ReactionClassifier()


def test_taxonomy_loads():
    tax = load_taxonomy()
    assert len(tax) > 1000
    assert name_for("1.3.5.5")  # known class has a name


def test_tier_path():
    assert tier_path("1.3.5.5") == ["1.3", "1.3.5", "1.3.5.5"]
    assert tier_path(None) == []


def test_granularity_loads():
    g = load_granularity()
    assert "granularity_examples" in g and "granularity_examples_medchem" in g


def test_classify_amide_coupling(clf):
    out = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
    assert out["matched"] is True
    assert out["code"].startswith("2.1.2")          # N-acylation to amide
    assert out["name"]
    assert out["tier_path"][0] == "2.1"
    assert 0.0 <= out["confidence"] <= 1.0


def test_classify_snar(clf):
    out = clf.classify("OCC1CNC1.COc1cnc(Cl)cc1>>COc1cnc(N2CC(CO)C2)cc1")
    assert out["matched"] is True
    assert out["code"].startswith("1.3")            # N-arylation with Ar-X


def test_result_schema(clf):
    out = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
    assert set(out) == {"code", "name", "tier_path", "matched", "gate_code", "confidence"}


def test_abstention_on_nonsense(clf):
    # gate will guess a class, but no template should reproduce this product
    out = clf.classify("[Na+].[Cl-]>>[Na+].[Cl-]")
    assert out["matched"] is False
    assert out["code"] is None


def test_invalid_input_does_not_crash(clf):
    out = clf.classify("not a reaction")
    assert out["matched"] is False
