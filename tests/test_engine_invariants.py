"""Property/invariant tests for the MCDM engine over real data.

Where ``test_engine_integration`` just checks methods *run*, these assert the
mathematical contracts every ranking must satisfy: finite scores, ranks that
form a clean permutation, score/rank agreement, the VIKOR ``1 - Q`` inversion,
normalised weights, determinism, and correct handling of custom weights and
degenerate pools.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcdm.data_processor import build_player_database, get_role_players
from mcdm.criteria import SLOT_TO_ROLE, ROLE_CRITERIA
from mcdm.engine import rank_players, SUPPORTED_METHODS

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METHODS = ["promethee", "vikor", "topsis", "saw",
           "wp", "waspas", "codas", "borda_consensus"]
WEIGHTINGS = ["critic", "entropy"]

# One representative (role, criteria, players) tuple per canonical role.
_DB = build_player_database(PARENT)


def _role_cases():
    seen = set()
    for info in SLOT_TO_ROLE.values():
        role = info["role"]
        if role in seen or role not in ROLE_CRITERIA:
            continue
        seen.add(role)
        players = get_role_players(_DB, info["pool"], info["broad"])
        if len(players) >= 2:
            yield role, ROLE_CRITERIA[role], players


ROLE_CASES = list(_role_cases())


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_scores_finite_and_ranks_are_a_permutation(method, weighting):
    for role, cfg, players in ROLE_CASES:
        ranked, obj_w, applied_w = rank_players(
            players, cfg, method=method, weighting=weighting
        )
        n = len(ranked)
        scores = ranked["score"].to_numpy(dtype=float)
        assert np.all(np.isfinite(scores)), f"{role}/{method}/{weighting} non-finite score"

        ranks = ranked["rank"].to_numpy(dtype=int)
        assert sorted(ranks.tolist()) == list(range(1, n + 1)), (
            f"{role}/{method}/{weighting} ranks not a 1..n permutation"
        )


@pytest.mark.parametrize("method", METHODS)
def test_score_order_matches_rank_order(method):
    """Result is returned rank-sorted; scores must be non-increasing down it."""
    for role, cfg, players in ROLE_CASES:
        ranked, _, _ = rank_players(players, cfg, method=method, weighting="critic")
        assert list(ranked["rank"]) == sorted(ranked["rank"]), f"{role} not rank-sorted"
        scores = ranked["score"].to_numpy(dtype=float)
        assert np.all(np.diff(scores) <= 1e-9), (
            f"{role}/{method}: score not monotonically non-increasing with rank"
        )
        # Rank 1 owns the maximum score.
        top = ranked.iloc[0]
        assert top["rank"] == 1
        assert np.isclose(top["score"], scores.max())


def test_vikor_score_is_one_minus_q():
    """The app-wide convention: VIKOR exposes score = 1 - Q (higher = better)."""
    for role, cfg, players in ROLE_CASES:
        ranked, _, _ = rank_players(players, cfg, method="vikor", weighting="critic")
        assert "q_value" in ranked.columns
        assert np.allclose(
            ranked["score"].to_numpy(float),
            1.0 - ranked["q_value"].to_numpy(float),
        ), f"{role}: VIKOR score != 1 - Q"


@pytest.mark.parametrize("weighting", WEIGHTINGS)
def test_objective_and_applied_weights_sum_to_one(weighting):
    for role, cfg, players in ROLE_CASES:
        _, obj_w, applied_w = rank_players(
            players, cfg, method="topsis", weighting=weighting
        )
        assert obj_w and applied_w
        assert abs(sum(obj_w.values()) - 1.0) < 1e-6, f"{role}: objective weights !~ 1"
        assert abs(sum(applied_w.values()) - 1.0) < 1e-6, f"{role}: applied weights !~ 1"
        assert all(w >= -1e-9 for w in obj_w.values()), f"{role}: negative objective weight"


def test_custom_weights_are_honoured_and_renormalised():
    role, cfg, players = ROLE_CASES[0]
    names = list(cfg.keys())
    # Heavily favour the first criterion; leave the rest equal.
    custom = {names[0]: 10.0}
    for n in names[1:]:
        custom[n] = 1.0
    _, _, applied = rank_players(
        players, cfg, method="saw", custom_weights=custom, weighting="critic"
    )
    assert abs(sum(applied.values()) - 1.0) < 1e-6
    # Dominant criterion must end up with the largest applied weight.
    assert applied[names[0]] == max(applied.values())


def test_engine_is_deterministic():
    role, cfg, players = ROLE_CASES[0]
    a, _, _ = rank_players(players, cfg, method="promethee", weighting="critic")
    b, _, _ = rank_players(players, cfg, method="promethee", weighting="critic")
    assert np.allclose(a["score"].to_numpy(float), b["score"].to_numpy(float))
    assert list(a["rank"]) == list(b["rank"])


def test_single_player_pool_degenerate_case():
    role, cfg, players = ROLE_CASES[0]
    one = players.iloc[:1].copy()
    ranked, obj_w, applied_w = rank_players(one, cfg, method="topsis")
    assert len(ranked) == 1
    assert int(ranked.iloc[0]["rank"]) == 1
    assert float(ranked.iloc[0]["score"]) == 1.0


def test_unknown_method_raises():
    role, cfg, players = ROLE_CASES[0]
    with pytest.raises(ValueError):
        rank_players(players, cfg, method="not-a-real-method")


def test_supported_methods_constant_matches_tested_set():
    assert set(METHODS) == set(SUPPORTED_METHODS)
