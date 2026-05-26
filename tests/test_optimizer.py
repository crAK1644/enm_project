"""Unit tests for the squad optimizer.

These tests exercise the constraints (one-per-slot, no double-assignment,
budget window) and the failure modes (empty pool, locked overspend) without
touching FPL/Transfermarkt data.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

# Make `mcdm` importable when running pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcdm.optimizer import optimize_squad


def _df(rows):
    """Tiny builder so test cases stay readable."""
    return pd.DataFrame(rows, columns=["id", "name", "price", "score"])


# ─────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────

def test_picks_one_player_per_slot():
    cands = {
        "GK":  _df([("1", "Cheap GK", 5, 0.6), ("2", "Elite GK", 50, 0.95)]),
        "ST":  _df([("3", "Budget ST", 15, 0.4), ("4", "Star ST", 120, 1.0)]),
    }
    r = optimize_squad(cands, budget_max=200)
    assert r["status"] == "optimal"
    assert set(r["picks"].keys()) == {"GK", "ST"}
    # Highest-score player chosen in each slot when budget is loose
    assert r["picks"]["GK"]["id"] == "2"
    assert r["picks"]["ST"]["id"] == "4"


def test_respects_hard_budget_cap():
    cands = {
        "GK":  _df([("1", "Cheap GK", 5, 0.6), ("2", "Elite GK", 50, 0.95)]),
        "ST":  _df([("3", "Budget ST", 15, 0.4), ("4", "Star ST", 120, 1.0)]),
    }
    r = optimize_squad(cands, budget_max=30)  # can't afford Elite + Star
    assert r["status"] == "optimal"
    assert r["total_cost"] <= 30.0 + 1e-6


def test_respects_minimum_spend_floor():
    cands = {
        "GK":  _df([("1", "Cheap GK", 5, 0.6), ("2", "Elite GK", 50, 0.95)]),
        "ST":  _df([("3", "Budget ST", 15, 0.4), ("4", "Star ST", 120, 1.0)]),
    }
    r = optimize_squad(cands, budget_min=100, budget_max=200)
    assert r["status"] == "optimal"
    assert r["total_cost"] >= 100.0 - 1e-6


def test_higher_budget_never_lowers_score():
    cands = {
        "GK": _df([("1", "Cheap", 5, 0.5), ("2", "Mid", 30, 0.8), ("3", "Top", 80, 1.0)]),
        "ST": _df([("4", "Cheap", 5, 0.5), ("5", "Mid", 30, 0.8), ("6", "Top", 80, 1.0)]),
    }
    scores = []
    for budget in (50, 100, 200):
        r = optimize_squad(cands, budget_max=budget)
        assert r["status"] == "optimal"
        scores.append(r["total_score"])
    # Monotone non-decreasing in budget.
    assert scores[0] <= scores[1] <= scores[2]


def test_no_player_assigned_to_two_slots():
    # Same player ID eligible for both CM and CAM. Optimizer must not pick them
    # in both slots even if their score is highest in each.
    shared = ("99", "Versatile", 40, 1.0)
    other_low = ("1", "Backup", 20, 0.3)
    other_low2 = ("2", "Backup2", 20, 0.3)
    cands = {
        "CM":  _df([shared, other_low]),
        "CAM": _df([shared, other_low2]),
    }
    r = optimize_squad(cands, budget_max=500)
    assert r["status"] == "optimal"
    assigned_ids = [p["id"] for p in r["picks"].values()]
    assert len(set(assigned_ids)) == len(assigned_ids), "Same ID used in two slots"


# ─────────────────────────────────────────────────────────────
# Locked players
# ─────────────────────────────────────────────────────────────

def test_locked_slot_kept_in_picks_and_counts_toward_budget():
    cands = {
        "ST": _df([("3", "Budget ST", 15, 0.4), ("4", "Star ST", 120, 1.0)]),
    }
    locked = {"GK": {"id": "99", "name": "Locked Keeper", "price": 40}}
    r = optimize_squad(cands, locked=locked, budget_max=200)
    assert r["status"] == "optimal"
    assert "GK" in r["picks"]
    assert r["picks"]["GK"]["locked"] is True
    # Locked cost (40) + chosen ST should fit under 200
    assert r["total_cost"] <= 200.0 + 1e-6
    assert r["total_cost"] >= 40.0  # at least the locked player


def test_locked_overspend_is_infeasible():
    cands = {"ST": _df([("1", "Cheap", 5, 0.5)])}
    locked = {"GK": {"id": "99", "name": "Expensive", "price": 300}}
    r = optimize_squad(cands, locked=locked, budget_max=100)
    assert r["status"] == "infeasible"
    assert "locked" in r["message"].lower()


def test_locked_player_excluded_from_other_slot_pools():
    # The same ID appears as a candidate for ST but is locked to a different slot.
    # The optimizer must not also pick them for ST.
    cands = {"ST": _df([("99", "Locked Star", 100, 1.0), ("4", "Other", 20, 0.4)])}
    locked = {"GK": {"id": "99", "name": "Locked Star", "price": 100}}
    r = optimize_squad(cands, locked=locked, budget_max=500)
    assert r["status"] == "optimal"
    assert r["picks"]["ST"]["id"] == "4"  # the other one, not the locked ID


# ─────────────────────────────────────────────────────────────
# Failure modes
# ─────────────────────────────────────────────────────────────

def test_empty_candidate_slot_is_infeasible():
    cands = {
        "GK":  _df([]),
        "ST":  _df([("4", "Star ST", 120, 1.0)]),
    }
    r = optimize_squad(cands, budget_max=500)
    assert r["status"] == "infeasible"
    assert "GK" in r["message"]


def test_budget_below_cheapest_xi_is_infeasible():
    cands = {
        "GK": _df([("1", "GK", 50, 1.0)]),
        "ST": _df([("2", "ST", 50, 1.0)]),
    }
    r = optimize_squad(cands, budget_max=30)
    assert r["status"] == "infeasible"


def test_normalisation_does_not_break_tied_scores():
    # All candidates in this slot have identical scores → normalisation must
    # not divide by zero. Optimizer should still return optimal.
    cands = {
        "GK": _df([("1", "A", 10, 0.5), ("2", "B", 20, 0.5), ("3", "C", 30, 0.5)]),
    }
    r = optimize_squad(cands, budget_max=100)
    assert r["status"] == "optimal"
    # With all scores equal, the cheapest player should win on no other criterion
    # — but the LP is free to pick any; just assert one was picked and price is sane.
    pick = r["picks"]["GK"]
    assert pick["id"] in {"1", "2", "3"}
    assert pick["score_norm"] == pytest.approx(1.0)  # ties → all normalize to 1


def test_normalisation_is_per_slot_not_global():
    # Slot A scores in [10, 12]; slot B scores in [0.1, 0.2]. Without per-slot
    # normalisation, slot A would dominate. With it, both contribute equally.
    cands = {
        "A": _df([("a1", "Lo", 10, 10.0), ("a2", "Hi", 50, 12.0)]),
        "B": _df([("b1", "Lo", 10, 0.10), ("b2", "Hi", 50, 0.20)]),
    }
    r = optimize_squad(cands, budget_max=500)
    assert r["status"] == "optimal"
    # Both slots should pick the higher-scoring player when budget is loose.
    assert r["picks"]["A"]["id"] == "a2"
    assert r["picks"]["B"]["id"] == "b2"
    # Both contribute equally to the objective after per-slot normalisation.
    assert r["picks"]["A"]["score_norm"] == pytest.approx(1.0)
    assert r["picks"]["B"]["score_norm"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────
# Result shape
# ─────────────────────────────────────────────────────────────

def test_result_shape_optimal():
    cands = {"GK": _df([("1", "X", 10, 1.0)])}
    r = optimize_squad(cands, budget_max=100)
    assert {"status", "message", "picks", "total_cost", "total_score"} <= set(r.keys())
    pick = r["picks"]["GK"]
    assert {"id", "name", "price", "score_norm"} <= set(pick.keys())


def test_result_shape_infeasible():
    r = optimize_squad({"GK": _df([])}, budget_max=100)
    assert r["status"] == "infeasible"
    assert r["picks"] == {}
    assert r["total_cost"] == 0.0
    assert r["total_score"] == 0.0
