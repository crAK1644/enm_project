"""Data-pipeline tests for build_player_database and the pool selectors.

Guards the contracts CLAUDE.md calls out: the >=450-minute floor, the derived
feature columns, the canonical role column, and the slot/role pool selection
with broad-position fallback.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcdm.data_processor import (
    build_player_database,
    get_position_players,
    get_role_players,
    normalize_tm_position,
)

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DB = build_player_database(PARENT)

COMPUTED_COLUMNS = [
    "goals_per_xg",
    "expected_goal_involvements_per90",
    "expected_goals_conceded_per90",
    "save_percentage",
    "goals_prevented_per90",
    "bps_per90",
]
CANONICAL_ROLES = {"GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST"}


def test_minutes_floor_is_respected():
    assert (_DB["minutes"] >= 450).all(), "players below the 450-minute floor leaked in"
    assert len(_DB) > 0


def test_raising_minutes_floor_shrinks_the_pool():
    strict = build_player_database(PARENT, min_minutes=1500)
    assert len(strict) < len(_DB)
    assert (strict["minutes"] >= 1500).all()


def test_computed_feature_columns_present_and_numeric():
    for col in COMPUTED_COLUMNS:
        assert col in _DB.columns, f"missing computed column: {col}"
        assert np.isfinite(_DB[col].to_numpy(dtype=float)).all(), f"{col} has non-finite values"


def test_goals_per_xg_is_guarded_against_tiny_xg():
    """goals_per_xg must stay finite even when xG is ~0 (guard at xG<=0.1)."""
    vals = _DB["goals_per_xg"].to_numpy(dtype=float)
    assert np.isfinite(vals).all()
    assert (vals >= 0).all()


def test_save_percentage_in_unit_range():
    sp = _DB["save_percentage"].to_numpy(dtype=float)
    assert np.isfinite(sp).all()
    assert ((sp >= 0) & (sp <= 1)).all(), "save_percentage outside [0, 1]"


def test_role_column_is_canonical():
    roles = set(_DB["role"].dropna().unique())
    extra = roles - CANONICAL_ROLES
    assert not extra, f"non-canonical roles in data: {extra}"


def test_broad_position_column_values():
    assert set(_DB["position"].unique()) <= {
        "Forward", "Midfielder", "Defender", "Goalkeeper"
    }


def test_get_position_players_filters_by_broad_position():
    fwds = get_position_players(_DB, "Forward")
    assert len(fwds) > 0
    assert (fwds["position"] == "Forward").all()


def test_get_role_players_respects_pool():
    gks = get_role_players(_DB, ["GK"], broad_fallback="Goalkeeper")
    assert len(gks) > 0
    # Everyone returned should be a keeper either by role or broad fallback.
    assert ((gks["role"] == "GK") | (gks["position"] == "Goalkeeper")).all()


def test_get_role_players_broad_fallback_recovers_unlabeled_players():
    """The broad-position fallback must include players whose specific role is
    unknown (``role_from_fallback``) but whose broad position matches the slot.

    Built on a synthetic frame: with every current player now carrying a
    Transfermarkt main_position, the real DB has no ``role_from_fallback`` rows,
    so this exercises the mechanism directly rather than relying on live data.
    """
    synth = pd.DataFrame({
        "id": [1, 2, 3],
        "role": ["CM", "RM", "ST"],
        "position": ["Midfielder", "Midfielder", "Forward"],
        "role_from_fallback": [False, True, False],
    })
    # Pool asks for RM only; player 2 has role RM (direct match), player 1 (CM,
    # not in pool) is excluded. Now ask a pool with NO direct matches but use the
    # broad fallback to recover the unlabeled midfielder.
    synth.loc[1, "role"] = "CM"  # make player 2 a fallback-tagged midfielder, role not in pool
    recovered = get_role_players(synth, ["LM"], broad_fallback="Midfielder")
    assert (recovered["id"] == 2).any(), "fallback should recover the unlabeled midfielder"
    assert 3 not in recovered["id"].values, "a forward must not be recovered for a midfield slot"


def test_normalize_tm_position_maps_to_roles():
    assert normalize_tm_position("Centre-Forward") == "ST"
    assert normalize_tm_position("Goalkeeper") == "GK"
    # Unknown / empty inputs must not crash.
    assert normalize_tm_position("") is not None or normalize_tm_position("") is None


def test_market_value_column_present_and_nonnegative():
    assert "market_value_eur_m" in _DB.columns
    mv = _DB["market_value_eur_m"].fillna(0).to_numpy(dtype=float)
    assert (mv >= 0).all()


def test_no_zero_value_players():
    """Players with no Transfermarkt match (value 0 / no current PL squad) are
    dropped in build_player_database — a €0 value would otherwise read as 'free'
    to the optimizer. Every remaining player must have a positive market value."""
    mv = _DB["market_value_eur_m"].fillna(0).to_numpy(dtype=float)
    assert (mv > 0).all(), (
        "found players with a non-positive market value: "
        + ", ".join(_DB.loc[_DB["market_value_eur_m"].fillna(0) <= 0, "display_name"].astype(str))
    )


def test_ids_are_unique():
    assert _DB["id"].is_unique, "duplicate player ids in the database"
