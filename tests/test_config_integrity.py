"""Configuration-integrity tests for criteria + formations.

These catch the cheap-but-deadly mistakes: a criterion pointing at a column
that doesn't exist in the player database, a benefit/cost flag that isn't ±1,
a new formation slot nobody mapped to a role, a role split into CDM6/CDM8
against the project's numbering rule, or a yellow/red-card criterion sneaking
back in. They need no math — just structural consistency.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcdm.data_processor import build_player_database
from mcdm.criteria import (
    POSITION_CRITERIA,
    ROLE_CRITERIA,
    SLOT_TO_ROLE,
    FORMATIONS,
)

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DB = build_player_database(PARENT)

CANONICAL_ROLES = {"GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "ST"}

ALL_CRITERIA_SETS = list(POSITION_CRITERIA.items()) + list(ROLE_CRITERIA.items())


def test_every_criterion_column_exists_in_database():
    missing = []
    for key, cfg in ALL_CRITERIA_SETS:
        for crit_name, spec in cfg.items():
            col = spec["column"]
            if col not in _DB.columns:
                missing.append(f"{key}.{crit_name} -> '{col}'")
    assert not missing, "Criteria reference non-existent columns:\n" + "\n".join(missing)


def test_every_criterion_has_valid_type_and_label():
    bad = []
    for key, cfg in ALL_CRITERIA_SETS:
        for crit_name, spec in cfg.items():
            if spec.get("type") not in (1, -1):
                bad.append(f"{key}.{crit_name} type={spec.get('type')!r} (must be 1 or -1)")
            if not spec.get("label"):
                bad.append(f"{key}.{crit_name} missing label")
            if "column" not in spec:
                bad.append(f"{key}.{crit_name} missing column")
    assert not bad, "Invalid criterion specs:\n" + "\n".join(bad)


def test_no_card_criteria_anywhere():
    """Yellow/red cards were removed by user request and must stay out."""
    offenders = []
    for key, cfg in ALL_CRITERIA_SETS:
        for crit_name, spec in cfg.items():
            blob = f"{crit_name} {spec.get('column','')} {spec.get('label','')}".lower()
            if "yellow" in blob or "red_card" in blob or "red card" in blob or "card" in blob:
                offenders.append(f"{key}.{crit_name}")
    assert not offenders, "Card-based criteria are forbidden: " + ", ".join(offenders)


def test_role_criteria_keys_are_canonical():
    """Roles must use bare traditional names — never CDM6/CDM8 style splits."""
    extra = set(ROLE_CRITERIA) - CANONICAL_ROLES
    assert not extra, f"Non-canonical role keys: {extra}"
    for role in ROLE_CRITERIA:
        assert not re.search(r"\d", role), f"Role '{role}' must not contain digits"


def test_slot_to_role_targets_exist():
    bad = []
    for slot, info in SLOT_TO_ROLE.items():
        if info["role"] not in ROLE_CRITERIA:
            bad.append(f"slot {slot}: role '{info['role']}' not in ROLE_CRITERIA")
        if info["broad"] not in POSITION_CRITERIA:
            bad.append(f"slot {slot}: broad '{info['broad']}' not in POSITION_CRITERIA")
        if not info["pool"]:
            bad.append(f"slot {slot}: empty role pool")
        for role in info["pool"]:
            if role not in CANONICAL_ROLES:
                bad.append(f"slot {slot}: pool role '{role}' not canonical")
    assert not bad, "SLOT_TO_ROLE inconsistencies:\n" + "\n".join(bad)


def test_every_formation_slot_is_mapped():
    """Each slot in every formation must resolve via SLOT_TO_ROLE (after the
    trailing-digit suffix is stripped the way app.py does)."""
    unmapped = []
    for formation, slots in FORMATIONS.items():
        for slot in slots:
            if slot not in SLOT_TO_ROLE:
                unmapped.append(f"{formation}.{slot}")
    assert not unmapped, "Formation slots missing from SLOT_TO_ROLE:\n" + "\n".join(unmapped)


def test_every_formation_has_eleven_slots_with_coords():
    bad = []
    for formation, slots in FORMATIONS.items():
        if len(slots) != 11:
            bad.append(f"{formation}: {len(slots)} slots (expected 11)")
        for slot, spec in slots.items():
            if not all(k in spec for k in ("x", "y", "pos")):
                bad.append(f"{formation}.{slot}: missing x/y/pos")
                continue
            if not (0 <= spec["x"] <= 100 and 0 <= spec["y"] <= 100):
                bad.append(f"{formation}.{slot}: coords out of 0-100 range")
    assert not bad, "Formation layout problems:\n" + "\n".join(bad)


def test_exactly_one_goalkeeper_per_formation():
    for formation, slots in FORMATIONS.items():
        gks = [s for s in slots if SLOT_TO_ROLE.get(s, {}).get("role") == "GK"]
        assert len(gks) == 1, f"{formation} has {len(gks)} GK slots"
