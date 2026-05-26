"""Linear-programming squad builder.

Given per-slot candidate frames with raw MCDM scores, pick one player per slot
that maximises the sum of per-slot min-max-normalised scores, subject to a
total-spend window and the constraint that no player can fill two slots.

The caller is responsible for:
  - resolving each slot's role pool and computing the MCDM scores
  - excluding players already locked to other slots from the candidate frames

This module is intentionally pure: it knows nothing about Dash, formations,
or the FPL/Transfermarkt data shapes beyond `id`, `name`, `price`, `score`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi == lo:
        return np.ones_like(scores, dtype=float)
    return (scores - lo) / (hi - lo)


def optimize_squad(
    slot_candidates: dict[str, pd.DataFrame],
    locked: dict[str, dict] | None = None,
    budget_min: float = 0.0,
    budget_max: float = float("inf"),
) -> dict:
    """Solve the squad-selection ILP.

    Parameters
    ----------
    slot_candidates : dict[slot -> DataFrame]
        Each frame must contain columns ``id``, ``name``, ``price``, ``score``.
        ``score`` is the raw MCDM score; it is min-max normalised per slot
        before optimisation so methods on different scales mix cleanly.
    locked : dict[slot -> {"id", "name", "price"}]
        Slots already filled by the user. These contribute to budget and
        ``picks`` but are not re-optimised. Their player IDs are blocked from
        other slots' candidate sets.
    budget_min, budget_max : float
        Total-spend window in € millions, inclusive.

    Returns
    -------
    dict with keys:
        status      : "optimal" | "infeasible"
        message     : human-readable explanation
        picks       : dict[slot -> {"id","name","price","score_norm","locked"?}]
        total_cost  : float
        total_score : float (sum of normalised scores; locked slots score 1.0)
    """
    locked = locked or {}

    def fail(msg: str) -> dict:
        return {
            "status": "infeasible",
            "message": msg,
            "picks": {},
            "total_cost": 0.0,
            "total_score": 0.0,
        }

    empty = [s for s, df in slot_candidates.items() if df is None or len(df) == 0]
    if empty:
        return fail(f"No eligible candidates for: {', '.join(empty)}")

    locked_cost = sum(float(p.get("price", 0) or 0) for p in locked.values())
    if locked_cost > budget_max:
        return fail(
            f"Locked players already cost €{locked_cost:.1f}m, above max €{budget_max:.1f}m"
        )

    locked_ids = {str(p["id"]) for p in locked.values()}

    normalized: dict[str, pd.DataFrame] = {}
    for slot, df in slot_candidates.items():
        df = df[~df["id"].astype(str).isin(locked_ids)].copy()
        if len(df) == 0:
            return fail(f"All candidates for {slot} are already assigned elsewhere")
        df["score_norm"] = _normalize(df["score"].astype(float).values)
        normalized[slot] = df.reset_index(drop=True)

    prob = pulp.LpProblem("squad", pulp.LpMaximize)

    x: dict[tuple[str, str], pulp.LpVariable] = {}
    for slot, df in normalized.items():
        safe_slot = slot.replace(" ", "_")
        for _, row in df.iterrows():
            pid = str(row["id"])
            x[(slot, pid)] = pulp.LpVariable(f"x_{safe_slot}_{pid}", cat="Binary")

    prob += pulp.lpSum(
        x[(slot, str(row["id"]))] * float(row["score_norm"])
        for slot, df in normalized.items()
        for _, row in df.iterrows()
    )

    for slot, df in normalized.items():
        prob += (
            pulp.lpSum(x[(slot, str(row["id"]))] for _, row in df.iterrows()) == 1,
            f"one_{slot.replace(' ', '_')}",
        )

    pid_to_slots: dict[str, list[str]] = {}
    for slot, df in normalized.items():
        for pid in df["id"].astype(str):
            pid_to_slots.setdefault(pid, []).append(slot)
    for pid, slots in pid_to_slots.items():
        if len(slots) > 1:
            prob += (
                pulp.lpSum(x[(slot, pid)] for slot in slots) <= 1,
                f"unique_{pid}",
            )

    spend = pulp.lpSum(
        x[(slot, str(row["id"]))] * float(row["price"])
        for slot, df in normalized.items()
        for _, row in df.iterrows()
    )
    if budget_max < float("inf"):
        prob += spend + locked_cost <= budget_max, "budget_max"
    if budget_min > 0:
        prob += spend + locked_cost >= budget_min, "budget_min"

    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]
    if status != "Optimal":
        return fail(
            f"No feasible squad in budget [€{budget_min:.1f}m, €{budget_max:.1f}m]"
        )

    picks: dict[str, dict] = {}
    total_cost = 0.0
    total_score = 0.0
    for slot, df in normalized.items():
        for _, row in df.iterrows():
            pid = str(row["id"])
            v = x[(slot, pid)].value()
            if v is not None and v > 0.5:
                price = float(row["price"])
                score_norm = float(row["score_norm"])
                picks[slot] = {
                    "id": pid,
                    "name": row["name"],
                    "team": str(row.get("team", "") or ""),
                    "price": price,
                    "score_norm": score_norm,
                }
                total_cost += price
                total_score += score_norm
                break

    for slot, p in locked.items():
        price = float(p.get("price", 0) or 0)
        picks[slot] = {
            "id": str(p["id"]),
            "name": p.get("name", "?"),
            "team": str(p.get("team", "") or ""),
            "price": price,
            "score_norm": 1.0,
            "locked": True,
        }
        total_cost += price
        total_score += 1.0

    return {
        "status": "optimal",
        "message": "OK",
        "picks": picks,
        "total_cost": total_cost,
        "total_score": total_score,
    }
