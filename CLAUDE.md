# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install deps (Python 3.10+)
pip install -r requirements.txt

# Run the Dash app — serves on http://localhost:8050
python app.py

# Refresh Transfermarkt market values + positions (writes data/market_values.csv).
# Run at the start of each transfer window.
python scraper/transfermarkt_scraper.py

# Sanity-check the scraper on just Arsenal (no CSV write, prints rows)
python scraper/transfermarkt_scraper.py --dry-run
```

There is no test suite, linter, or build step configured.

A local `.venv/` is checked in alongside the source; activate it before running if you want the pinned interpreter.

## Architecture

Football-Manager-style transfer-window decision support. Single-page Dash app backed by an MCDM engine that supports five ranking methods (PROMETHEE II, VIKOR, TOPSIS, WASPAS, CODAS) and two objective weighting schemes (CRITIC, Shannon Entropy). Big-picture split is **data prep → MCDM engine → Dash UI**, with criteria + role mapping sitting between them.

### Data flow

1. **`mcdm/data_processor.py`** loads three CSVs and produces the per-position decision matrices used by the engine:
   - `players.csv` — FPL roster (id, name, position, team)
   - `playerstats.csv` — FPL cumulative per-gameweek stats (large, ~7MB)
   - `data/market_values.csv` — Transfermarkt values + `main_position` produced by the scraper

   It computes per-90 normalized stats, joins market values, and filters to players with **≥ 450 minutes played** (≈ 5 full matches). Players below the threshold should never appear in rankings — check this filter before changing it.

   It also derives a canonical **role column** (GK/CB/LB/RB/CDM/CM/CAM/LM/RM/LW/RW/ST) via `TM_POSITION_TO_ROLE` from the scraped `main_position`, falling back to `BROAD_TO_ROLE` when a player's specific role is unknown. `get_role_players(db, role_pool, broad_fallback)` enforces the slot→role pool with broad-position fallback.

   **Computed feature columns** (not raw FPL stats) live here too — adding a new computed criterion means: add the column here AND register it in `criteria.py` with the correct `type` (1 = benefit, -1 = cost):
   - `goals_per_xg` — finishing efficiency, guarded for xG ≤ 0.1
   - `expected_goal_involvements_per90` — xG + xA
   - `expected_goals_conceded_per90` — defensive workload faced
   - `save_percentage` — saves / (saves + GC) for keepers
   - `goals_prevented_per90` — (xGC − GC)/min × 90, GK overperformance
   - `bps_per90` — Bonus Points System rate; rewards tackles/recoveries/key passes

2. **`mcdm/criteria.py`** holds four deliberately co-located concerns:
   - `POSITION_CRITERIA` — broad-position fallback sets (Forward/Midfielder/Defender/Goalkeeper). Used when a slot is unmapped or a player has no specific role tag.
   - `ROLE_CRITERIA` — 12 specific-role criteria sets keyed by canonical role code. Each criterion is `{column, type, label}` where `type` is 1 (benefit) or -1 (cost). Roles follow traditional football numbering: **CDM = 6 anchor, CM = 8 box-to-box, CAM = 10 playmaker** — never split into CDM6/CDM8.
   - `SLOT_TO_ROLE` — maps formation slot names (e.g. `CB1`, `CDM2`, `ST`) → `{role, pool, broad}`. `pool` is the list of canonical roles eligible for that slot; `broad` is the FPL position fallback. Trailing digits on slot names are display-only — strip with `re.sub(r'\d+$', '', slot)`.
   - `FORMATIONS` — six layouts (4-3-3, 4-4-2, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2) as `{slot: {x, y, pos}}` coordinate dicts. `pos` is the broad-position tag kept for backward compat; specific-role behavior is driven by `SLOT_TO_ROLE`.

   These four are co-located because changing a formation often means changing which slots exist, which determines which role applies, which depends on the slot→role mapping.

3. **`mcdm/engine.py`** implements:
   - **CRITIC** — objective weight derivation: std-dev × Σ(1 − correlation). Discrimination × non-redundancy.
   - **Shannon Entropy weighting** — alternative objective scheme; weight ∝ (1 − entropy of the normalized column).
   - **PROMETHEE II** — net outranking flow Φ; higher is better.
   - **VIKOR** — compromise ranking via S/R/Q; lower Q is better, **but the app inverts Q for display so higher = better in the UI for all five methods**. Preserve this inversion when touching ranking output.
   - **TOPSIS** — closeness coefficient to ideal vs. anti-ideal; 0–1, higher is better.
   - **WASPAS** — λ = 0.5 hybrid of WSM (additive) + WPM (multiplicative); penalizes weak-on-any-single-criterion profiles.
   - **CODAS** — Euclidean distance from anti-ideal with Taxicab tie-breaker.

   `rank_players(players, criteria_cfg, method, custom_weights, weighting)` returns `(ranked_df, objective_weights_dict, applied_weights_dict)`. `SUPPORTED_METHODS` / `SUPPORTED_WEIGHTINGS` constants gate valid inputs.

4. **`app.py`** is the entire Dash UI — layout and callbacks in one file. Callbacks own the cross-cutting state:
   - Selected formation, selected position node, selected method, selected weighting scheme, budget, slider weight overrides, selected player (for the radar/breakdown panel).
   - **Stable weights across unrelated changes**: in `update_rankings`, slider values are only rebalanced when a `weight-slider` is the actual trigger. `reset-weights-btn`, `store-selected-position`, and `weighting-selector` drop to fresh objective weights; budget / search / assignment changes pass current slider values through untouched. Breaking this causes weights to silently re-normalize every time the user touches an unrelated control.
   - **Assigned-player exclusion**: once a player is assigned to a slot, they must be filtered out of every other position's candidate list. This invariant lives in the callback layer (`update_rankings` filters by `assigned_ids`), not the engine.
   - **Phantom trigger guards**: both `select_player` and `assign_player` early-return when `all((n or 0) == 0 for n in player_clicks)` because pattern-matching components re-emit `n_clicks=0` on table re-render. Without these guards, the first row would auto-assign whenever the table refreshes.
   - **Method/weighting explanation panel** is a standalone panel below the main grid (always renders); content comes from `METHOD_EXPLANATIONS` / `WEIGHTING_EXPLANATIONS` dicts at module scope.
   - **Player infographic always renders** once a position is selected — `build_player_detail` falls back to the position-average radar when no player is selected, and overlays the player when one is.

5. **`scraper/transfermarkt_scraper.py`** is a standalone script (not imported by the app). It iterates all 20 Premier League team pages and extracts player name + `main_position` + market value row-by-row from the `td.posrela` cell's nested `inline-table`, then fuzzy-matches Transfermarkt names to `players.csv` via `SequenceMatcher` (threshold 0.6). Writes `data/market_values.csv` with `main_position` (consumed by `data_processor.normalize_tm_position`). The committed CSV is the source of truth at runtime; re-run only when you want fresh values.

### Conventions worth knowing

- **Cost vs. benefit criteria** are declared in `criteria.py`; the engine relies on the `type` flag for normalization direction. Adding a new criterion means updating both `data_processor.py` (to produce the column) and `criteria.py` (to register it with the correct direction).
- **Weights flow**: CRITIC or Entropy computes objective defaults → user slider overrides → engine consumes (engine normalizes internally). Sliders use `rebalance_weights_after_change` to preserve the touched slider and redistribute the rest; other triggers pass slider values through unchanged.
- **Role numbering** is traditional: **CDM = 6, CM = 8, CAM = 10** — bare names only, never CDM6/CDM8 splits. The user is strict about this.
- **Slot-name digits** (`CB1`, `CB2`, `CDM1`) are uniqueness suffixes for the formation graph; display strips trailing digits via `re.sub(r'\d+$', '', slot)`.
- **Yellow/Red cards are intentionally NOT used** as criteria anywhere — they were removed by user request.
- **Current per-role criteria highlights** (full lists live in `criteria.py`):
  - GK: Saves, Clean Sheets, Save %, Goals Prevented, xGC, Influence
  - CB: Tackles, CBI, Clean Sheets, Recoveries, Defensive Contribution, xGC, Influence
  - LB / RB: Tackles, CBI, Recoveries, Defensive Contribution, xA, Assists, Creativity
  - CDM: Tackles, CBI, Recoveries, Defensive Contribution, BPS, Influence
  - CM: Goals, Assists, xGI, Creativity, Tackles, Recoveries, Influence
  - CAM: xA, xG, Goals, Assists, Creativity, Shooting Threat
  - LM / RM: xA, Assists, Creativity, Shooting Threat, Tackles, Defensive Contribution
  - LW / RW: xG, xA, xGI, Goals, Goals/xG, Assists, Creativity, Shooting Threat
  - ST: xG, Goals, Goals/xG, xGI, Shooting Threat, Assists, Aerial Ability, Defensive Contribution
