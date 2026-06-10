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

# Run tests (pytest). ~80 tests across five files; all should pass.
python -m pytest tests/ -v
python -m pytest tests/test_optimizer.py -v                          # LP / ILP constraints
python -m pytest tests/test_engine_invariants.py -v                  # MCDM math contracts
python -m pytest tests/test_config_integrity.py -v                   # criteria/formation consistency
python -m pytest tests/test_data_processor.py -v                     # data pipeline + 450-min floor
python -m pytest tests/test_ui_callbacks.py -v                       # Dash callback layer
python -m pytest tests/test_optimizer.py::test_no_player_assigned_to_two_slots
```

There is no linter or build step configured. The test suite is `pytest`-based; `pulp` is a runtime dep (CBC solver ships with the wheel).

**Interpreter**: a local `.venv/` is checked in. On this machine there is no bare `python`/`python3` on PATH — invoke the pinned interpreter directly, e.g. `.venv/bin/python3 app.py` and `.venv/bin/python3 -m pytest tests/`.

**Test layers** (each catches a different class of regression):
- `test_optimizer.py` — synthetic ILP cases (one-per-slot, no double-assignment, budget window, infeasible reasons).
- `test_engine_invariants.py` — over real data, all 9 methods × 2 weightings: finite scores, ranks a clean `1..n` permutation, score↔rank agreement, **VIKOR `1−Q`** inversion, weights sum to 1, determinism.
- `test_config_integrity.py` — every criterion column exists in the DB, types are ±1, roles are canonical (no CDM6/CDM8), every formation slot maps via `SLOT_TO_ROLE`, no card criteria.
- `test_data_processor.py` — 450-min floor, computed columns finite/in-range, role/position values, pool selectors + broad fallback.
- `test_ui_callbacks.py` — static **callback arity guard** (every multi-output callback returns the declared count on every path) and a **layout-id guard** (every plain-string callback Input/State id exists in the static layout); plus functional sweeps that invoke callbacks with a mocked `callback_context`.

**Driving the live app in a browser**: `.claude/launch.json` defines an `enm-app` config (`.venv/bin/python3 app.py`, port 8050) for the preview tooling. Dash uses pattern-matching ids (e.g. `{"index":"ST","type":"position-node"}`); select them with `[id='{"index":"ST","type":"position-node"}']`. The budget `dcc.Slider` responds to `Home`/`End`/arrow keys on its `[role="slider"]` thumb; `search-input` has `debounce=True`, so commit it with a `change` event (or Enter), not just `input`.

## Architecture

Football-Manager-style transfer-window decision support. Single-page Dash app backed by an MCDM engine that supports eight ranking methods (PROMETHEE II, VIKOR, TOPSIS, SAW, WP, WASPAS, CODAS, Borda Consensus) and two objective weighting schemes (CRITIC, Shannon Entropy). AHP was removed by user request — weights come from CRITIC/Entropy, so AHP's synthesis-only role was redundant with SAW. A PuLP/CBC ILP optimizer turns the per-slot rankings into a budget-bounded XI. Big-picture split is **data prep → MCDM engine → Dash UI**, with criteria + role mapping sitting between them, and **optimizer** as a sibling of the engine driven by the UI.

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
   - **VIKOR** — compromise ranking via S/R/Q; lower Q is better, **but the app inverts Q (`1 − Q`) so higher = better in the UI across every method**. Preserve this inversion when touching ranking output.
   - **TOPSIS** — closeness coefficient to ideal vs. anti-ideal; 0–1, higher is better.
   - **SAW** — Simple Additive Weighting; linear sum after min-max normalisation.
   - **WP** — Weighted Product; multiplicative scoring, penalises weak-on-any criterion. Uses an `epsilon = 1e-5` offset on the matrix to avoid zero-base negative exponents.
   - **WASPAS** — λ = 0.5 hybrid of SAW + WP.
   - **CODAS** — Euclidean distance from anti-ideal with Taxicab tie-breaker.
   - **Borda Consensus** — aggregates ranks across the other seven base methods via `points = n − rank + 1`.

   `rank_players(players, criteria_cfg, method, custom_weights, weighting)` returns `(ranked_df, objective_weights_dict, applied_weights_dict)`. `SUPPORTED_METHODS` / `SUPPORTED_WEIGHTINGS` / `METHOD_ALIASES` constants gate valid inputs.

4. **`mcdm/optimizer.py`** is the LP-based squad builder. `optimize_squad(slot_candidates, locked, budget_min, budget_max)` consumes per-slot candidate frames (`id, name, team, price, score`) that the caller has already produced via `rank_players`, min-max-normalises `score` to `[0, 1]` **per slot** (so methods on different scales — PROMETHEE Φ ≈ 0.02 vs. VIKOR `1−Q` ≈ 0.4 — mix cleanly), then solves a PuLP/CBC ILP:
   - binary `x[slot, player_id]`
   - exactly one player per slot: `Σ_p x[slot,p] = 1`
   - no player in two slots: `Σ_slot x[slot,p] ≤ 1` for any `p` eligible in more than one slot pool (matters for CM ∩ CAM type overlaps)
   - `budget_min ≤ Σ price·x + locked_cost ≤ budget_max`
   - maximise `Σ score_norm · x`

   The optimizer is pure and Dash-agnostic — `app.py:_build_optimizer_inputs` is the bridge that resolves each slot's role pool, calls `rank_players`, and packs the candidate frames + locked dict. Returns `{status, message, picks, total_cost, total_score}`; `status="infeasible"` carries a user-readable reason (locked overspend, empty pool, budget too tight).

5. **`app.py`** is the entire Dash UI — layout and callbacks in one file. Callbacks own the cross-cutting state:
   - Selected formation, selected position node, selected method, selected weighting scheme, budget (max + min), slider weight overrides, selected player (for the radar/breakdown panel), optimizer-preview store.
   - **Stable weights across unrelated changes**: in `update_rankings`, slider values are only rebalanced when a `weight-slider` is the actual trigger. `reset-weights-btn`, `store-selected-position`, and `weighting-selector` drop to fresh objective weights; budget / search / assignment changes pass current slider values through untouched. Breaking this causes weights to silently re-normalize every time the user touches an unrelated control.
   - **Weight-slider feedback cycle — DO NOT rebuild `weights-container` on every run**: `update_rankings` takes the `weight-slider` values as an `Input` *and* outputs `weights-container` (which builds those sliders). Re-emitting the container recreates the sliders, whose value change re-fires the callback. That re-render is only warranted when the weights actually change — a position/weighting/reset change or a slider drag (`weight_triggers = fresh_triggers + ("weight-slider",)`). For every other trigger (**method switch**, budget, search, assignment, player selection) the callback returns `dash.no_update` for the three weight-related outputs (indices 1/4/5). Without this guard, rapid method switches stacked the cascade — each iteration recomputing every MCDM method (the per-method rank map below) — until the UI **froze**. Tests: `test_method_switch_does_not_rebuild_weight_sliders`, `test_weight_outputs_rebuild_on_position_and_slider`.
   - **Per-method rank map**: `update_rankings` ranks the pool under **all** supported methods (same weights) and stores `{pid: {method: rank}}` in `store-position-data` as `method_ranks`; `build_player_detail` renders it as "Rank by method" chips when a player is selected. This replaced the old single-alternate-method Δ "stability badge" column in the rankings table. It is the expensive part of the callback (~60–90 ms on the largest pool) — which is why the no-rebuild guard above matters.
   - **Assigned-player exclusion**: once a player is assigned to a slot, they must be filtered out of every other position's candidate list. This invariant lives in the callback layer (`update_rankings` filters by `assigned_ids`), not the engine. The optimizer enforces it independently via its own no-double-assignment ILP constraint.
   - **Phantom trigger guards**: both `select_player` and `assign_player` early-return when `all((n or 0) == 0 for n in player_clicks)` because pattern-matching components re-emit `n_clicks=0` on table re-render. Without these guards, the first row would auto-assign whenever the table refreshes.
   - **Method/weighting explanation panel** is a standalone panel below the main grid (always renders); content comes from `METHOD_EXPLANATIONS` / `WEIGHTING_EXPLANATIONS` dicts at module scope.
   - **Player infographic always renders** once a position is selected — `build_player_detail` falls back to the position-average radar when no player is selected, and overlays the player when one is.
   - **Optimizer callbacks**: `run_optimizer` (Fill Empty / Optimize XI buttons or formation change) writes to `store-optimizer-preview`; `render_optimizer_preview` renders only the panel *content* from that store; `apply_or_discard_preview` writes to `store-assigned-players` with `allow_duplicate=True` (the only other writer to that store is `assign_player`). "Fill Empty" mode locks user picks and reduces the ILP to empty slots; "Optimize XI" wipes the formation's picks before applying.
   - **Apply/Discard buttons are STATIC, not dynamic**: `apply-preview-btn` / `discard-preview-btn` live in the base `app.layout` inside `optimizer-actions` (they are Inputs to `apply_or_discard_preview`, so they must always exist — see the layout-id invariant below). `toggle_optimizer_actions` shows/hides that row from `store-optimizer-preview` and swaps the second button's label between "Discard" (optimal) and "Dismiss" (infeasible), hiding Apply in the infeasible case. Do **not** move these buttons back into `render_optimizer_preview`.

6. **`scraper/transfermarkt_scraper.py`** is a standalone script (not imported by the app). It iterates all 20 Premier League team pages and extracts player name + `main_position` + market value row-by-row from the `td.posrela` cell's nested `inline-table`, then fuzzy-matches Transfermarkt names to `players.csv` via `SequenceMatcher` (threshold 0.6). Writes `data/market_values.csv` with `main_position` (consumed by `data_processor.normalize_tm_position`). The committed CSV is the source of truth at runtime; re-run only when you want fresh values.

### Conventions worth knowing

- **Cost vs. benefit criteria** are declared in `criteria.py`; the engine relies on the `type` flag for normalization direction. Adding a new criterion means updating both `data_processor.py` (to produce the column) and `criteria.py` (to register it with the correct direction).
- **Weights flow**: CRITIC or Entropy computes objective defaults → user slider overrides → engine consumes (engine normalizes internally). Sliders use `rebalance_weights_after_change` to preserve the touched slider and redistribute the rest; other triggers pass slider values through unchanged.
- **Role numbering** is traditional: **CDM = 6, CM = 8, CAM = 10** — bare names only, never CDM6/CDM8 splits. The user is strict about this.
- **Slot-name digits** (`CB1`, `CB2`, `CDM1`) are uniqueness suffixes for the formation graph; display strips trailing digits via `re.sub(r'\d+$', '', slot)`.
- **Yellow/Red cards are intentionally NOT used** as criteria anywhere — they were removed by user request.
- **Dash 4.x renamed component CSS classes** — `dcc.Dropdown`, `dcc.Slider`, and `dcc.Input` no longer use react-select/rc-slider classes. Style with `.dash-dropdown-*` (trigger / content / option / clear / search-icon), `.dash-slider-*` (rail / track / thumb / mark / mark-outside-selection / tooltip), and `.dash-input-*` (container / element / stepper). The legacy `.Select-*` and `.rc-slider-*` rules in `assets/style.css` match nothing under Dash 4 and exist only as compatibility shims.
- **Optimizer score semantics**: the per-slot min-max normalisation lives inside `optimize_squad`; callers pass raw MCDM `score` (higher = better) and don't pre-normalise. Locked players get `score_norm = 1.0` in the result so they don't drag the total down.
- **Layout-id invariant**: any component used as a callback `Input`/`State` with a plain string id MUST exist in the static `app.layout`. `suppress_callback_exceptions=True` lets the app run, but referencing a *dynamically-rendered* component as an Input floods the browser console with "nonexistent object used in an Input" errors. Components built inside a render callback must be either outputs only, or addressed via pattern-matching (dict) ids — which are exempt. `tests/test_ui_callbacks.py::test_callback_input_ids_exist_in_layout` enforces this.
- **Callback output arity**: every return path of a multi-output callback must return exactly as many values as it declares `Output`s (e.g. `update_rankings` has 6; `handle_player_assignment_and_selection` has 10). `tests/test_ui_callbacks.py::test_callback_return_arity_matches_outputs` is a static guard against the easy mistake of an early-return path with the wrong count.
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
