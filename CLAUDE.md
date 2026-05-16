# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install deps (Python 3.10+)
pip install -r requirements.txt

# Run the Dash app — serves on http://localhost:8050
python app.py

# Refresh Transfermarkt market values (writes data/market_values.csv).
# Run at the start of each transfer window.
python scraper/transfermarkt_scraper.py
```

There is no test suite, linter, or build step configured.

A local `.venv/` is checked in alongside the source; activate it before running if you want the pinned interpreter.

## Architecture

This is a single-page Dash application (Football-Manager-style transfer-window decision support) backed by three MCDM methods. The "big picture" split is **data prep → MCDM engine → Dash UI**, with criteria definitions sitting between them.

### Data flow

1. **`mcdm/data_processor.py`** loads three CSVs and produces the per-position decision matrices used by the engine:
   - `players.csv` — FPL roster (id, name, position, team)
   - `playerstats.csv` — FPL cumulative per-gameweek stats (large, ~7MB)
   - `data/market_values.csv` — Transfermarkt values produced by the scraper
   It computes per-90 normalized stats, joins market values, and filters to players with **≥ 450 minutes played** (≈ 5 full matches). Players that fall below the threshold should never appear in rankings — check this filter before changing it.

2. **`mcdm/criteria.py`** defines, per position (FWD/MID/DEF/GK):
   - Which columns are criteria, and whether each is **benefit** (higher better) or **cost** (lower better — yellow/red cards).
   - Formation layouts (4-3-3, 4-4-2, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2) — coordinates of position nodes on the pitch.
   These two concerns are deliberately co-located because changing a formation often means changing which positions exist, which determines which criteria set applies.

3. **`mcdm/engine.py`** implements:
   - **CRITIC** — objective weight derivation from the decision matrix (std-dev × Σ(1 − correlation)). Weights are recomputed on every position click; sliders override but the engine normalizes and re-ranks without snapping the slider value back.
   - **PROMETHEE II** — net outranking flow Φ; higher is better.
   - **VIKOR** — compromise ranking via S/R/Q; lower Q is better, **but the app inverts Q for display so higher = better in the UI for both methods**. Preserve this inversion when touching ranking output.

4. **`app.py`** is the entire Dash UI — layout and callbacks in one file (~36KB). Callbacks own the cross-cutting state:
   - Selected formation, selected position node, current method (PROMETHEE/VIKOR), budget, slider weight overrides.
   - **Assigned-player exclusion**: once a player is assigned to a slot, they must be filtered out of every other position's candidate list. This invariant lives in the callback layer, not the engine — be careful when refactoring ranking flow.

5. **`scraper/transfermarkt_scraper.py`** is a standalone script (not imported by the app). It iterates all 20 Premier League team pages and writes `data/market_values.csv`. The committed CSV is the source of truth at runtime; re-run only when you want fresh values.

### Conventions worth knowing

- **Cost vs. benefit criteria** are declared in `criteria.py`; the engine relies on that flag for normalization direction. Adding a new criterion means updating both the data-processor (to produce the column) and `criteria.py` (to register it with the correct direction).
- **Weights** flow: CRITIC computes defaults → user slider overrides → normalize to sum 1 → engine consumes. Don't bypass normalization.
- Position criteria (per README): FWD uses xG/Goals/Assists/Shooting Threat/Creativity/Defensive Contribution/Aerial/Yellows/Reds; MID uses xA/Goals/Assists/Shooting Threat/Creativity/Tackles/Interceptions/Yellows/Reds; DEF uses Goals/Tackles/Interceptions/Clean Sheets/Influence/Yellows/Reds; GK uses Saves/Clean Sheets/Goals Conceded/Influence/Yellows/Reds.
