# Transfer Window Manager

A Football-Manager-style decision-support app for evaluating Premier League players during the transfer window. Pick a formation, click a position on the pitch to see role-specific MCDM rankings, set a budget, and let the linear-programming optimizer assemble the best XI for you.

Built with **Python + Dash**. Rankings flow from an MCDM engine that supports eight ranking methods, two objective weighting schemes, and a Borda consensus aggregator. Squad selection is solved as an integer linear program via PuLP/CBC.

## What's in the box

- **Interactive pitch** — six formations (4-3-3, 4-4-2, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2). Each slot resolves to a specific role (GK/CB/LB/RB/CDM/CM/CAM/LM/RM/LW/RW/ST) with its own criteria set.
- **Eight MCDM methods** — PROMETHEE II, VIKOR, TOPSIS, SAW, WP, WASPAS, CODAS, plus a Borda-count Consensus that aggregates them.
- **Two objective weighting schemes** — CRITIC (variance × dis-correlation) and Shannon Entropy. Sliders let you override either.
- **Rank by method** — a selected player's profile shows their rank under every MCDM method, so you can see who's robust vs. method-sensitive.
- **LP-based squad builder** — set a `Min`/`Max` spend window, click **Optimize XI** for a from-scratch best XI, or **Fill Empty** to keep your manual picks and solve the rest. Preview before applying.
- **Transfermarkt market values** — committed in `data/market_values.csv`; refresh per transfer window with the scraper.

## Quick start

Python 3.10+ required.

```bash
pip install -r requirements.txt
python app.py                          # → http://localhost:8050
```

Refresh market values (run at the start of each transfer window):

```bash
python scraper/transfermarkt_scraper.py
python scraper/transfermarkt_scraper.py --dry-run    # just Arsenal, no write
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

A pinned `.venv/` is checked in; activate it if you want the exact interpreter the project was developed on.

## How it works

```
        ┌─────────────────────┐    ┌──────────────────┐    ┌──────────────┐
        │  data_processor.py  │ →  │   engine.py      │ →  │   app.py     │
        │  FPL + Transfermkt  │    │ CRITIC / Entropy │    │   Dash UI    │
        │  per-90 features    │    │ 9 MCDM methods   │    │   callbacks  │
        └─────────────────────┘    └──────────────────┘    └──────┬───────┘
                                                                  │
                                            ┌─────────────────────┘
                                            ▼
                                   ┌──────────────────┐
                                   │  optimizer.py    │
                                   │  PuLP/CBC ILP    │
                                   │  budget-bounded  │
                                   └──────────────────┘
```

1. **`mcdm/data_processor.py`** joins `players.csv` (FPL roster), `playerstats.csv` (FPL cumulative stats), and `data/market_values.csv` (Transfermarkt). It computes per-90 features, filters to players with **≥ 450 minutes** played, and derives a canonical `role` column from the scraped `main_position`.
2. **`mcdm/criteria.py`** holds the broad-position fallback criteria, the 12 specific-role criteria sets, slot→role pool mappings, and the formation layouts.
3. **`mcdm/engine.py`** implements CRITIC, Shannon Entropy, the nine MCDM methods, and a Borda consensus that aggregates ranks across methods. VIKOR's `Q` is inverted to `1 − Q` so "higher is better" holds across the UI.
4. **`mcdm/optimizer.py`** turns each slot's MCDM ranking into a candidate frame, min-max normalises scores per slot, then solves an integer LP: one player per slot, no player in two slots, `budget_min ≤ Σ price ≤ budget_max`, maximise Σ normalised score.
5. **`app.py`** is the Dash UI. Callbacks own the cross-cutting state (formation, selected slot, method, weighting, budget window, slider overrides, assigned players, optimizer preview).

## MCDM methods at a glance

| Method | Idea | Score interpretation |
|---|---|---|
| PROMETHEE II | Pairwise outranking flows | Net Φ (higher = better) |
| VIKOR | Compromise ranking | `1 − Q` (higher = better; inverted from raw Q) |
| TOPSIS | Distance to ideal vs. anti-ideal | Closeness coefficient ∈ [0, 1] |
| SAW | Weighted sum after normalisation | Linear additive |
| WP | Weighted product | Multiplicative (penalises weak-on-any) |
| WASPAS | λ-blend of SAW + WP (λ = 0.5) | Hybrid additive/multiplicative |
| CODAS | Euclidean distance from anti-ideal, taxicab tiebreak | Higher = farther from worst |
| Borda Consensus | Ordinal aggregation across all seven base methods | Σ (n − rank + 1) |

## Weighting schemes

**CRITIC** — weight ∝ std-dev × Σ(1 − correlation). Rewards criteria that discriminate strongly and aren't redundant with others.

**Shannon Entropy** — weight ∝ (1 − normalised entropy). Rewards criteria with high information content (concentrated, non-uniform columns). A tiny `epsilon = 1e-12` offset keeps the log finite on zero rows.

Both are objective — derived from the data alone. Slider overrides let you push specific criteria up or down; the engine re-normalises on the fly. Switching scheme or position resets sliders to the new objective baseline.

## Position criteria (current)

Criteria are role-specific. Yellow/red cards are intentionally **not** used.

| Role | Criteria |
|---|---|
| GK | Saves, Clean Sheets, Save %, Goals Prevented, xGC, Influence |
| CB | Tackles, CBI, Clean Sheets, Recoveries, Defensive Contribution, xGC, Influence |
| LB / RB | Tackles, CBI, Recoveries, Defensive Contribution, xA, Assists, Creativity |
| CDM | Tackles, CBI, Recoveries, Defensive Contribution, BPS, Influence |
| CM | Goals, Assists, xGI, Creativity, Tackles, Recoveries, Influence |
| CAM | xA, xG, Goals, Assists, Creativity, Shooting Threat |
| LM / RM | xA, Assists, Creativity, Shooting Threat, Tackles, Defensive Contribution |
| LW / RW | xG, xA, xGI, Goals, Goals/xG, Assists, Creativity, Shooting Threat |
| ST | xG, Goals, Goals/xG, xGI, Shooting Threat, Assists, Aerial Ability, Defensive Contribution |

Role numbering follows the traditional convention: **CDM = 6** (anchor), **CM = 8** (box-to-box), **CAM = 10** (playmaker) — no CDM6/CDM8 splits.

## Squad optimizer (LP)

Click **Optimize XI** to build the best XI from scratch under the budget window. Click **Fill Empty** to keep your manually-assigned players and solve only the empty slots.

The ILP maximises `Σ score_norm` where each slot's MCDM score is min-max normalised to `[0, 1]` before optimisation — this keeps slots on different score scales (e.g. PROMETHEE Φ ≈ 0.02 vs. VIKOR `1−Q` ≈ 0.4) comparable. Constraints:

- exactly one player per slot
- no player assigned to more than one slot (matters when role pools overlap, e.g. CM ∩ CAM)
- `budget_min ≤ Σ price + Σ locked_price ≤ budget_max`

Locked players (in "Fill Empty" mode) are kept out of every other slot's candidate set.

Solved via PuLP with the CBC solver (pure pip install, no system dependencies). All 11 slots × hundreds of candidates resolve well under a second.

## Project layout

```
enm_project/
├── app.py                        # Dash layout + callbacks
├── requirements.txt
├── players.csv                   # FPL roster
├── playerstats.csv               # FPL cumulative per-gameweek stats
├── assets/
│   └── style.css                 # Dark theme
├── data/
│   └── market_values.csv         # Transfermarkt values + main_position
├── mcdm/
│   ├── criteria.py               # Role criteria, slot→role mapping, formations
│   ├── data_processor.py         # Feature engineering, role resolution
│   ├── engine.py                 # CRITIC, Entropy, 9 MCDM methods, Borda
│   └── optimizer.py              # PuLP ILP squad builder
├── scraper/
│   └── transfermarkt_scraper.py  # Pulls market values + main_position
└── tests/
    ├── test_engine_integration.py
    ├── test_mcdm_comparator.py
    └── test_optimizer.py         # 14 pytest cases for the LP
```

## Data sources

| File | Source | Description |
|---|---|---|
| `players.csv` | Fantasy Premier League API | Roster: id, name, position, team |
| `playerstats.csv` | Fantasy Premier League API | Cumulative stats per gameweek |
| `data/market_values.csv` | Transfermarkt (scraped) | Market values €m + `main_position` |

Players with fewer than 450 minutes played are excluded from rankings (≈ 5 full matches' worth of evidence).

## Typical workflow

1. **Pick a formation** and set the budget window (`Min` / `Max`).
2. **Click a position** on the pitch — the right panel shows ranked candidates for that role under the active method and weighting.
3. **Adjust weights** with the sliders if a specific criterion matters more for your tactical plan. Other unrelated controls won't reset your tuning.
4. **Click a player row** to assign them. They're locked out of every other slot's ranking.
5. **Optimize**: hit **Fill Empty** to LP-solve the remaining slots, or **Optimize XI** to rebuild from scratch. Preview the proposed XI, click **Apply** to commit.
6. **Switch methods** to gut-check stability — players who stay top of the rankings across PROMETHEE/VIKOR/TOPSIS are robust picks.

## Notes

- VIKOR's raw `Q` is inverted to `1 − Q` for display so "higher is better" holds across every method in the UI.
- The CBC solver is bundled with PuLP — no system install needed.
- The scraper writes `data/market_values.csv` and is **not** imported by the app; the committed CSV is the runtime source of truth.
