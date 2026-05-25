# Transfer Window Manager

A football manager-style decision support application for evaluating and ranking Premier League players during the transfer window. Built with Python and Dash, it uses two Multi-Criteria Decision Making (MCDM) methods — **PROMETHEE II** and **VIKOR** — combined with the **CRITIC** objective weighting method to rank candidates at each position.

## Overview

The app loads FPL player and stats data, merges Transfermarkt market values, and lets you build a squad interactively. Click a position node on the pitch to see a ranked list of the best available players for that role. Assign players, track your budget, and adjust the criteria weights to match your scouting priorities.

![screenshot placeholder]

## Features

- **Interactive pitch** — select any formation, click a position to see ranked candidates
- **Two MCDM methods** — switch between PROMETHEE II (outranking flows) and VIKOR (compromise ranking) live
- **CRITIC weighting** — objective weights computed from data variance and inter-criteria correlation; override any weight with the sliders
- **Budget tracking** — set a transfer budget, see spend and remaining update in real time as you assign players
- **Player exclusion** — once a player is assigned to a position, they are removed from all other positions' ranking lists
- **Formation support** — 4-3-3, 4-4-2, 3-5-2, 4-2-3-1, 3-4-3, 5-3-2

## Project Structure

```
enm_project/
├── app.py                        # Dash application — layout and callbacks
├── requirements.txt
├── players.csv                   # FPL player roster (id, name, position, team)
├── playerstats.csv               # FPL cumulative per-gameweek stats
├── assets/
│   └── style.css                 # Dark theme stylesheet
├── data/
│   └── market_values.csv         # Scraped Transfermarkt market values
├── mcdm/
│   ├── criteria.py               # Position criteria definitions and formation layouts
│   ├── data_processor.py         # Data loading, per-90 computation, DB build
│   └── engine.py                 # CRITIC, PROMETHEE II, and VIKOR implementations
└── scraper/
    └── transfermarkt_scraper.py  # Scrapes market values for all 20 PL teams
```

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**Run the app:**

```bash
python app.py
```

Then open [http://localhost:8050](http://localhost:8050) in your browser.

**Refresh market values** (optional — `data/market_values.csv` is included):

```bash
python scraper/transfermarkt_scraper.py
```

This scrapes all 20 Premier League team pages on Transfermarkt and writes `data/market_values.csv`. Run it again at the start of each transfer window to get updated values.

## Data Sources

| File | Source | Description |
|---|---|---|
| `players.csv` | Fantasy Premier League API | Player roster: id, name, position, team |
| `playerstats.csv` | Fantasy Premier League API | Cumulative stats per gameweek |
| `data/market_values.csv` | Transfermarkt (scraped) | Market values in €m for PL players |

Only players with **450+ minutes played** are included in rankings (roughly 5 full matches).

## MCDM & Consensus Methods

### ⚖️ Objective Weighting & Hybridization (Shannon-CRITIC)
Criteria weights are determined objectively from the data, combining two complementary Operations Research methodologies:
1. **CRITIC:** Focuses on inter-criteria correlation and column variance.
2. **Shannon Entropy:** Measures informational uncertainty and diversification degree.
   * *Epsilon Limit:* A structural probability offset of `epsilon = 1e-12` is applied to satisfy logarithmic constraints and prevent domain errors.
3. **$\alpha$-Blending Interface:** Blends both models using a compromise slider:
   $$W_{h} = \alpha W_{c} + (1 - \alpha) W_{e}$$

---

### 🚀 Vectorized Mathematical Engines (All $O(m \times n)$ Complexity)
All six algorithms are fully vectorized using NumPy and Pandas for rapid runtime calculations:
- **PROMETHEE II:** Outranking method calculating Net Flows ($\Phi$).
- **VIKOR:** Compromise ranking balancing utility and regret (inverted as $1.0 - Q$ for display uniformity).
- **AHP (Analytic Hierarchy Process):** Derives composite alternatives' priority vectors.
- **TOPSIS:** Spatial closeness calculation matching ideal positive and negative solutions.
- **SAW (Simple Additive Weighting):** Fast linear min-max aggregation.
- **WP (Weighted Product):** Product scoring using exponential weights.
  * *Epsilon Limit:* Extends numerical stability using an offset matrix boundary of `epsilon = 1e-5` to completely avoid zero-base negative exponent division-by-zero crashes on cost attributes.

---

### 🗳️ Master Borda Count Consensus Aggregator
Synthesizes the ordinal outputs from all 6 active algorithms into a single mathematically sound consensus team recommendation. Point distribution is computed as:
$$\text{Points} = (\text{Alternatives}) - \text{Rank} + 1$$
Consensus points are summed across the 6 models to produce a unified compromise squad.

## Position Criteria

| Position | Criteria |
|---|---|
| Forward | xG, Goals, Assists, Shooting Threat, Creativity, Defensive Contribution, Aerial Ability, Yellow Cards, Red Cards |
| Midfielder | xA, Goals, Assists, Shooting Threat, Creativity, Tackles, Interceptions, Yellow Cards, Red Cards |
| Defender | Goals, Tackles, Interceptions, Clean Sheets, Influence, Yellow Cards, Red Cards |
| Goalkeeper | Saves, Clean Sheets, Goals Conceded, Influence, Yellow Cards, Red Cards |

Yellow and Red Cards are cost criteria (lower is better). All others are benefit criteria.

## Usage

1. **Set your budget** using the input or slider at the top
2. **Select a formation** from the dropdown
3. **Click a position node** on the pitch — the right panel shows ranked players for that role
4. **Click a player row** to assign them to that position
5. **Adjust criteria weights** in the panel at the bottom — rankings update immediately
6. **Switch methods** (PROMETHEE / VIKOR) with the toggle in the header
7. **Clear Squad** to start over
