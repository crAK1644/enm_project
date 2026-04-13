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

## MCDM Methods

### CRITIC (weight determination)
Weights are derived from the decision matrix itself — criteria that vary a lot across players and correlate little with other criteria get higher weights. Weights are recomputed each time a position is selected. You can override any weight with the sliders; the app will normalize and re-rank without snapping the slider back.

### PROMETHEE II
Pairwise outranking method. Each player is compared against every other candidate across all criteria. The net outranking flow (Φ) gives a complete ranking — higher Φ is better.

### VIKOR
Compromise ranking method. Finds the solution closest to the ideal, balancing group utility (S) and individual regret (R) via the Q-value. Lower Q is better; the app inverts it for consistent display (higher = better).

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
