"""
Data processor: loads, merges, and aggregates player data from CSVs.
Produces a clean DataFrame ready for MCDM evaluation.
"""

import pandas as pd
import numpy as np
import os


def load_players(project_dir):
    """Load player roster from players.csv."""
    path = os.path.join(project_dir, "players.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={"player_id": "id"})
    df["id"] = df["id"].astype(str)
    return df


def load_stats(project_dir):
    """Load player stats from playerstats.csv, taking the latest gameweek (cumulative)."""
    path = os.path.join(project_dir, "playerstats.csv")
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")

    # Take the latest gameweek row for each player (data is cumulative)
    idx = df.groupby("id")["gw"].idxmax()
    latest = df.loc[idx].copy()

    # Convert numeric columns
    numeric_cols = [
        "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "yellow_cards", "red_cards", "saves", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded", "influence",
        "creativity", "threat", "ict_index", "tackles",
        "clearances_blocks_interceptions", "recoveries", "defensive_contribution",
        "saves_per_90", "clean_sheets_per_90", "goals_conceded_per_90",
        "defensive_contribution_per_90", "now_cost",
    ]
    for col in numeric_cols:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors="coerce").fillna(0)

    return latest


def compute_per90(df):
    """Compute per-90-minute stats for counting metrics."""
    df = df.copy()
    minutes = df["minutes"].clip(lower=1)  # avoid division by zero

    per90_mappings = {
        "goals_scored_per90": "goals_scored",
        "assists_per90": "assists",
        "expected_goals_per90": "expected_goals",
        "expected_assists_per90": "expected_assists",
        "yellow_cards_per90": "yellow_cards",
        "red_cards_per90": "red_cards",
        "tackles_per90": "tackles",
        "clearances_blocks_interceptions_per90": "clearances_blocks_interceptions",
        "saves_per90_calc": "saves",
        "clean_sheets_per90_calc": "clean_sheets",
        "goals_conceded_per90_calc": "goals_conceded",
        "influence_per90": "influence",
        "creativity_per90": "creativity",
        "threat_per90": "threat",
        "defensive_contribution_per90_calc": "defensive_contribution",
        "recoveries_per90": "recoveries",
    }

    for new_col, src_col in per90_mappings.items():
        if src_col in df.columns:
            df[new_col] = (df[src_col] / minutes) * 90

    # Use pre-computed per90 where available, fall back to our calculation
    if "saves_per_90" in df.columns:
        df["saves_per90"] = df["saves_per_90"]
    else:
        df["saves_per90"] = df.get("saves_per90_calc", 0)

    if "clean_sheets_per_90" in df.columns:
        df["clean_sheets_per90"] = df["clean_sheets_per_90"]
    else:
        df["clean_sheets_per90"] = df.get("clean_sheets_per90_calc", 0)

    if "goals_conceded_per_90" in df.columns:
        df["goals_conceded_per90"] = df["goals_conceded_per_90"]
    else:
        df["goals_conceded_per90"] = df.get("goals_conceded_per90_calc", 0)

    if "defensive_contribution_per_90" in df.columns:
        df["defensive_contribution_per90"] = df["defensive_contribution_per_90"]
    else:
        df["defensive_contribution_per90"] = df.get("defensive_contribution_per90_calc", 0)

    return df


def load_market_values(project_dir):
    """Load scraped market values from data/market_values.csv."""
    path = os.path.join(project_dir, "data", "market_values.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["player_id"] = df["player_id"].astype(str)

    # Keep only the best-matching entry per player (highest match_score).
    # This prevents duplicate rows after the left join in build_player_database.
    if "match_score" in df.columns:
        df = (df.sort_values("match_score", ascending=False)
                .drop_duplicates(subset="player_id", keep="first")
                .reset_index(drop=True))

    return df


def build_player_database(project_dir, min_minutes=450):
    """
    Build the complete player database:
    1. Load player roster
    2. Load latest stats 
    3. Compute per-90 stats
    4. Merge market values
    5. Filter by minimum minutes
    """
    players = load_players(project_dir)
    stats = load_stats(project_dir)
    
    # Merge players with stats
    merged = stats.merge(
        players[["id", "position", "first_name", "second_name", "web_name", "team_code"]],
        on="id",
        how="inner",
        suffixes=("", "_roster"),
    )

    # Use roster names if stats names are missing
    for col in ["first_name", "second_name", "web_name"]:
        if f"{col}_roster" in merged.columns:
            mask = merged[col].isna() | (merged[col] == "")
            merged.loc[mask, col] = merged.loc[mask, f"{col}_roster"]

    # Compute per-90 stats
    merged = compute_per90(merged)

    # Filter by minimum minutes played
    merged = merged[merged["minutes"] >= min_minutes].copy()

    # Load and merge market values
    mv = load_market_values(project_dir)
    if mv is not None:
        merged = merged.merge(
            mv[["player_id", "market_value_eur_m", "team_tm"]],
            left_on="id",
            right_on="player_id",
            how="left",
        )
        merged["market_value_eur_m"] = merged["market_value_eur_m"].fillna(0)
    else:
        # Use FPL cost as fallback (in £ tenths of millions, e.g. 146 = £14.6m)
        merged["market_value_eur_m"] = merged["now_cost"] / 10.0
        merged["team_tm"] = ""

    # Create display name
    merged["display_name"] = merged["web_name"]
    merged["full_name"] = merged["first_name"].fillna("") + " " + merged["second_name"].fillna("")

    # Replace inf/nan with 0
    merged = merged.replace([np.inf, -np.inf], 0)
    merged = merged.fillna(0)

    print(f"Player database built: {len(merged)} players (min {min_minutes} mins)")
    print(f"  Forwards: {len(merged[merged['position'] == 'Forward'])}")
    print(f"  Midfielders: {len(merged[merged['position'] == 'Midfielder'])}")
    print(f"  Defenders: {len(merged[merged['position'] == 'Defender'])}")
    print(f"  Goalkeepers: {len(merged[merged['position'] == 'Goalkeeper'])}")

    return merged


def get_position_players(db, position):
    """Get all players for a given position."""
    return db[db["position"] == position].copy()
