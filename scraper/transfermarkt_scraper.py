"""
Transfermarkt Premier League Market Value Scraper
Scrapes all 20 PL team squad pages and extracts player market values.
Outputs: data/market_values.csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import re
import sys
from difflib import SequenceMatcher

# Transfermarkt team URLs for PL 2025/26
TEAMS = {
    "Manchester City": ("manchester-city", 281),
    "Arsenal FC": ("fc-arsenal", 11),
    "Chelsea FC": ("fc-chelsea", 631),
    "Liverpool FC": ("fc-liverpool", 31),
    "Tottenham Hotspur": ("tottenham-hotspur", 148),
    "Manchester United": ("manchester-united", 985),
    "Newcastle United": ("newcastle-united", 762),
    "Nottingham Forest": ("nottingham-forest", 703),
    "Aston Villa": ("aston-villa", 405),
    "Crystal Palace": ("crystal-palace", 873),
    "AFC Bournemouth": ("afc-bournemouth", 989),
    "Brighton & Hove Albion": ("brighton-amp-hove-albion", 1237),
    "Brentford FC": ("fc-brentford", 1148),
    "Everton FC": ("fc-everton", 29),
    "Fulham FC": ("fc-fulham", 931),
    "Sunderland AFC": ("afc-sunderland", 289),
    "West Ham United": ("west-ham-united", 379),
    "Leeds United": ("leeds-united", 399),
    "Wolverhampton Wanderers": ("wolverhampton-wanderers", 543),
    "Burnley FC": ("fc-burnley", 1132),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.transfermarkt.com/",
    "Connection": "keep-alive",
}

def parse_market_value(value_str):
    """Convert '€50.00m' or '€500k' to float in millions."""
    if not value_str:
        return 0.0
    value_str = value_str.strip().replace("€", "").replace(",", ".")
    try:
        if "bn" in value_str.lower():
            return float(value_str.lower().replace("bn", "")) * 1000
        elif "m" in value_str.lower():
            return float(value_str.lower().replace("m", ""))
        elif "k" in value_str.lower():
            return float(value_str.lower().replace("k", "")) / 1000
        else:
            return float(value_str) / 1_000_000
    except ValueError:
        return 0.0


def extract_position_from_cell(cell):
    """Extract the position text (e.g. 'Centre-Forward') from a player cell.

    Transfermarkt nests an `inline-table` inside `td.posrela`; the second row
    of that table holds the position as plain text. Fall back to scanning
    any non-link cell text if the structure changes.
    """
    if not cell:
        return ""
    inline = cell.find("table", class_="inline-table")
    if inline:
        rows = inline.find_all("tr")
        if len(rows) >= 2:
            txt = rows[1].get_text(strip=True)
            if txt:
                return txt
    # Fallback: take the cell's text minus any link text.
    link = cell.find("a", href=re.compile(r"/profil/spieler/"))
    full = cell.get_text(" ", strip=True)
    if link:
        full = full.replace(link.get_text(strip=True), "", 1).strip()
    return full


def scrape_team(team_name, slug, team_id, session):
    """Scrape a single team's squad page: player name, main position, market value."""
    url = f"https://www.transfermarkt.com/{slug}/kader/verein/{team_id}/saison_id/2025"
    print(f"  Scraping {team_name}... ({url})")

    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    players = []

    # Walk the squad table row by row so name, position, and value stay aligned.
    # Each player row contains a td.posrela (with nested inline-table holding
    # the player name + position) and a td.rechts.hauptlink (market value).
    for row in soup.find_all("tr"):
        name_cell = row.find("td", class_="posrela")
        if not name_cell:
            continue
        name_link = name_cell.find("a", href=re.compile(r"/profil/spieler/\d+"))
        if not name_link or not name_link.text.strip():
            continue

        player_name = name_link.text.strip()
        position = extract_position_from_cell(name_cell)

        value_link = row.find("a", href=re.compile(r"/marktwertverlauf/spieler/"))
        value_text = value_link.text.strip() if value_link else ""
        if not value_text.startswith("€"):
            # No market value for this player — skip rather than store zero.
            continue

        players.append({
            "player_name": player_name,
            "main_position": position,
            "market_value_eur_m": parse_market_value(value_text),
            "market_value_str": value_text,
            "team": team_name,
        })

    print(f"    Found {len(players)} players")
    return players


def load_csv_players(csv_path):
    """Load player names from players.csv for matching."""
    players = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_name = f"{row['first_name']} {row['second_name']}"
            players.append({
                "player_id": row["player_id"],
                "full_name": full_name,
                "web_name": row["web_name"],
                "first_name": row["first_name"],
                "second_name": row["second_name"],
                "position": row["position"],
                "team_code": row["team_code"],
            })
    return players


def fuzzy_match(tm_name, csv_players, threshold=0.6):
    """Match a Transfermarkt name to the closest CSV player name."""
    best_match = None
    best_score = 0

    tm_name_lower = tm_name.lower().strip()

    for p in csv_players:
        # Try full name match
        score_full = SequenceMatcher(None, tm_name_lower, p["full_name"].lower()).ratio()
        # Try second name match (often more reliable)
        score_second = SequenceMatcher(None, tm_name_lower, p["second_name"].lower()).ratio()
        # Try web name match
        score_web = SequenceMatcher(None, tm_name_lower, p["web_name"].lower()).ratio()

        score = max(score_full, score_second, score_web)

        if score > best_score:
            best_score = score
            best_match = p

    if best_score >= threshold:
        return best_match, best_score
    return None, 0


def run_scraper(project_dir):
    """Main scraper entry point."""
    output_path = os.path.join(project_dir, "data", "market_values.csv")
    players_csv = os.path.join(project_dir, "players.csv")

    # Load CSV players for matching
    csv_players = load_csv_players(players_csv)
    print(f"Loaded {len(csv_players)} players from players.csv")

    all_players = []
    session = requests.Session()

    print(f"\nScraping {len(TEAMS)} Premier League teams from Transfermarkt...\n")

    for i, (team_name, (slug, team_id)) in enumerate(TEAMS.items()):
        players = scrape_team(team_name, slug, team_id, session)
        all_players.extend(players)

        # Rate limiting - be respectful
        if i < len(TEAMS) - 1:
            delay = 3
            print(f"    Waiting {delay}s...")
            time.sleep(delay)

    print(f"\nTotal scraped: {len(all_players)} players")

    # Match to CSV players
    matched = []
    unmatched = []

    for p in all_players:
        match, score = fuzzy_match(p["player_name"], csv_players)
        if match:
            matched.append({
                "player_id": match["player_id"],
                "player_name_tm": p["player_name"],
                "player_name_csv": match["full_name"],
                "web_name": match["web_name"],
                "market_value_eur_m": p["market_value_eur_m"],
                "market_value_str": p["market_value_str"],
                "team_tm": p["team"],
                "position": match["position"],
                "main_position": p.get("main_position", ""),
                "match_score": round(score, 3),
            })
        else:
            unmatched.append(p["player_name"])

    print(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    if unmatched:
        print(f"Unmatched players (sample): {unmatched[:10]}")

    # Write output CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "player_id", "player_name_tm", "player_name_csv", "web_name",
            "market_value_eur_m", "market_value_str", "team_tm", "position",
            "main_position", "match_score"
        ])
        writer.writeheader()
        writer.writerows(matched)

    print(f"\nSaved to {output_path}")
    return matched


def dry_run_one_team():
    """Scrape a single team (Arsenal) and print results for sanity-checking."""
    session = requests.Session()
    name = "Arsenal FC"
    slug, team_id = TEAMS[name]
    players = scrape_team(name, slug, team_id, session)
    print(f"\n--- Dry run: {name} ---")
    for p in players:
        print(f"  {p['player_name']:30s} | {p['main_position']:25s} | {p['market_value_str']}")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        dry_run_one_team()
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_scraper(project_dir)
