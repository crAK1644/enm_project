import sys
import os
import numpy as np
import pandas as pd

# Add the parent directory of this test file (enm_project) to the path so mcdm can be imported
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from mcdm.data_processor import build_player_database, get_position_players
from mcdm.criteria import POSITION_CRITERIA
from mcdm.engine import rank_players

def run_tests():
    print("Initializing FPL player database data pipeline...")
    db = build_player_database(parent_dir)
    
    print("\nSelecting 'Forward' position pool...")
    forwards = get_position_players(db, "Forward")
    
    print(f"Loaded {len(forwards)} forward candidates.")
    
    criteria_config = POSITION_CRITERIA["Forward"]
    
    methods = ["PROMETHEE II", "VIKOR", "AHP", "TOPSIS", "SAW", "WP", "Borda Consensus"]
    
    # Evaluate with alpha = 0.5 (Shannon Entropy & CRITIC blend)
    alpha = 0.5
    print(f"\nEvaluating with dynamic weight compromise factor (alpha = {alpha})...")
    
    for method in methods:
        print(f"\n--- {method} Top 3 Players ---")
        ranked, _, _ = rank_players(forwards, criteria_config, method=method, alpha=alpha)
        for i in range(3):
            if i < len(ranked):
                row = ranked.iloc[i]
                print(f"{i+1}. {row['full_name']} (Score: {row['score']:.4f}, Rank: {row['rank']})")

if __name__ == "__main__":
    run_tests()
