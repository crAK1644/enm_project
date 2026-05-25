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
    print("Testing Engine integration of 6 MCDM methods...")
    db = build_player_database(parent_dir)
    forwards = get_position_players(db, "Forward")
    criteria_config = POSITION_CRITERIA["Forward"]
    
    methods = ["PROMETHEE II", "VIKOR", "AHP", "TOPSIS", "SAW", "WP"]
    
    for method in methods:
        print(f"\n--- Testing {method} ---")
        try:
            ranked, critic_w, applied_w = rank_players(forwards, criteria_config, method=method)
            print(f"  - Calculated successfully for {len(ranked)} candidates.")
            print(f"  - Top candidate: {ranked.iloc[0]['full_name']} (Score: {ranked.iloc[0]['score']:.4f})")
        except Exception as e:
            print(f"  - FAILED: {e}")
            sys.exit(1)
            
    print("\nALL 6 METHODS WORK PERFECTLY IN THE INTEGRATED ENGINE!")

if __name__ == "__main__":
    run_tests()
