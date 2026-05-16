"""
Position-specific criteria definitions for MCDM evaluation.
Maps football criteria to CSV column names and specifies benefit/cost types.
"""

# Criteria type: 1 = benefit (higher is better), -1 = cost (lower is better)

# Broad-position criteria (used as fallback when a player's specific role is unknown).
POSITION_CRITERIA = {
    "Forward": {
        "xG": {"column": "expected_goals_per90", "type": 1, "label": "Expected Goals (xG)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Goals/xG": {"column": "goals_per_xg", "type": 1, "label": "Goals / xG (Finishing)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
        "Aerial Ability": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Aerial Ability"},
    },
    "Midfielder": {
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Interceptions": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Interceptions"},
    },
    "Defender": {
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Interceptions": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Interceptions"},
        "Clean Sheets": {"column": "clean_sheets_per90", "type": 1, "label": "Clean Sheets"},
        "Influence": {"column": "influence_per90", "type": 1, "label": "Influence"},
    },
    "Goalkeeper": {
        "Saves": {"column": "saves_per90", "type": 1, "label": "Saves"},
        "Clean Sheets": {"column": "clean_sheets_per90", "type": 1, "label": "Clean Sheets"},
        "Save %": {"column": "save_percentage", "type": 1, "label": "Save %"},
        "Goals Prevented": {"column": "goals_prevented_per90", "type": 1, "label": "Goals Prevented (xGC − GC)"},
        "xGC": {"column": "expected_goals_conceded_per90", "type": -1, "label": "Expected Goals Conceded"},
        "Influence": {"column": "influence_per90", "type": 1, "label": "Influence"},
    },
}

# Specific-role criteria. Naming follows football shirt numbers:
#   CDM = 6 (anchor / destroyer)
#   CM  = 8 (box-to-box)
#   CAM = 10 (playmaker)
ROLE_CRITERIA = {
    "GK": POSITION_CRITERIA["Goalkeeper"],
    "CB": {
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Clearances/Blocks/Int": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Clearances/Blocks/Int"},
        "Clean Sheets": {"column": "clean_sheets_per90", "type": 1, "label": "Clean Sheets"},
        "Recoveries": {"column": "recoveries_per90", "type": 1, "label": "Recoveries"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
        "xGC": {"column": "expected_goals_conceded_per90", "type": -1, "label": "Expected Goals Conceded"},
        "Influence": {"column": "influence_per90", "type": 1, "label": "Influence"},
    },
    "LB": {
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Clearances/Blocks/Int": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Clearances/Blocks/Int"},
        "Recoveries": {"column": "recoveries_per90", "type": 1, "label": "Recoveries"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
    },
    "RB": {
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Clearances/Blocks/Int": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Clearances/Blocks/Int"},
        "Recoveries": {"column": "recoveries_per90", "type": 1, "label": "Recoveries"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
    },
    # CDM — the 6: anchor / destroyer, screens the back line.
    "CDM": {
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Clearances/Blocks/Int": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Clearances/Blocks/Int"},
        "Recoveries": {"column": "recoveries_per90", "type": 1, "label": "Recoveries"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
        "BPS": {"column": "bps_per90", "type": 1, "label": "Bonus Point Score"},
        "Influence": {"column": "influence_per90", "type": 1, "label": "Influence"},
    },
    # CM — the 8: box-to-box, balanced creation and defensive output.
    "CM": {
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "xGI": {"column": "expected_goal_involvements_per90", "type": 1, "label": "Expected G+A (xGI)"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Recoveries": {"column": "recoveries_per90", "type": 1, "label": "Recoveries"},
        "Influence": {"column": "influence_per90", "type": 1, "label": "Influence"},
    },
    # CAM — the 10: playmaker, between-the-lines threat and creation.
    "CAM": {
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "xG": {"column": "expected_goals_per90", "type": 1, "label": "Expected Goals (xG)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
    },
    "LM": {
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
    },
    "RM": {
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
        "Tackles": {"column": "tackles_per90", "type": 1, "label": "Tackles"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
    },
    "LW": {
        "xG": {"column": "expected_goals_per90", "type": 1, "label": "Expected Goals (xG)"},
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "xGI": {"column": "expected_goal_involvements_per90", "type": 1, "label": "Expected G+A (xGI)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Goals/xG": {"column": "goals_per_xg", "type": 1, "label": "Goals / xG (Finishing)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
    },
    "RW": {
        "xG": {"column": "expected_goals_per90", "type": 1, "label": "Expected Goals (xG)"},
        "xA": {"column": "expected_assists_per90", "type": 1, "label": "Expected Assists (xA)"},
        "xGI": {"column": "expected_goal_involvements_per90", "type": 1, "label": "Expected G+A (xGI)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Goals/xG": {"column": "goals_per_xg", "type": 1, "label": "Goals / xG (Finishing)"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Creativity": {"column": "creativity_per90", "type": 1, "label": "Creativity"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
    },
    "ST": {
        "xG": {"column": "expected_goals_per90", "type": 1, "label": "Expected Goals (xG)"},
        "Goals": {"column": "goals_scored_per90", "type": 1, "label": "Goals"},
        "Goals/xG": {"column": "goals_per_xg", "type": 1, "label": "Goals / xG (Finishing)"},
        "xGI": {"column": "expected_goal_involvements_per90", "type": 1, "label": "Expected G+A (xGI)"},
        "Shooting Threat": {"column": "threat_per90", "type": 1, "label": "Shooting Threat"},
        "Assists": {"column": "assists_per90", "type": 1, "label": "Assists"},
        "Aerial Ability": {"column": "clearances_blocks_interceptions_per90", "type": 1, "label": "Aerial Ability"},
        "Defensive Contribution": {"column": "defensive_contribution_per90", "type": 1, "label": "Defensive Work"},
    },
}

# Slot → role/pool/broad mapping.
#   role  : key into ROLE_CRITERIA (which criteria set to use)
#   pool  : canonical roles a player can be tagged as to be eligible for this slot
#   broad : FPL broad position used as fallback when a player has no specific role tag
SLOT_TO_ROLE = {
    "GK":   {"role": "GK",  "pool": ["GK"],               "broad": "Goalkeeper"},
    "CB":   {"role": "CB",  "pool": ["CB"],               "broad": "Defender"},
    "CB1":  {"role": "CB",  "pool": ["CB"],               "broad": "Defender"},
    "CB2":  {"role": "CB",  "pool": ["CB"],               "broad": "Defender"},
    "CB3":  {"role": "CB",  "pool": ["CB"],               "broad": "Defender"},
    "LB":   {"role": "LB",  "pool": ["LB"],               "broad": "Defender"},
    "RB":   {"role": "RB",  "pool": ["RB"],               "broad": "Defender"},
    "LWB":  {"role": "LB",  "pool": ["LB"],               "broad": "Defender"},
    "RWB":  {"role": "RB",  "pool": ["RB"],               "broad": "Defender"},
    "CDM":  {"role": "CDM", "pool": ["CDM"],              "broad": "Midfielder"},
    "CDM1": {"role": "CDM", "pool": ["CDM"],              "broad": "Midfielder"},
    "CDM2": {"role": "CDM", "pool": ["CDM"],              "broad": "Midfielder"},
    "CM":   {"role": "CM",  "pool": ["CM", "CDM", "CAM"], "broad": "Midfielder"},
    "CM1":  {"role": "CM",  "pool": ["CM", "CDM", "CAM"], "broad": "Midfielder"},
    "CM2":  {"role": "CM",  "pool": ["CM", "CDM", "CAM"], "broad": "Midfielder"},
    "CM3":  {"role": "CM",  "pool": ["CM", "CDM", "CAM"], "broad": "Midfielder"},
    "CAM":  {"role": "CAM", "pool": ["CAM", "CM"],        "broad": "Midfielder"},
    "LM":   {"role": "LM",  "pool": ["LM", "LW"],         "broad": "Midfielder"},
    "RM":   {"role": "RM",  "pool": ["RM", "RW"],         "broad": "Midfielder"},
    "LW":   {"role": "LW",  "pool": ["LW", "LM"],         "broad": "Forward"},
    "RW":   {"role": "RW",  "pool": ["RW", "RM"],         "broad": "Forward"},
    "ST":   {"role": "ST",  "pool": ["ST"],               "broad": "Forward"},
    "ST1":  {"role": "ST",  "pool": ["ST"],               "broad": "Forward"},
    "ST2":  {"role": "ST",  "pool": ["ST"],               "broad": "Forward"},
}

# Formation definitions: each slot has x/y pitch coords and a broad-position tag
# (kept for backward compat). Specific-role behavior is driven by SLOT_TO_ROLE.
FORMATIONS = {
    "4-3-3": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "LB":  {"pos": "Defender",   "x": 15, "y": 72},
        "CB1": {"pos": "Defender",   "x": 37, "y": 75},
        "CB2": {"pos": "Defender",   "x": 63, "y": 75},
        "RB":  {"pos": "Defender",   "x": 85, "y": 72},
        "CDM": {"pos": "Midfielder", "x": 50, "y": 60},
        "CM":  {"pos": "Midfielder", "x": 30, "y": 50},
        "CAM": {"pos": "Midfielder", "x": 70, "y": 50},
        "LW":  {"pos": "Forward",    "x": 20, "y": 28},
        "ST":  {"pos": "Forward",    "x": 50, "y": 25},
        "RW":  {"pos": "Forward",    "x": 80, "y": 28},
    },
    "4-4-2": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "LB":  {"pos": "Defender",   "x": 15, "y": 72},
        "CB1": {"pos": "Defender",   "x": 37, "y": 75},
        "CB2": {"pos": "Defender",   "x": 63, "y": 75},
        "RB":  {"pos": "Defender",   "x": 85, "y": 72},
        "LM":  {"pos": "Midfielder", "x": 15, "y": 50},
        "CDM": {"pos": "Midfielder", "x": 37, "y": 53},
        "CM":  {"pos": "Midfielder", "x": 63, "y": 53},
        "RM":  {"pos": "Midfielder", "x": 85, "y": 50},
        "ST1": {"pos": "Forward",    "x": 37, "y": 25},
        "ST2": {"pos": "Forward",    "x": 63, "y": 25},
    },
    "3-5-2": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "CB1": {"pos": "Defender",   "x": 25, "y": 75},
        "CB2": {"pos": "Defender",   "x": 50, "y": 77},
        "CB3": {"pos": "Defender",   "x": 75, "y": 75},
        "LWB": {"pos": "Defender",   "x": 10, "y": 55},
        "CDM": {"pos": "Midfielder", "x": 32, "y": 52},
        "CM":  {"pos": "Midfielder", "x": 50, "y": 50},
        "CAM": {"pos": "Midfielder", "x": 68, "y": 52},
        "RWB": {"pos": "Defender",   "x": 90, "y": 55},
        "ST1": {"pos": "Forward",    "x": 37, "y": 25},
        "ST2": {"pos": "Forward",    "x": 63, "y": 25},
    },
    # User-specified 4-2-3-1: GK LB CB CB RB CDM(6) CM(8) LM CAM(10) RM ST
    "4-2-3-1": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "LB":  {"pos": "Defender",   "x": 15, "y": 72},
        "CB1": {"pos": "Defender",   "x": 37, "y": 75},
        "CB2": {"pos": "Defender",   "x": 63, "y": 75},
        "RB":  {"pos": "Defender",   "x": 85, "y": 72},
        "CDM": {"pos": "Midfielder", "x": 37, "y": 58},
        "CM":  {"pos": "Midfielder", "x": 63, "y": 58},
        "LM":  {"pos": "Midfielder", "x": 20, "y": 40},
        "CAM": {"pos": "Midfielder", "x": 50, "y": 38},
        "RM":  {"pos": "Midfielder", "x": 80, "y": 40},
        "ST":  {"pos": "Forward",    "x": 50, "y": 22},
    },
    "3-4-3": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "CB1": {"pos": "Defender",   "x": 25, "y": 75},
        "CB2": {"pos": "Defender",   "x": 50, "y": 77},
        "CB3": {"pos": "Defender",   "x": 75, "y": 75},
        "LM":  {"pos": "Midfielder", "x": 15, "y": 52},
        "CDM": {"pos": "Midfielder", "x": 37, "y": 55},
        "CM":  {"pos": "Midfielder", "x": 63, "y": 55},
        "RM":  {"pos": "Midfielder", "x": 85, "y": 52},
        "LW":  {"pos": "Forward",    "x": 20, "y": 28},
        "ST":  {"pos": "Forward",    "x": 50, "y": 25},
        "RW":  {"pos": "Forward",    "x": 80, "y": 28},
    },
    "5-3-2": {
        "GK":  {"pos": "Goalkeeper", "x": 50, "y": 90},
        "LWB": {"pos": "Defender",   "x": 10, "y": 65},
        "CB1": {"pos": "Defender",   "x": 30, "y": 75},
        "CB2": {"pos": "Defender",   "x": 50, "y": 77},
        "CB3": {"pos": "Defender",   "x": 70, "y": 75},
        "RWB": {"pos": "Defender",   "x": 90, "y": 65},
        "CDM": {"pos": "Midfielder", "x": 30, "y": 50},
        "CM":  {"pos": "Midfielder", "x": 50, "y": 48},
        "CAM": {"pos": "Midfielder", "x": 70, "y": 50},
        "ST1": {"pos": "Forward",    "x": 37, "y": 25},
        "ST2": {"pos": "Forward",    "x": 63, "y": 25},
    },
}
