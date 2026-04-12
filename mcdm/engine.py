"""
MCDM Engine: CRITIC weight determination + PROMETHEE II + VIKOR ranking methods.
All implemented from scratch (no external MCDM library needed).
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# CRITIC: Objective weight determination
# ─────────────────────────────────────────────────────────────

def critic_weights(matrix, types):
    """
    Calculate objective weights using the CRITIC method.
    
    CRITIC (Criteria Importance Through Intercriteria Correlation) determines
    weights based on both the contrast intensity (standard deviation) and
    the conflict between criteria (correlation).
    
    Parameters:
        matrix: np.ndarray (m x n) - decision matrix (m alternatives, n criteria)
        types:  np.ndarray (n,)    - criterion types (1=benefit, -1=cost)
    
    Returns:
        weights: np.ndarray (n,) - normalized CRITIC weights
    """
    m, n = matrix.shape
    
    # Step 1: Normalize the matrix (min-max normalization)
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        denom = col_max - col_min if col_max != col_min else 1.0
        
        if types[j] == 1:  # benefit
            norm[:, j] = (col - col_min) / denom
        else:  # cost
            norm[:, j] = (col_max - col) / denom
    
    # Step 2: Calculate standard deviation of each criterion
    std = np.std(norm, axis=0, ddof=0)
    # Prevent zero std
    std = np.where(std == 0, 1e-10, std)
    
    # Step 3: Calculate Pearson correlation matrix
    # Use errstate to suppress divide-by-zero/invalid warnings that arise when
    # a criterion has zero variance (all players share the same value).
    # nan_to_num then converts those entries to 0, giving the column zero
    # conflict weight — the correct outcome for a non-discriminating criterion.
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.corrcoef(norm.T)
    corr = np.nan_to_num(corr, nan=0.0)
    
    # Step 4: Calculate information content for each criterion
    # Cj = std_j * sum(1 - r_jk) for all k
    info = np.zeros(n)
    for j in range(n):
        conflict = np.sum(1 - corr[j, :])
        info[j] = std[j] * conflict
    
    # Step 5: Normalize to get weights
    total = info.sum()
    if total == 0:
        weights = np.ones(n) / n
    else:
        weights = info / total
    
    return weights


# ─────────────────────────────────────────────────────────────
# PROMETHEE II: Outranking method
# ─────────────────────────────────────────────────────────────

def promethee_ii(matrix, weights, types, preference_fn="usual"):
    """
    PROMETHEE II complete ranking.
    
    Uses pairwise comparison of alternatives across all criteria,
    then calculates net outranking flow (Phi) for a complete ranking.
    
    Parameters:
        matrix:        np.ndarray (m x n) - decision matrix
        weights:       np.ndarray (n,)    - criteria weights
        types:         np.ndarray (n,)    - criterion types (1=benefit, -1=cost)
        preference_fn: str                - preference function type
    
    Returns:
        phi_net:  np.ndarray (m,) - net flow scores (higher = better)
        rankings: np.ndarray (m,) - rank positions (1 = best)
    """
    m, n = matrix.shape
    
    # Preference function
    def preference(d, fn_type="usual"):
        """Calculate preference degree for difference d."""
        if fn_type == "usual":
            return 1.0 if d > 0 else 0.0
        elif fn_type == "linear":
            q, p = 0.0, 1.0  # thresholds
            if d <= q:
                return 0.0
            elif d >= p:
                return 1.0
            else:
                return (d - q) / (p - q)
        return 1.0 if d > 0 else 0.0
    
    # Normalize matrix for comparison
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        col_min, col_max = col.min(), col.max()
        denom = col_max - col_min if col_max != col_min else 1.0
        norm[:, j] = (col - col_min) / denom
    
    # Calculate pairwise preference index
    pi = np.zeros((m, m))
    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            weighted_pref = 0.0
            for j in range(n):
                if types[j] == 1:  # benefit
                    d = norm[i, j] - norm[k, j]
                else:  # cost
                    d = norm[k, j] - norm[i, j]
                weighted_pref += weights[j] * preference(d, preference_fn)
            pi[i, k] = weighted_pref
    
    # Calculate flows
    phi_plus = np.sum(pi, axis=1) / (m - 1) if m > 1 else np.zeros(m)   # positive flow
    phi_minus = np.sum(pi, axis=0) / (m - 1) if m > 1 else np.zeros(m)  # negative flow
    phi_net = phi_plus - phi_minus  # net flow
    
    # Ranking (1 = best, i.e., highest phi_net)
    rankings = np.argsort(-phi_net) + 1
    rank_positions = np.empty_like(rankings)
    rank_positions[np.argsort(-phi_net)] = np.arange(1, m + 1)
    
    return phi_net, rank_positions


# ─────────────────────────────────────────────────────────────
# VIKOR: Compromise ranking
# ─────────────────────────────────────────────────────────────

def vikor(matrix, weights, types, v=0.5):
    """
    VIKOR compromise ranking method.
    
    Determines a compromise solution based on the closeness to the
    ideal solution, balancing group utility (S) and individual regret (R).
    
    Parameters:
        matrix:  np.ndarray (m x n) - decision matrix
        weights: np.ndarray (n,)    - criteria weights
        types:   np.ndarray (n,)    - criterion types (1=benefit, -1=cost)
        v:       float              - strategy weight (0.5 = consensus)
    
    Returns:
        Q:        np.ndarray (m,) - VIKOR Q-values (lower = better)
        S:        np.ndarray (m,) - group utility values
        R:        np.ndarray (m,) - individual regret values
        rankings: np.ndarray (m,) - rank positions (1 = best)
    """
    m, n = matrix.shape
    
    # Step 1: Determine ideal (f*) and anti-ideal (f-) values
    f_star = np.zeros(n)
    f_minus = np.zeros(n)
    
    for j in range(n):
        if types[j] == 1:  # benefit
            f_star[j] = matrix[:, j].max()
            f_minus[j] = matrix[:, j].min()
        else:  # cost
            f_star[j] = matrix[:, j].min()
            f_minus[j] = matrix[:, j].max()
    
    # Step 2: Calculate S (group utility) and R (individual regret)
    S = np.zeros(m)
    R = np.zeros(m)
    
    for i in range(m):
        for j in range(n):
            denom = f_star[j] - f_minus[j]
            if abs(denom) < 1e-10:
                normalized = 0.0
            else:
                if types[j] == 1:
                    normalized = (f_star[j] - matrix[i, j]) / denom
                else:
                    normalized = (matrix[i, j] - f_star[j]) / denom
            
            weighted = weights[j] * normalized
            S[i] += weighted
            R[i] = max(R[i], weighted)
    
    # Step 3: Calculate Q values
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    
    Q = np.zeros(m)
    for i in range(m):
        s_term = (S[i] - S_star) / (S_minus - S_star) if (S_minus - S_star) > 1e-10 else 0.0
        r_term = (R[i] - R_star) / (R_minus - R_star) if (R_minus - R_star) > 1e-10 else 0.0
        Q[i] = v * s_term + (1 - v) * r_term
    
    # Ranking (1 = best, i.e., lowest Q)
    rank_positions = np.empty(m, dtype=int)
    rank_positions[np.argsort(Q)] = np.arange(1, m + 1)
    
    return Q, S, R, rank_positions


# ─────────────────────────────────────────────────────────────
# Unified ranking function
# ─────────────────────────────────────────────────────────────

def rank_players(player_df, criteria_config, method="promethee", custom_weights=None):
    """
    Rank players using the specified MCDM method.
    
    Parameters:
        player_df:       pd.DataFrame       - player data
        criteria_config: dict               - from POSITION_CRITERIA
        method:          str                - "promethee" or "vikor"
        custom_weights:  dict or None       - custom weights {criteria_name: weight}
    
    Returns:
        pd.DataFrame with added rank and score columns
    """
    if len(player_df) < 2:
        player_df = player_df.copy()
        player_df["rank"] = 1
        player_df["score"] = 1.0
        return player_df
    
    # Build decision matrix
    criteria_names = list(criteria_config.keys())
    columns = [criteria_config[c]["column"] for c in criteria_names]
    types = np.array([criteria_config[c]["type"] for c in criteria_names])
    
    # Extract matrix
    matrix = player_df[columns].values.astype(float)
    
    # Handle NaN/Inf
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate CRITIC weights
    critic_w = critic_weights(matrix, types)
    
    # Use custom weights if provided, otherwise use CRITIC
    if custom_weights:
        weights = np.array([custom_weights.get(c, critic_w[i]) for i, c in enumerate(criteria_names)])
        # Normalize
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = critic_w
    else:
        weights = critic_w
    
    # Apply MCDM method
    result = player_df.copy()
    
    if method == "promethee":
        scores, ranks = promethee_ii(matrix, weights, types)
        result["score"] = scores
        result["rank"] = ranks
    elif method == "vikor":
        Q, S, R, ranks = vikor(matrix, weights, types)
        result["score"] = 1 - Q  # Invert so higher = better (for display consistency)
        result["q_value"] = Q
        result["rank"] = ranks
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add CRITIC weights info
    result.attrs["critic_weights"] = dict(zip(criteria_names, critic_w))
    result.attrs["applied_weights"] = dict(zip(criteria_names, weights))
    
    # Sort by rank
    result = result.sort_values("rank")
    
    return result, dict(zip(criteria_names, critic_w)), dict(zip(criteria_names, weights))
