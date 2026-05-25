"""
MCDM Engine: CRITIC weight determination + PROMETHEE II + VIKOR ranking methods.
All implemented from scratch (no external MCDM library needed).
"""

import numpy as np
import pandas as pd


def _ahp(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    X = df.values
    benefit_mask = criteria_types == 1
    cost_mask = criteria_types == -1
    norm = np.empty_like(X, dtype=float)
    
    ben_sum = X[:, benefit_mask].sum(axis=0)
    norm[:, benefit_mask] = X[:, benefit_mask] / np.maximum(1e-10, ben_sum)
    
    X_cost = np.where(X[:, cost_mask] == 0, 1e-10, X[:, cost_mask])
    inv_cost = 1.0 / X_cost
    norm[:, cost_mask] = inv_cost / np.maximum(1e-10, inv_cost.sum(axis=0))
    
    scores = norm @ weights
    return pd.Series(scores, index=df.index)

def _topsis(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    matrix = df.values
    norm_matrix = matrix / np.maximum(1e-10, np.sqrt((matrix**2).sum(axis=0)))
    weighted_matrix = norm_matrix * weights
    
    ideal_best = np.where(criteria_types == 1, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
    ideal_worst = np.where(criteria_types == 1, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))
    
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
    
    scores = dist_worst / np.maximum(1e-10, dist_best + dist_worst)
    return pd.Series(scores, index=df.index)

def _saw(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    matrix = df.values
    norm_matrix = np.where(criteria_types == 1, 
                           matrix / np.maximum(1e-10, np.max(matrix, axis=0)), 
                           np.min(matrix, axis=0) / np.maximum(1e-10, matrix))
    scores = np.dot(norm_matrix, weights)
    return pd.Series(scores, index=df.index)

def _wp(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    epsilon = 1e-5
    matrix = df.values + epsilon
    norm_matrix = np.where(criteria_types == 1, 
                           matrix / np.maximum(1e-10, np.max(matrix, axis=0)), 
                           np.min(matrix, axis=0) / np.maximum(1e-10, matrix))
    scores = np.prod(norm_matrix ** weights, axis=1)
    return pd.Series(scores, index=df.index)

def calculate_mcdm(method_name: str, matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray):
    method_upper = method_name.upper().strip()
    if method_upper == 'BORDA CONSENSUS':
        return _borda_consensus_from_methods(matrix, weights, criteria_types)
        
    methods = {
        'PROMETHEE II': lambda: promethee_ii(matrix.values, weights, criteria_types),
        'VIKOR': lambda: vikor(matrix.values, weights, criteria_types),
        'AHP': lambda: _ahp(matrix, weights, criteria_types),
        'TOPSIS': lambda: _topsis(matrix, weights, criteria_types),
        'SAW': lambda: _saw(matrix, weights, criteria_types),
        'WP': lambda: _wp(matrix, weights, criteria_types)
    }
    
    for key in methods:
        if key.upper() == method_upper:
            return methods[key]()
    return methods[method_name]()



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


def shannon_entropy_weights(df, criteria_types, epsilon=1e-12):
    X = df.to_numpy(dtype=float) if isinstance(df, pd.DataFrame) else np.asarray(df, dtype=float)
    t = np.asarray(criteria_types, dtype=float)
    benefit_mask = t == 1
    cost_mask = t == -1
    norm = np.empty_like(X, dtype=float)
    
    ben_sum = X[:, benefit_mask].sum(axis=0)
    norm[:, benefit_mask] = X[:, benefit_mask] / np.maximum(epsilon, ben_sum)
    
    inv_cost = 1.0 / (X[:, cost_mask] + epsilon)
    norm[:, cost_mask] = inv_cost / np.maximum(epsilon, inv_cost.sum(axis=0))
    
    P = norm + epsilon
    m = X.shape[0]
    k = 1.0 / np.log(m) if m > 1 else 1.0
    entropy = -k * np.sum(P * np.log(P), axis=0)
    diversification = 1.0 - entropy
    weights = diversification / np.maximum(epsilon, diversification.sum())
    return weights

def hybridize_weights(critic_w: np.ndarray, shannon_w: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * critic_w + (1.0 - alpha) * shannon_w


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


def borda_consensus(rank_df):
    R = rank_df.to_numpy(dtype=float)
    n_alternatives = R.shape[0]
    points = n_alternatives - R + 1
    borda_scores = points.sum(axis=1)
    return pd.Series(
        borda_scores,
        index=rank_df.index
    ).sort_values(ascending=False)


def _borda_consensus_from_methods(matrix, weights, criteria_types):
    base_methods = ['PROMETHEE II', 'VIKOR', 'AHP', 'TOPSIS', 'SAW', 'WP']
    rank_matrix = pd.DataFrame(index=matrix.index)
    for m in base_methods:
        result = calculate_mcdm(m, matrix, weights, criteria_types)
        if m == 'PROMETHEE II':
            scores, _ = result
            ranks = pd.Series(scores, index=matrix.index).rank(ascending=False, method='min').astype(int)
        elif m == 'VIKOR':
            Q, _, _, _ = result
            ranks = pd.Series(Q, index=matrix.index).rank(ascending=True, method='min').astype(int)
        else:
            ranks = result.rank(ascending=False, method='min').astype(int)
        rank_matrix[m] = ranks
    borda_scores = borda_consensus(rank_matrix)
    return borda_scores.reindex(matrix.index)


# ─────────────────────────────────────────────────────────────
# Unified ranking function
# ─────────────────────────────────────────────────────────────

def rank_players(player_df, criteria_config, method="PROMETHEE II", custom_weights=None, alpha=0.5):
    """
    Rank players using the specified MCDM method.
    
    Parameters:
        player_df:       pd.DataFrame       - player data
        criteria_config: dict               - from POSITION_CRITERIA
        method:          str                - "promethee" or "vikor"
        custom_weights:  dict or None       - custom weights {criteria_name: weight}
        alpha:           float              - compromise weight blending factor (0 = Entropy, 1 = CRITIC)
    
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
    
    # Calculate Shannon Entropy weights
    shannon_w = shannon_entropy_weights(matrix, types)
    
    # Hybridize weights
    hybrid_w = hybridize_weights(critic_w, shannon_w, alpha=alpha)
    
    # Use custom weights if provided, otherwise use hybrid weights
    if custom_weights:
        weights = np.array([custom_weights.get(c, hybrid_w[i]) for i, c in enumerate(criteria_names)])
        # Normalize
        w_sum = weights.sum()
        if w_sum > 0:
            weights = weights / w_sum
        else:
            weights = hybrid_w
    else:
        weights = hybrid_w
    
    # Apply MCDM method
    result = player_df.copy()
    
    matrix_df = pd.DataFrame(matrix, index=player_df.index, columns=columns)
    scores_data = calculate_mcdm(method, matrix_df, weights, types)
    
    # Map the output scores to result columns based on method
    method_normalized = method.upper().replace("_", " ").replace("-", " ")
    if "PROMETHEE" in method_normalized:
        scores, ranks = scores_data
        result["score"] = scores
        result["rank"] = ranks
    elif "VIKOR" in method_normalized:
        Q, S, R, ranks = scores_data
        result["score"] = 1 - Q  # Invert so higher = better
        result["q_value"] = Q
        result["rank"] = ranks
    else:
        result["score"] = scores_data.values
        ranks = np.empty(len(scores_data), dtype=int)
        ranks[np.argsort(-scores_data.values)] = np.arange(1, len(scores_data) + 1)
        result["rank"] = ranks

    
    # Add Hybrid weights info
    result.attrs["critic_weights"] = dict(zip(criteria_names, hybrid_w))
    result.attrs["applied_weights"] = dict(zip(criteria_names, weights))
    
    # Sort by rank
    result = result.sort_values("rank")
    
    return result, dict(zip(criteria_names, hybrid_w)), dict(zip(criteria_names, weights))
