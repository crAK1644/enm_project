"""
MCDM Engine: CRITIC weight determination + PROMETHEE II + VIKOR ranking methods.
All implemented from scratch (no external MCDM library needed).
"""

import numpy as np
import pandas as pd


def _saw(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    matrix = df.values
    norm_matrix = np.where(criteria_types == 1, 
                           matrix / np.maximum(1e-10, np.max(matrix, axis=0)), 
                           np.min(matrix, axis=0) / np.maximum(1e-10, matrix))
    scores = np.dot(norm_matrix, weights)
    return pd.Series(scores, index=df.index)

def _wp(df: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray) -> pd.Series:
    epsilon = 1e-5
    matrix = df.values.astype(float)
    col_min = matrix.min(axis=0)
    shift = np.where(col_min < 0, -col_min, 0.0)
    matrix = matrix + shift + epsilon
    norm_matrix = np.where(criteria_types == 1,
                           matrix / np.maximum(1e-10, np.max(matrix, axis=0)),
                           np.min(matrix, axis=0) / np.maximum(1e-10, matrix))
    norm_matrix = np.clip(norm_matrix, 1e-12, None)
    scores = np.prod(norm_matrix ** weights, axis=1)
    return pd.Series(scores, index=df.index)

def calculate_mcdm(method_name: str, matrix: pd.DataFrame, weights: np.ndarray, criteria_types: np.ndarray):
    method_upper = method_name.upper().strip()
    if method_upper == 'BORDA CONSENSUS':
        return _borda_consensus_from_methods(matrix, weights, criteria_types)
        
    methods = {
        'PROMETHEE II': lambda: promethee_ii(matrix.values, weights, criteria_types),
        'VIKOR': lambda: vikor(matrix.values, weights, criteria_types),
        'TOPSIS': lambda: topsis(matrix.values, weights, criteria_types),
        'SAW': lambda: _saw(matrix, weights, criteria_types),
        'WP': lambda: _wp(matrix, weights, criteria_types),
        'WASPAS': lambda: waspas(matrix.values, weights, criteria_types),
        'CODAS': lambda: codas(matrix.values, weights, criteria_types)
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
    base_methods = ['PROMETHEE II', 'VIKOR', 'TOPSIS', 'SAW', 'WP', 'WASPAS', 'CODAS']
    rank_matrix = pd.DataFrame(index=matrix.index)
    for m in base_methods:
        result = calculate_mcdm(m, matrix, weights, criteria_types)
        if m == 'PROMETHEE II':
            scores, _ = result
            ranks = pd.Series(scores, index=matrix.index).rank(ascending=False, method='min').astype(int)
        elif m == 'VIKOR':
            Q, _, _, _ = result
            ranks = pd.Series(Q, index=matrix.index).rank(ascending=True, method='min').astype(int)
        elif m in ('TOPSIS', 'WASPAS', 'CODAS'):
            scores, _ = result
            ranks = pd.Series(scores, index=matrix.index).rank(ascending=False, method='min').astype(int)
        else:
            ranks = result.rank(ascending=False, method='min').astype(int)
        rank_matrix[m] = ranks
    borda_scores = borda_consensus(rank_matrix)
    return borda_scores.reindex(matrix.index)


# ─────────────────────────────────────────────────────────────
# Shannon Entropy: alternative objective weight determination
# ─────────────────────────────────────────────────────────────

def entropy_weights(matrix, types):
    """
    Shannon-entropy objective weights.

    Lower entropy = higher information content = higher weight. Used as an
    objective alternative to CRITIC; the football MCDM literature commonly
    reports CRITIC and Entropy results side by side for sensitivity.
    """
    m, n = matrix.shape

    # Direction-aware normalization so cost criteria contribute correctly.
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if types[j] == 1:
            col_pos = col - col.min() + 1e-12
        else:
            col_pos = col.max() - col + 1e-12
        s = col_pos.sum()
        norm[:, j] = col_pos / s if s > 0 else 1.0 / m

    k = 1.0 / np.log(m) if m > 1 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -k * np.sum(np.where(norm > 0, norm * np.log(norm), 0.0), axis=0)
    ent = np.clip(ent, 0.0, 1.0)
    diversity = 1.0 - ent
    total = diversity.sum()
    return diversity / total if total > 0 else np.ones(n) / n


# ─────────────────────────────────────────────────────────────
# TOPSIS: Distance to ideal / anti-ideal
# ─────────────────────────────────────────────────────────────

def topsis(matrix, weights, types):
    """
    TOPSIS ranking via closeness to the positive-ideal solution.

    Returns closeness coefficient C* in [0,1] (higher = better) and rank positions.
    """
    m, n = matrix.shape

    # Vector (Euclidean) normalization — TOPSIS standard.
    denom = np.sqrt((matrix ** 2).sum(axis=0))
    denom = np.where(denom == 0, 1.0, denom)
    norm = matrix / denom

    weighted = norm * weights

    pis = np.where(types == 1, weighted.max(axis=0), weighted.min(axis=0))
    nis = np.where(types == 1, weighted.min(axis=0), weighted.max(axis=0))

    d_pos = np.sqrt(((weighted - pis) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nis) ** 2).sum(axis=1))

    denom2 = d_pos + d_neg
    closeness = np.where(denom2 > 0, d_neg / denom2, 0.0)

    rank_positions = np.empty(m, dtype=int)
    rank_positions[np.argsort(-closeness)] = np.arange(1, m + 1)
    return closeness, rank_positions


# ─────────────────────────────────────────────────────────────
# WASPAS: Weighted Aggregated Sum-Product
# ─────────────────────────────────────────────────────────────

def waspas(matrix, weights, types, lam=0.5):
    """
    WASPAS = lam * WSM + (1-lam) * WPM on a benefit-direction-normalized matrix.

    Görcün (2021) paired CRITIC + WASPAS for goalkeeper selection — the closest
    published match to this app's setup.
    """
    m, n = matrix.shape

    # Direction-aware linear normalization to [0,1] so WPM exponents stay well-defined.
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if types[j] == 1:
            mx = col.max() if col.max() != 0 else 1.0
            norm[:, j] = col / mx
        else:
            mn = col.min() if col.min() != 0 else 1e-12
            norm[:, j] = mn / np.where(col == 0, 1e-12, col)
    norm = np.clip(norm, 1e-12, None)  # keep strictly positive for WPM

    wsm = (norm * weights).sum(axis=1)
    wpm = np.prod(norm ** weights, axis=1)
    q = lam * wsm + (1 - lam) * wpm

    rank_positions = np.empty(m, dtype=int)
    rank_positions[np.argsort(-q)] = np.arange(1, m + 1)
    return q, rank_positions


# ─────────────────────────────────────────────────────────────
# CODAS: COmbinative Distance-based ASsessment
# ─────────────────────────────────────────────────────────────

def codas(matrix, weights, types, tau=0.02):
    """
    CODAS scores alternatives by their combined Euclidean + Taxicab distance
    from the negative-ideal solution. Taxicab distance is used as a tie-breaker
    via a threshold function with threshold tau (Keshavarz-Ghorabai et al. 2016).
    """
    m, n = matrix.shape

    # Direction-aware linear normalization.
    norm = np.zeros_like(matrix, dtype=float)
    for j in range(n):
        col = matrix[:, j]
        if types[j] == 1:
            mx = col.max() if col.max() != 0 else 1.0
            norm[:, j] = col / mx
        else:
            mn = col.min() if col.min() != 0 else 1e-12
            norm[:, j] = mn / np.where(col == 0, 1e-12, col)

    weighted = norm * weights
    nis = weighted.min(axis=0)

    diff = weighted - nis
    eucl = np.sqrt((diff ** 2).sum(axis=1))
    taxi = np.abs(diff).sum(axis=1)

    def psi(x):
        return 1.0 if abs(x) >= tau else 0.0

    scores = np.zeros(m)
    for i in range(m):
        s = eucl[i]
        for k in range(m):
            if i == k:
                continue
            s += psi(eucl[i] - eucl[k]) * (taxi[i] - taxi[k])
        scores[i] = s

    rank_positions = np.empty(m, dtype=int)
    rank_positions[np.argsort(-scores)] = np.arange(1, m + 1)
    return scores, rank_positions


# ─────────────────────────────────────────────────────────────
# Unified ranking function
# ─────────────────────────────────────────────────────────────

SUPPORTED_METHODS = (
    "promethee", "vikor", "topsis", "saw", "wp", "waspas", "codas", "borda_consensus"
)
SUPPORTED_WEIGHTINGS = ("critic", "entropy", "hybrid")

METHOD_ALIASES = {
    "promethee ii": "promethee",
    "promethee": "promethee",
    "vikor": "vikor",
    "topsis": "topsis",
    "saw": "saw",
    "wp": "wp",
    "waspas": "waspas",
    "codas": "codas",
    "borda consensus": "borda_consensus",
    "borda_consensus": "borda_consensus",
}

def rank_players(player_df, criteria_config, method="promethee", custom_weights=None,
                 weighting="critic", alpha=None):
    """
    Rank players using the specified MCDM method.

    Parameters:
        player_df:       pd.DataFrame  - player data
        criteria_config: dict          - from POSITION_CRITERIA or ROLE_CRITERIA
        method:          str           - one of SUPPORTED_METHODS (aliases accepted)
        custom_weights:  dict or None  - user slider overrides {criteria_name: weight}
        weighting:       str           - 'critic', 'entropy', or 'hybrid'
        alpha:           float | None  - hybrid blending (0=entropy, 1=critic), for backward compatibility

    Returns:
        (ranked_df, objective_weights_dict, applied_weights_dict)
        The objective_weights_dict reflects whichever scheme is active (CRITIC or
        Entropy) so the UI can show "Objective: <name>" against the slider.
    """
    if len(player_df) < 2:
        player_df = player_df.copy()
        player_df["rank"] = 1
        player_df["score"] = 1.0
        return player_df, {}, {}

    criteria_names = list(criteria_config.keys())
    columns = [criteria_config[c]["column"] for c in criteria_names]
    types = np.array([criteria_config[c]["type"] for c in criteria_names])

    matrix = player_df[columns].values.astype(float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    method_key = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    if method_key not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method: {method}")

    if alpha is not None:
        weighting = "hybrid"
    weighting_key = str(weighting).strip().lower()
    if weighting_key not in SUPPORTED_WEIGHTINGS:
        weighting_key = "critic"

    critic_w = critic_weights(matrix, types)
    entropy_w = entropy_weights(matrix, types)
    if weighting_key == "entropy":
        objective_w = entropy_w
    elif weighting_key == "hybrid":
        blend_alpha = 0.5 if alpha is None else alpha
        objective_w = hybridize_weights(critic_w, entropy_w, alpha=blend_alpha)
    else:
        objective_w = critic_w

    if custom_weights:
        weights = np.array([custom_weights.get(c, objective_w[i]) for i, c in enumerate(criteria_names)])
        w_sum = weights.sum()
        weights = weights / w_sum if w_sum > 0 else objective_w
    else:
        weights = objective_w

    result = player_df.copy()

    if method_key == "promethee":
        scores, ranks = promethee_ii(matrix, weights, types)
        result["score"] = scores
    elif method_key == "vikor":
        Q, _, _, ranks = vikor(matrix, weights, types)
        result["score"] = 1 - Q
        result["q_value"] = Q
    elif method_key == "topsis":
        scores, ranks = topsis(matrix, weights, types)
        result["score"] = scores
    elif method_key == "waspas":
        scores, ranks = waspas(matrix, weights, types)
        result["score"] = scores
    elif method_key == "codas":
        scores, ranks = codas(matrix, weights, types)
        result["score"] = scores
    elif method_key == "saw":
        scores = _saw(pd.DataFrame(matrix, index=player_df.index, columns=columns), weights, types)
        result["score"] = scores.values
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores.values)] = np.arange(1, len(scores) + 1)
    elif method_key == "wp":
        scores = _wp(pd.DataFrame(matrix, index=player_df.index, columns=columns), weights, types)
        result["score"] = scores.values
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores.values)] = np.arange(1, len(scores) + 1)
    elif method_key == "borda_consensus":
        matrix_df = pd.DataFrame(matrix, index=player_df.index, columns=columns)
        scores = calculate_mcdm("Borda Consensus", matrix_df, weights, types)
        result["score"] = scores.values
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores.values)] = np.arange(1, len(scores) + 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    result["rank"] = ranks
    result.attrs["objective_weights"] = dict(zip(criteria_names, objective_w))
    result.attrs["applied_weights"] = dict(zip(criteria_names, weights))
    result.attrs["weighting"] = weighting_key
    result = result.sort_values("rank")

    return (result,
            dict(zip(criteria_names, objective_w)),
            dict(zip(criteria_names, weights)))
