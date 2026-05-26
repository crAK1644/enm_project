"""
Football Transfer Window MCDM Application
Main Dash application with clean grey UI.

Run: python app.py
Open: http://localhost:8050
"""

import os
import sys
import json
import re
import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from mcdm.data_processor import build_player_database, get_position_players, get_role_players
from mcdm.criteria import POSITION_CRITERIA, ROLE_CRITERIA, SLOT_TO_ROLE, FORMATIONS
from mcdm.engine import rank_players, METHOD_ALIASES
from mcdm.optimizer import optimize_squad

# ─────────────────────────────────────────────────────────────
# Initialize App
# ─────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="Transfer Window Manager",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"name": "description", "content": "Football Transfer Window MCDM Decision Support System"},
    ],
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Force method dropdown to render block to fix layout */
            html body .header-stat > div#method-selector,
            html body .header-stat > div[id$="-selector"] {
                display: block !important;
            }
            
            /* Specific dark theme styles for the dropdown control and option items */
            html body .dash-dropdown,
            html body .dash-dropdown *,
            html body .Select,
            html body .Select-control,
            html body .Select-menu,
            html body .Select-menu-outer,
            html body .Select-option,
            html body .VirtualizedSelectOption,
            html body [class*="Select"],
            html body [class*="menu"],
            html body [class*="option"],
            html body [class*="Option"],
            html body [class*="Menu"] {
                background-color: #2a2a2f !important;
                background: #2a2a2f !important;
                color: #ececf0 !important;
                border-color: rgba(255, 255, 255, 0.12) !important;
                -webkit-text-fill-color: #ececf0 !important;
            }
            
            html body .Select-option.is-focused,
            html body .VirtualizedSelectFocusedOption,
            html body [class*="option"]:hover,
            html body [class*="option"]:focus,
            html body [class*="Focused"] {
                background-color: #38383f !important;
                background: #38383f !important;
                color: #ececf0 !important;
            }
            
            html body .Select-value-label,
            html body [class*="value-label"],
            html body [class*="singleValue"] {
                color: #ececf0 !important;
            }
            
            html body .Select-arrow,
            html body [class*="indicatorSeparator"],
            html body [class*="dropdownIndicator"] {
                border-top-color: #a0a0aa !important;
                color: #a0a0aa !important;
            }
            
            /* Slider mark labels dark theme fixes */
            html body .rc-slider-mark-text,
            html body .rc-slider-mark-text-active,
            html body .rc-slider-mark span,
            html body .rc-slider-mark div,
            html body .rc-slider span,
            html body .rc-slider div,
            html body .rc-slider-mark-text * {
                color: #a0a0aa !important;
                -webkit-text-fill-color: #a0a0aa !important;
                font-weight: 600 !important;
                font-size: 11px !important;
                font-family: 'JetBrains Mono', monospace !important;
                opacity: 1 !important;
            }
            
            /* Slider rail */
            html body .rc-slider input,
            html body [class*="rc-slider"] input,
            .rc-slider input {
                display: none !important;
            }
            
            /* Slider rail */
            html body .rc-slider-rail,
            html body [class*="rc-slider-rail"] {
                background-color: rgba(255, 255, 255, 0.08) !important;
                background: rgba(255, 255, 255, 0.08) !important;
                height: 4px !important;
            }
            
            /* Slider dots */
            html body .rc-slider-dot,
            html body [class*="rc-slider-dot"] {
                border-color: rgba(255, 255, 255, 0.15) !important;
                background-color: #2a2a2f !important;
            }
            
            /* Numeric Input box and general inputs dark theme styling */
            html body input:not([type="checkbox"]):not([type="radio"]),
            html body select,
            html body textarea {
                background-color: #2a2a2f !important;
                background: #2a2a2f !important;
                color: #ececf0 !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                border-radius: 8px !important;
                outline: none !important;
                box-shadow: none !important;
                -webkit-text-fill-color: #ececf0 !important;
            }
            
            html body #budget-input:focus,
            html body input:focus {
                border: 1px solid #d4845a !important; /* Force entire border property on focus! */
                outline: none !important;
                box-shadow: 0 0 0 2px rgba(212, 132, 90, 0.25) !important;
            }
            
            /* Budget display boxes and percent fixes */
            html body .budget-percent-text {
                color: #a0a0aa !important;
            }
            
            html body .pitch-container {
                overflow: visible !important;
            }
            
            /* Slider tooltip (rc-slider) dark theme overrides */
            html body .rc-slider-tooltip-inner,
            html body [class*="rc-slider-tooltip-inner"],
            html body .rc-slider-tooltip,
            html body [class*="rc-slider-tooltip"],
            html body .rc-slider-tooltip *,
            html body [class*="tooltip"] {
                background-color: #2a2a2f !important;
                background: #2a2a2f !important;
                color: #ececf0 !important;
                border: 1px solid rgba(255, 255, 255, 0.12) !important;
                font-family: 'JetBrains Mono', monospace !important;
                font-size: 11px !important;
                opacity: 1 !important;
                -webkit-text-fill-color: #ececf0 !important;
            }
            
            html body .rc-slider-tooltip-arrow,
            html body [class*="rc-slider-tooltip-arrow"] {
                border-top-color: #2a2a2f !important;
                border-bottom-color: #2a2a2f !important;
            }
            
            /* Slider track and handle overrides to match our design system (accent orange/peach) */
            html body .rc-slider-track,
            html body [class*="rc-slider-track"] {
                background-color: #d4845a !important;
                background: #d4845a !important;
                height: 4px !important;
            }
            
            html body .rc-slider-handle,
            html body [class*="rc-slider-handle"] {
                border-color: #d4845a !important;
                background-color: #222226 !important;
                background: #222226 !important;
                width: 16px !important;
                height: 16px !important;
                margin-top: -6px !important;
            }
            
            /* Hide Dash dev-tools bar (Plotly Cloud / Errors / Callbacks / Server) */
            ._dash-debug-menu,
            ._dash-debug-menu *,
            .dash-debug-menu,
            .dash-debug-menu__outer,
            .dash-debug-menu__content,
            [class*="dash-debug"],
            [class*="_dash-debug"] {
                display: none !important;
                visibility: hidden !important;
                opacity: 0 !important;
                pointer-events: none !important;
                height: 0 !important;
                width: 0 !important;
                overflow: hidden !important;
                position: absolute !important;
                z-index: -9999 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load player database
print("Loading player database...")
PLAYER_DB = build_player_database(PROJECT_DIR, min_minutes=450)
print(f"Database loaded: {len(PLAYER_DB)} players\n")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def fmt_val(val_m):
    """Format market value."""
    if val_m >= 1:
        return f"€{val_m:.1f}m"
    elif val_m > 0:
        return f"€{val_m * 1000:.0f}k"
    return "—"


def rank_class(r):
    if r == 1: return "rank-1"
    if r == 2: return "rank-2"
    if r == 3: return "rank-3"
    return "rank-other"


def normalize_active_weights(weight_map, active_criteria):
    """Normalize active criteria weights so they sum to 1."""
    if not active_criteria:
        return {}

    cleaned = {
        name: max(0.0, float(weight_map.get(name, 0.0)))
        for name in active_criteria
    }
    total = sum(cleaned.values())

    if total <= 0:
        equal = 1.0 / len(active_criteria)
        return {name: equal for name in active_criteria}

    return {name: val / total for name, val in cleaned.items()}


def rebalance_weights_after_change(weight_map, active_criteria, changed_name):
    """Keep the changed slider value and redistribute the rest to sum to 1."""
    if not active_criteria:
        return {}

    if len(active_criteria) == 1:
        return {active_criteria[0]: 1.0}

    if changed_name not in active_criteria:
        return normalize_active_weights(weight_map, active_criteria)

    changed_val = max(0.0, min(1.0, float(weight_map.get(changed_name, 0.0))))
    other_names = [n for n in active_criteria if n != changed_name]
    remaining = 1.0 - changed_val

    rebalanced = {changed_name: changed_val}
    if remaining <= 0:
        for name in other_names:
            rebalanced[name] = 0.0
        return rebalanced

    other_raw = {name: max(0.0, float(weight_map.get(name, 0.0))) for name in other_names}
    other_total = sum(other_raw.values())

    if other_total > 0:
        scale = remaining / other_total
        for name in other_names:
            rebalanced[name] = other_raw[name] * scale
    else:
        equal = remaining / len(other_names)
        for name in other_names:
            rebalanced[name] = equal

    # Guard against tiny floating-point drift so the sum remains exactly 1.
    drift = 1.0 - sum(rebalanced.values())
    anchor = other_names[-1] if other_names else changed_name
    rebalanced[anchor] = max(0.0, rebalanced[anchor] + drift)

    return rebalanced


def build_pitch(formation_key, assigned_players, selected_position):
    """Build pitch with position nodes."""
    formation = FORMATIONS[formation_key]
    children = [
        html.Div(className="pitch-center-dot"),
        html.Div(className="pitch-penalty-top"),
        html.Div(className="pitch-penalty-bottom"),
    ]

    for slot, info in formation.items():
        player = assigned_players.get(slot, "")
        classes = ["position-node"]
        if slot == selected_position:
            classes.append("selected")
        if player:
            classes.append("filled")

        # Strip trailing digits for display: CM1 → CM, CB2 → CB, ST1 → ST
        display_label = re.sub(r'\d+$', '', slot)

        node = html.Div(
            [
                html.Div(
                    player[:7] if player else display_label,
                    className="position-label",
                ),
                html.Div(player, className="position-player-name") if player else None,
            ],
            className=" ".join(classes),
            style={"left": f"{info['x']}%", "top": f"{info['y']}%"},
            id={"type": "position-node", "index": slot},
            key=f"{formation_key}-{slot}",
            n_clicks=0,
        )
        children.append(node)

    return html.Div(children, className="pitch-container", key=formation_key)


METHOD_EXPLANATIONS = {
    "PROMETHEE II": {
        "name": "PROMETHEE II",
        "summary": "Pairwise outranking.",
        "body": ("Compares every player against every other across all criteria, "
                 "then sums their 'wins minus losses' into a net outranking flow Φ. "
                 "Higher Φ means the player beats more rivals more often. "
                 "The most-used outranking method in football MCDM literature."),
    },
    "VIKOR": {
        "name": "VIKOR",
        "summary": "Compromise ranking.",
        "body": ("Finds the player closest to the ideal across all criteria while "
                 "keeping their worst-criterion shortfall small. Balances group "
                 "utility (S = sum of weighted gaps) against individual regret "
                 "(R = the single largest gap). Scores shown are 1 − Q so higher is better."),
    },
    "AHP": {
        "name": "AHP",
        "summary": "Analytic Hierarchy Process.",
        "body": ("Structures the decision into a hierarchy of criteria and alternatives, "
                 "then derives priority weights from pairwise comparison matrices. "
                 "The final score is a weighted sum of normalised criterion values. "
                 "Widely used in management science since Saaty (1980)."),
    },
    "TOPSIS": {
        "name": "TOPSIS",
        "summary": "Distance to ideal & anti-ideal.",
        "body": ("Each player gets a closeness coefficient = distance from the "
                 "anti-ideal ÷ (distance from ideal + distance from anti-ideal). "
                 "Range is 0–1; closer to 1 means closer to the best-possible player "
                 "across the chosen criteria. Most-cited MCDM method in football."),
    },
    "SAW": {
        "name": "SAW",
        "summary": "Simple Additive Weighting.",
        "body": ("Normalises each criterion to 0–1, multiplies by its weight, and "
                 "sums them up. The most intuitive MCDM method — a player's final "
                 "score is simply the weighted average of their normalised stats. "
                 "Fast and transparent, often used as a baseline."),
    },
    "WP": {
        "name": "WP",
        "summary": "Weighted Product Model.",
        "body": ("Each criterion value is raised to the power of its weight, then "
                 "all are multiplied together. Unlike SAW's additive approach, WP "
                 "is multiplicative — a zero on any criterion collapses the score. "
                 "Rewards well-rounded players and penalises extreme weaknesses."),
    },
    "WASPAS": {
        "name": "WASPAS",
        "summary": "Weighted sum + weighted product hybrid.",
        "body": ("Combines a weighted sum (WSM, additive) and a weighted product "
                 "(WPM, multiplicative) at λ = 0.5. The product half penalizes "
                 "players who are weak on any single criterion, so WASPAS rewards "
                 "balanced profiles. Görcün (2021) paired CRITIC + WASPAS for goalkeeper selection."),
    },
    "CODAS": {
        "name": "CODAS",
        "summary": "Combined distance from the anti-ideal.",
        "body": ("Each player is scored by Euclidean distance from the worst-case "
                 "(anti-ideal) plus a Taxicab tie-breaker when two players are nearly "
                 "tied. Higher score = farther from the worst-case. Keshavarz-Ghorabaee "
                 "(2016); gaining traction in sports MCDM since 2020."),
    },
    "Borda Consensus": {
        "name": "Borda Consensus",
        "summary": "Multi-method rank aggregation.",
        "body": ("Runs AHP, TOPSIS, and SAW independently, assigns Borda points "
                 "based on each method's ranking, then sums them into a consensus "
                 "score. Players consistently ranked high across all three methods "
                 "rise to the top — reducing single-method bias."),
    },
}

WEIGHTING_EXPLANATIONS = {
    "critic": ("Weights come from data variability (std dev) × disagreement with "
               "other criteria (1 − correlation). Criteria that discriminate well "
               "and aren't redundant get higher weight."),
    "entropy": ("Weights from Shannon entropy of each criterion's normalized "
                "distribution. The more spread out a criterion's values are across "
                "players, the more information it carries — and the higher its weight."),
}


def build_method_explanation(method, weighting):
    """Plain-English description of the active method + weighting scheme."""
    info = METHOD_EXPLANATIONS.get(method)
    if not info:
        return None
    weight_label = "Entropy" if weighting == "entropy" else "CRITIC"
    weight_body = WEIGHTING_EXPLANATIONS.get(weighting, "")
    return html.Div([
        html.Div([
            html.Span(info["name"], className="method-explainer-name"),
            html.Span(info["summary"], className="method-explainer-summary"),
        ], className="method-explainer-header"),
        html.Div(info["body"], className="method-explainer-body"),
        html.Div([
            html.Span(f"Weights: {weight_label}", className="method-explainer-weight-label"),
            html.Span(weight_body, className="method-explainer-weight-body"),
        ], className="method-explainer-weight"),
    ])


METHOD_EXPLANATIONS = {
    "promethee": {
        "name": "PROMETHEE II",
        "summary": "Pairwise outranking.",
        "body": ("Compares every player against every other across all criteria, "
                 "then sums their 'wins minus losses' into a net outranking flow Φ. "
                 "Higher Φ means the player beats more rivals more often. "
                 "The most-used outranking method in football MCDM literature."),
    },
    "vikor": {
        "name": "VIKOR",
        "summary": "Compromise ranking.",
        "body": ("Finds the player closest to the ideal across all criteria while "
                 "keeping their worst-criterion shortfall small. Balances group "
                 "utility (S = sum of weighted gaps) against individual regret "
                 "(R = the single largest gap). Scores shown are 1 − Q so higher is better."),
    },
    "ahp": {
        "name": "AHP",
        "summary": "Weighted priority synthesis.",
        "body": ("Builds a composite priority for each player from normalized criterion "
                 "scores and criterion weights. Strong all-round profiles rise to the top."),
    },
    "topsis": {
        "name": "TOPSIS",
        "summary": "Distance to ideal & anti-ideal.",
        "body": ("Each player gets a closeness coefficient = distance from the "
                 "anti-ideal ÷ (distance from ideal + distance from anti-ideal). "
                 "Range is 0–1; closer to 1 means closer to the best-possible player "
                 "across the chosen criteria. Most-cited MCDM method in football."),
    },
    "saw": {
        "name": "SAW",
        "summary": "Simple weighted sum.",
        "body": ("Normalizes each criterion, multiplies by weights, then sums. "
                 "Fast, transparent, and useful as a linear baseline."),
    },
    "wp": {
        "name": "WP",
        "summary": "Multiplicative weighted product.",
        "body": ("Multiplies normalized criteria raised to their weights. "
                 "Penalizes players who are weak on any heavily weighted criterion."),
    },
    "waspas": {
        "name": "WASPAS",
        "summary": "Weighted sum + weighted product hybrid.",
        "body": ("Combines a weighted sum (WSM, additive) and a weighted product "
                 "(WPM, multiplicative) at λ = 0.5. The product half penalizes "
                 "players who are weak on any single criterion, so WASPAS rewards "
                 "balanced profiles. Görcün (2021) paired CRITIC + WASPAS for goalkeeper selection."),
    },
    "codas": {
        "name": "CODAS",
        "summary": "Combined distance from the anti-ideal.",
        "body": ("Each player is scored by Euclidean distance from the worst-case "
                 "(anti-ideal) plus a Taxicab tie-breaker when two players are nearly "
                 "tied. Higher score = farther from the worst-case. Keshavarz-Ghorabaee "
                 "(2016); gaining traction in sports MCDM since 2020."),
    },
    "borda_consensus": {
        "name": "Borda Consensus",
        "summary": "Rank aggregation consensus.",
        "body": ("Combines rank positions from multiple MCDM methods into one consensus "
                 "score by awarding points to each rank and summing across methods."),
    },
}

WEIGHTING_EXPLANATIONS = {
    "critic": ("Weights come from data variability (std dev) × disagreement with "
               "other criteria (1 − correlation). Criteria that discriminate well "
               "and aren't redundant get higher weight."),
    "entropy": ("Weights from Shannon entropy of each criterion's normalized "
                "distribution. The more spread out a criterion's values are across "
                "players, the more information it carries — and the higher its weight."),
}


def build_method_explanation(method, weighting):
    """Plain-English description of the active method + weighting scheme."""
    info = METHOD_EXPLANATIONS.get(method)
    if not info:
        return None
    weight_label = "Entropy" if weighting == "entropy" else "CRITIC"
    weight_body = WEIGHTING_EXPLANATIONS.get(weighting, "")
    return html.Div([
        html.Div([
            html.Span(info["name"], className="method-explainer-name"),
            html.Span(info["summary"], className="method-explainer-summary"),
        ], className="method-explainer-header"),
        html.Div(info["body"], className="method-explainer-body"),
        html.Div([
            html.Span(f"Weights: {weight_label}", className="method-explainer-weight-label"),
            html.Span(weight_body, className="method-explainer-weight-body"),
        ], className="method-explainer-weight"),
    ])


def position_indicator_text(formation, slot):
    """Build indicator text for the currently selected slot."""
    if not slot:
        return "Click a position on the pitch"
    role_info = SLOT_TO_ROLE.get(slot)
    if role_info:
        return f"{slot} · {role_info['role']}"
    pos = FORMATIONS.get(formation, {}).get(slot, {}).get("pos", "")
    return f"{slot} · {pos}" if pos else slot


def build_table(ranked_df, alt_ranks=None, search="", budget_filter=False, remaining=9999, selected_player=None):
    """Build ranking table with team, stability badge, search and budget filter."""
    if ranked_df is None or len(ranked_df) == 0:
        return html.Div([
            html.Div("⚽", className="empty-state-icon"),
            html.Div("Select a position on the pitch to see player rankings.",
                     className="empty-state-text"),
        ], className="empty-state")

    df = ranked_df.copy()

    # Apply search filter
    if search and search.strip():
        mask = df["display_name"].str.contains(search.strip(), case=False, na=False)
        df = df[mask]

    # Apply budget filter
    if budget_filter:
        df = df[df["market_value_eur_m"] <= remaining]

    if len(df) == 0:
        return html.Div("No players match your filters.", className="empty-state-text",
                        style={"padding": "24px", "textAlign": "center"})

    scores = df["score"].values
    s_min, s_max = scores.min(), scores.max()
    s_range = s_max - s_min if s_max != s_min else 1

    rows = []
    for _, row in df.head(50).iterrows():
        r = int(row["rank"])
        norm = (row["score"] - s_min) / s_range * 100
        pid = str(row["id"])

        # Stability badge — compare rank in alternate method
        stability_el = html.Span()
        if alt_ranks and pid in alt_ranks:
            delta = alt_ranks[pid] - r
            if delta == 0:
                stability_el = html.Span("=", className="stability-badge stability-stable")
            elif abs(delta) <= 3:
                arrow = "↑" if delta < 0 else "↓"
                stability_el = html.Span(f"{arrow}{abs(delta)}", className="stability-badge stability-ok")
            else:
                arrow = "↑" if delta < 0 else "↓"
                stability_el = html.Span(f"{arrow}{abs(delta)}", className="stability-badge stability-volatile")

        # Team name
        team = str(row.get("team_tm", "")).strip()

        is_selected = (selected_player is not None and str(selected_player) == pid)
        tr_class = "player-tr selected-tr" if is_selected else "player-tr"

        rows.append(
            html.Tr([
                html.Td(html.Span(str(r), className=f"rank-badge {rank_class(r)}")),
                html.Td([
                    html.Div(row["display_name"], className="player-name-cell"),
                    html.Div(team, className="team-label"),
                ]),
                html.Td(fmt_val(row.get("market_value_eur_m", 0)), className="market-value"),
                html.Td(html.Div([
                    html.Div(className="score-bar-fill", style={"width": f"{max(5, norm):.0f}%"})
                ], className="score-bar")),
                html.Td(html.Div([
                    html.Span(f"{row['score']:.3f}",
                              style={"fontSize": "12px", "color": "#a0a0aa",
                                     "fontFamily": "'JetBrains Mono', monospace"}),
                    stability_el,
                ], className="score-cell-container"), className="score-cell"),
            ],
            id={"type": "player-row", "index": pid},
            n_clicks=0,
            className=tr_class,
            )
        )

    return html.Table([
        html.Thead(html.Tr([
            html.Th("#"), html.Th("Player"), html.Th("Value"), html.Th("Score"), html.Th("Δ"),
        ])),
        html.Tbody(rows),
    ], className="ranking-table")


def build_weights(criteria_config, objective_w, applied_w, active_criteria, weighting="critic"):
    """Build weight sliders. objective_w is whatever scheme is active (CRITIC / Entropy)."""
    items = []
    scheme_label = "Entropy" if weighting == "entropy" else "CRITIC"
    for name, info in criteria_config.items():
        cw = objective_w.get(name, 0)
        aw = applied_w.get(name, cw)

        items.append(html.Div([
            html.Div([
                html.Div(info["label"], className="weight-name", style={"flex": "1"}),
                html.Div(f"{aw:.3f}", className="weight-value",
                         id={"type": "weight-display", "index": name}),
            ], className="weight-label"),
            dcc.Slider(
                id={"type": "weight-slider", "index": name},
                min=0, max=1, step=0.01, value=aw,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(f"{scheme_label}: {cw:.3f}",
                     className="weight-scheme-tag"),
        ], className="weight-item"))

    return html.Div(items, className="weight-grid")


def build_player_detail(player_id, position_data):
    """Build player detail panel: radar chart + per-criterion breakdown bars.

    Renders even without a selected player — falls back to position averages
    so the infographic is always visible once a position is chosen.
    """
    if not position_data or not position_data.get("criteria_labels"):
        return html.Div([
            html.Div("Position infographic", className="detail-player-name"),
            html.Div("Select a position to view the radar.", className="detail-hint"),
        ], className="player-detail-content", style={"padding": "16px 18px"})

    avg_scores    = position_data.get("avg", {})
    labels        = position_data.get("criteria_labels", {})
    has_player    = bool(player_id) and player_id in position_data.get("players", {})
    if has_player:
        player_scores = position_data["players"][player_id]
        player_name   = position_data.get("names", {}).get(player_id, "Player")
    else:
        player_scores = avg_scores
        player_name   = "Position average"
    criteria_keys = list(labels.keys())
    criteria_names = [labels[k] for k in criteria_keys]

    p_vals  = [player_scores.get(k, 0) for k in criteria_keys]
    av_vals = [avg_scores.get(k, 0)    for k in criteria_keys]

    # Close radar polygon
    theta   = criteria_names + [criteria_names[0]]
    p_ring  = p_vals + [p_vals[0]]
    av_ring = av_vals + [av_vals[0]]

    fig = go.Figure()
    if has_player:
        fig.add_trace(go.Scatterpolar(
            r=p_ring, theta=theta, fill="toself", name=player_name,
            line=dict(color="#d4845a", width=2),
            fillcolor="rgba(212,132,90,0.18)",
        ))
    fig.add_trace(go.Scatterpolar(
        r=av_ring, theta=theta, fill="toself", name="Position avg",
        line=dict(color="#60a5fa", width=1.5, dash="dot"),
        fillcolor="rgba(96,165,250,0.08)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1.02], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.08)",
                            tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                            showline=False, linecolor="rgba(0,0,0,0)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             linecolor="rgba(255,255,255,0.08)",
                             showline=False,
                             tickfont=dict(color="#a0a0aa", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#a0a0aa", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=-0.12),
        margin=dict(l=115, r=115, t=40, b=40),
        height=290,
    )

    # Criteria breakdown bars
    bar_items = []
    for k in criteria_keys:
        pv = player_scores.get(k, 0)
        av = avg_scores.get(k, 0)
        lbl = labels[k]
        bar_items.append(html.Div([
            html.Div(lbl.replace("<br>", " "), className="breakdown-label"),
            html.Div([
                html.Div(style={"width": f"{pv*100:.0f}%"}, className="breakdown-bar-fill"),
                html.Div(style={"left": f"{av*100:.0f}%"},  className="breakdown-bar-avg"),
            ], className="breakdown-bar"),
            html.Span(f"{pv:.2f}", className="breakdown-val"),
        ], className="breakdown-item"))

    hint = "Click another player to compare" if has_player else "Click a player for their profile"
    return html.Div([
        html.Div([
            html.Div(player_name, className="detail-player-name"),
            html.Div(hint, className="detail-hint"),
        ], className="detail-header"),
        html.Div([
            html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}),
                     className="detail-radar"),
            html.Div(bar_items, className="detail-breakdown"),
        ], className="detail-body"),
    ], className="player-detail-content")


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────

app.layout = html.Div([
    # Stores
    dcc.Store(id="store-assigned-players", data={}),
    dcc.Store(id="store-selected-position", data=None),
    dcc.Store(id="store-rankings-cache", data={}),
    dcc.Store(id="store-budget", data=200),
    dcc.Store(id="store-selected-player", data=None),
    dcc.Store(id="store-position-data", data={}),
    dcc.Store(id="store-min-budget", data=0),
    dcc.Store(id="store-optimizer-preview", data=None),

    # Download
    dcc.Download(id="squad-download"),

    # ── Header ──
    html.Div([
        html.Div([
            html.Span("Transfer Window Manager"),
        ], className="app-title"),

        html.Div([
            html.Div([
                html.Div("Method", className="header-stat-label"),
                dcc.Dropdown(
                    id="method-selector",
                    options=[
                        {"label": "PROMETHEE II", "value": "PROMETHEE II"},
                        {"label": "VIKOR", "value": "VIKOR"},
                        {"label": "AHP", "value": "AHP"},
                        {"label": "TOPSIS", "value": "TOPSIS"},
                        {"label": "SAW", "value": "SAW"},
                        {"label": "WP", "value": "WP"},
                        {"label": "WASPAS", "value": "WASPAS"},
                        {"label": "CODAS", "value": "CODAS"},
                        {"label": "Borda Consensus", "value": "Borda Consensus"},
                    ],
                    value="PROMETHEE II",
                    clearable=False,
                    searchable=False,
                    className="custom-grey-dropdown method-dropdown",
                ),
            ], className="header-stat header-stat-method"),

            html.Div([
                html.Div("Weighting", className="header-stat-label"),
                dcc.RadioItems(
                    id="weighting-selector",
                    options=[
                        {"label": "CRITIC",  "value": "critic"},
                        {"label": "Entropy", "value": "entropy"},
                    ],
                    value="critic",
                    inline=True,
                    className="weighting-segmented",
                ),
            ], className="header-stat"),

            html.Div(className="header-divider"),

            html.Div([
                html.Div("Spent", className="header-stat-label"),
                html.Div("€0.0m", id="spent-display", className="header-stat-value"),
            ], className="header-stat"),

            html.Div([
                html.Div("Remaining", className="header-stat-label"),
                html.Div("€200.0m", id="remaining-display",
                         className="header-stat-value budget-ok"),
            ], className="header-stat"),
        ], className="header-controls"),
    ], className="header-bar"),

    # ── Budget Bar ──
    html.Div([
        html.Div([
            html.Div([
                html.Div("Budget", className="panel-title"),
                html.Div([
                    html.Div([
                        html.Span("Min", className="budget-mini-label"),
                        dcc.Input(
                            id="min-budget-input",
                            type="number",
                            value=0,
                            min=0, max=2000, step=10,
                            style={"width": "62px", "textAlign": "center"},
                            className="budget-mini-input",
                        ),
                    ], className="budget-mini-group",
                       title="Minimum total spend for the optimizer"),
                    html.Div([
                        html.Span("€", className="budget-currency"),
                        dcc.Input(
                            id="budget-input",
                            type="number",
                            value=200,
                            min=10, max=2000, step=10,
                            style={"width": "90px", "textAlign": "center"},
                        ),
                        html.Span("M", className="budget-currency"),
                    ], className="budget-input-group"),
                ], className="budget-input-row"),
            ], className="panel-header"),

            html.Div([
                dcc.Slider(
                    id="budget-slider",
                    min=10, max=2000, step=10, value=200,
                    marks={
                        50: {"label": "50m", "style": {"color": "#a0a0aa", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600", "fontSize": "11px", "opacity": "1"}},
                        200: {"label": "200m", "style": {"color": "#a0a0aa", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600", "fontSize": "11px", "opacity": "1"}},
                        500: {"label": "500m", "style": {"color": "#a0a0aa", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600", "fontSize": "11px", "opacity": "1"}},
                        1000: {"label": "1bn", "style": {"color": "#a0a0aa", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600", "fontSize": "11px", "opacity": "1"}},
                        2000: {"label": "2bn", "style": {"color": "#a0a0aa", "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "600", "fontSize": "11px", "opacity": "1"}},
                    },
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
            ], className="budget-slider-section"),

            html.Div([
                html.Div([
                    html.Div(id="budget-bar-fill",
                             className="budget-progress-fill budget-fill-ok",
                             style={"width": "0%"}),
                ], className="budget-progress"),
                html.Div(id="budget-percent", className="budget-percent-text"),
            ], className="budget-bar-section"),
        ], className="panel"),
    ], className="budget-panel-wrapper"),

    # ── Main Content ──
    html.Div([
        # Left: Pitch
        html.Div([
            html.Div([
                html.Div("Formation", className="panel-title"),
                dcc.RadioItems(
                    id="formation-dropdown",
                    options=[{"label": f, "value": f} for f in FORMATIONS.keys()],
                    value="4-3-3",
                    inline=True,
                    className="formation-radio",
                    labelStyle={"marginRight": "0"},
                ),
            ], className="panel-header formation-header"),
            html.Div([
                html.Div(id="pitch-display"),
                html.Div([
                    html.Div([
                        html.Div("0/11", id="squad-count", className="squad-stat-value"),
                        html.Div("Players", className="squad-stat-label"),
                    ], className="squad-stat"),
                    html.Div([
                        html.Div("€0.0m", id="squad-cost", className="squad-stat-value"),
                        html.Div("Total Cost", className="squad-stat-label"),
                    ], className="squad-stat"),
                    html.Div([
                        html.Button("✕ Remove Player", id="remove-player-btn",
                                    className="btn-remove", n_clicks=0,
                                    style={"display": "none"}),
                        html.Button("Fill Empty", id="fill-empty-btn",
                                    className="btn-secondary", n_clicks=0,
                                    title="Run LP on empty slots, keep your picks"),
                        html.Button("Optimize XI", id="optimize-xi-btn",
                                    className="btn-secondary btn-secondary-accent",
                                    n_clicks=0,
                                    title="Build the best XI from scratch under budget"),
                        html.Button("Clear Squad", id="clear-squad-btn",
                                    className="btn-remove", n_clicks=0),
                        html.Button("Export Squad", id="export-squad-btn",
                                    className="btn-secondary", n_clicks=0),
                    ], className="squad-actions"),
                ], className="squad-summary"),
            ], className="panel-body"),
        ], className="panel"),

        # Right: Rankings
        html.Div([
            html.Div([
                html.Div("Player Rankings", className="panel-title"),
                html.Div([
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search player",
                        debounce=True,
                        className="search-input",
                    ),
                    dcc.Checklist(
                        id="budget-filter",
                        options=[{"label": " Affordable only", "value": "filter"}],
                        value=[],
                        className="budget-filter-check",
                    ),
                ], className="ranking-controls"),
            ], className="panel-header"),
            html.Div([
                html.Div(id="position-indicator", className="position-indicator-bar"),
            ], className="position-bar"),
            html.Div(id="ranking-container", className="panel-body ranking-body"),
            html.Div(id="player-detail-panel"),
        ], className="panel"),
    ], className="main-container"),

    # ── Optimizer Preview Panel (hidden until a run has output) ──
    html.Div(id="optimizer-preview-wrapper", className="optimizer-preview-wrapper"),

    # ── Method Info Panel ──
    html.Div([
        html.Div([
            html.Div([
                html.Div("Method & Weighting", className="panel-title"),
            ], className="panel-header"),
            html.Div(id="method-explanation", className="panel-body method-explainer"),
        ], className="panel method-info-panel"),
    ], className="method-panel-wrapper"),

    # ── Weights Panel ──
    html.Div([
        html.Div([
            html.Div([
                html.Div("Criteria Weights", className="panel-title"),
                html.Button("Reset to CRITIC", id="reset-weights-btn",
                            className="btn-secondary", n_clicks=0,
                            disabled=True),
            ], className="panel-header"),
            html.Div(id="weights-container", className="panel-body"),
        ], className="panel weight-panel"),
    ], style={"padding": "0 28px 28px 28px", "maxWidth": "1500px", "margin": "0 auto"}),
])


# ─────────────────────────────────────────────────────────────
# Optimizer helpers
# ─────────────────────────────────────────────────────────────

def _slot_candidate_frame(slot, formation, method, weighting, exclude_ids):
    """Build the (id, name, price, score) candidate frame for a single slot.

    Mirrors the role-pool + criteria resolution logic of update_rankings, but
    returns a plain candidate DataFrame for the optimizer. ``exclude_ids`` are
    player IDs already locked to other slots and removed from this slot's pool.
    """
    formation_def = FORMATIONS.get(formation, {})
    role_info = SLOT_TO_ROLE.get(slot)
    if role_info:
        criteria_config = ROLE_CRITERIA.get(role_info["role"], {})
        players = get_role_players(PLAYER_DB, role_info["pool"], role_info["broad"])
    else:
        broad = formation_def.get(slot, {}).get("pos", "Forward")
        criteria_config = POSITION_CRITERIA.get(broad, {})
        players = get_position_players(PLAYER_DB, broad)

    if not criteria_config:
        return pd.DataFrame(columns=["id", "name", "price", "score"])

    if exclude_ids:
        players = players[~players["id"].astype(str).isin(exclude_ids)]

    if len(players) < 2:
        return pd.DataFrame(columns=["id", "name", "price", "score"])

    try:
        ranked, _, _ = rank_players(
            players, criteria_config, method=method, weighting=weighting,
        )
    except Exception:
        return pd.DataFrame(columns=["id", "name", "price", "score"])

    team_col = "team_tm" if "team_tm" in ranked.columns else None
    return pd.DataFrame({
        "id":    ranked["id"].astype(str).values,
        "name":  ranked["display_name"].values,
        "team":  (ranked[team_col].fillna("").astype(str).values
                  if team_col else [""] * len(ranked)),
        "price": ranked["market_value_eur_m"].fillna(0).astype(float).values,
        "score": ranked["score"].astype(float).values,
    })


def _build_optimizer_inputs(formation, method, weighting, assigned, fill_only):
    """Return ``(slot_candidates, locked)`` for ``optimize_squad``.

    fill_only=True keeps user picks as locked slots and only optimises the
    empty ones. fill_only=False ignores user picks entirely and optimises
    every slot from scratch.
    """
    formation_def = FORMATIONS.get(formation, {})
    assigned = assigned or {}
    locked: dict[str, dict] = {}

    if fill_only:
        for slot in formation_def:
            pid = assigned.get(f"{slot}_id")
            if pid is None:
                continue
            team = ""
            try:
                row = PLAYER_DB.loc[PLAYER_DB["id"].astype(str) == str(pid)]
                if len(row) and "team_tm" in row.columns:
                    team = str(row.iloc[0]["team_tm"] or "")
            except Exception:
                pass
            locked[slot] = {
                "id":    str(pid),
                "name":  assigned.get(slot, "?"),
                "team":  team,
                "price": float(assigned.get(f"{slot}_value", 0) or 0),
            }

    exclude_ids = {str(p["id"]) for p in locked.values()}

    slot_candidates: dict[str, pd.DataFrame] = {}
    for slot in formation_def:
        if slot in locked:
            continue
        slot_candidates[slot] = _slot_candidate_frame(
            slot, formation, method, weighting, exclude_ids,
        )
    return slot_candidates, locked


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

# Budget sync — slider only
@app.callback(
    [Output("store-budget", "data"),
     Output("budget-value-display", "children")],
    Input("budget-slider", "value"),
    prevent_initial_call=True,
)
def sync_budget(slider_val):
    v = slider_val or 200
    if v >= 1000:
        label = f"€{v / 1000:.1f}bn"
    else:
        label = f"€{v}M"
    return v, label


@app.callback(
    Output("store-min-budget", "data"),
    Input("min-budget-input", "value"),
    prevent_initial_call=True,
)
def sync_min_budget(v):
    try:
        return max(0, float(v or 0))
    except (TypeError, ValueError):
        return 0


# ── Optimizer: run, render preview, apply ──
@app.callback(
    Output("store-optimizer-preview", "data"),
    [Input("fill-empty-btn", "n_clicks"),
     Input("optimize-xi-btn", "n_clicks"),
     Input("formation-dropdown", "value")],
    [State("store-assigned-players", "data"),
     State("store-budget", "data"),
     State("store-min-budget", "data"),
     State("method-selector", "value"),
     State("weighting-selector", "value"),
     State("formation-dropdown", "value")],
    prevent_initial_call=True,
)
def run_optimizer(fill_n, opt_n, _formation_change, assigned, budget_max,
                  budget_min, method, weighting, formation):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Formation change clears any stale preview so the panel collapses.
    if "formation-dropdown" in triggered:
        return None
    if "fill-empty-btn" not in triggered and "optimize-xi-btn" not in triggered:
        return None
    if (fill_n or 0) == 0 and (opt_n or 0) == 0:
        return None

    fill_only = "fill-empty-btn" in triggered
    budget_max = float(budget_max or 200)
    budget_min = float(budget_min or 0)
    if budget_min > budget_max:
        budget_min, budget_max = budget_max, budget_min

    slot_cands, locked = _build_optimizer_inputs(
        formation, method, weighting, assigned, fill_only,
    )
    if not slot_cands and not locked:
        return {"status": "infeasible", "message": "No slots to optimise.",
                "picks": {}, "total_cost": 0.0, "total_score": 0.0,
                "mode": "fill" if fill_only else "full"}

    if not slot_cands and locked:
        # All slots already filled → nothing to do.
        return {"status": "infeasible",
                "message": "All slots are already filled. Try 'Optimize XI'.",
                "picks": {}, "total_cost": 0.0, "total_score": 0.0,
                "mode": "fill"}

    result = optimize_squad(slot_cands, locked=locked,
                            budget_min=budget_min, budget_max=budget_max)
    result["mode"] = "fill" if fill_only else "full"
    result["budget_min"] = budget_min
    result["budget_max"] = budget_max
    return result


@app.callback(
    Output("optimizer-preview-wrapper", "children"),
    Input("store-optimizer-preview", "data"),
    State("formation-dropdown", "value"),
)
def render_optimizer_preview(preview, formation):
    if not preview:
        return None

    mode_label = "Fill Empty" if preview.get("mode") == "fill" else "Optimize XI"

    if preview.get("status") != "optimal":
        return html.Div([
            html.Div([
                html.Div([
                    html.Div("Optimizer", className="panel-title"),
                    html.Span(mode_label, className="optimizer-mode-tag"),
                ], className="panel-header"),
                html.Div([
                    html.Div(preview.get("message", "Infeasible."),
                             className="optimizer-warn-text"),
                    html.Button("Dismiss", id="discard-preview-btn",
                                className="btn-secondary", n_clicks=0),
                ], className="optimizer-infeasible panel-body"),
            ], className="panel"),
        ], className="optimizer-preview-inner")

    formation_def = FORMATIONS.get(formation, {})
    picks = preview.get("picks", {})

    line_order = [("GK",  "Goalkeeper"),
                  ("DEF", "Defender"),
                  ("MID", "Midfielder"),
                  ("FWD", "Forward")]
    lines = {label: [] for label, _ in line_order}
    for slot, slot_def in formation_def.items():
        p = picks.get(slot)
        if not p:
            continue
        broad = slot_def.get("pos", "Forward")
        for label, match in line_order:
            if match == broad:
                lines[label].append((slot, p))
                break

    def _team_abbr(team_name):
        if not team_name:
            return ""
        words = [w for w in team_name.split() if w and w[0].isalpha()]
        if len(words) >= 2:
            return (words[0][0] + words[1][:2]).upper()
        return team_name[:3].upper()

    rows = []
    for line_label, _ in line_order:
        members = lines[line_label]
        if not members:
            continue
        rows.append(html.Tr([
            html.Td(line_label, colSpan=4, className="opt-line-th"),
        ], className="opt-line-row"))
        for slot, p in members:
            display_slot = re.sub(r"\d+$", "", slot)
            locked = bool(p.get("locked"))
            team_abbr = _team_abbr(p.get("team", ""))
            rows.append(html.Tr([
                html.Td(display_slot, className="opt-cell-slot"),
                html.Td([
                    html.Div(p["name"], className="player-name-cell"),
                    html.Div(p.get("team", ""), className="team-label"),
                ]),
                html.Td(team_abbr, className="opt-cell-teamabbr"),
                html.Td([
                    html.Span(f"€{p['price']:.1f}m", className="market-value"),
                    html.Span("LOCKED", className="opt-row-locktag")
                        if locked else None,
                ], className="opt-cell-price"),
            ], className="opt-tr" + (" opt-tr-locked" if locked else "")))

    cost = preview["total_cost"]
    budget_max = preview.get("budget_max", 0) or 0
    budget_min = preview.get("budget_min", 0) or 0
    pct = min(100, (cost / budget_max * 100) if budget_max > 0 else 0)

    return html.Div([
        html.Div([
            html.Div([
                html.Div("Proposed XI", className="panel-title"),
                html.Span(mode_label, className="optimizer-mode-tag"),
            ], className="panel-header"),
            html.Div([
                # Single compact stats row
                html.Div([
                    html.Div([
                        html.Span(f"€{cost:.1f}m", className="opt-summary-cost"),
                        html.Span(f"of €{budget_max:.0f}m",
                                  className="opt-summary-of"),
                    ], className="opt-summary-cost-cell"),
                    html.Div(className="opt-summary-divider"),
                    html.Div([
                        html.Span("Σ Score", className="opt-summary-label"),
                        html.Span(f"{preview['total_score']:.2f}",
                                  className="opt-summary-value"),
                    ], className="opt-summary-stat"),
                    html.Div(className="opt-summary-divider"),
                    html.Div([
                        html.Span("Window", className="opt-summary-label"),
                        html.Span(f"€{budget_min:.0f}m–€{budget_max:.0f}m",
                                  className="opt-summary-value"),
                    ], className="opt-summary-stat"),
                    html.Div([
                        html.Div(className="opt-summary-bar-fill",
                                 style={"width": f"{pct:.0f}%"}),
                    ], className="opt-summary-bar"),
                ], className="opt-summary-row"),

                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Slot"), html.Th("Player"),
                        html.Th("Team"), html.Th("Value"),
                    ])),
                    html.Tbody(rows),
                ], className="ranking-table opt-table"),

                html.Div([
                    html.Button("Apply", id="apply-preview-btn",
                                className="btn-secondary btn-secondary-accent",
                                n_clicks=0),
                    html.Button("Discard", id="discard-preview-btn",
                                className="btn-secondary", n_clicks=0),
                ], className="opt-actions"),
            ], className="panel-body"),
        ], className="panel"),
    ], className="optimizer-preview-inner")


@app.callback(
    Output("store-assigned-players", "data", allow_duplicate=True),
    Output("store-optimizer-preview", "data", allow_duplicate=True),
    Input("apply-preview-btn", "n_clicks"),
    Input("discard-preview-btn", "n_clicks"),
    State("store-optimizer-preview", "data"),
    State("store-assigned-players", "data"),
    State("formation-dropdown", "value"),
    prevent_initial_call=True,
)
def apply_or_discard_preview(apply_n, discard_n, preview, assigned, formation):
    ctx = callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "discard-preview-btn" in triggered:
        return assigned or {}, None

    if "apply-preview-btn" not in triggered or not apply_n:
        return dash.no_update, dash.no_update
    if not preview or preview.get("status") != "optimal":
        return dash.no_update, dash.no_update

    formation_def = FORMATIONS.get(formation, {})
    assigned = dict(assigned or {})
    if preview.get("mode") == "full":
        # Wipe any existing picks for this formation before applying.
        for slot in formation_def:
            for suffix in ("", "_value", "_id"):
                assigned.pop(f"{slot}{suffix}", None)

    for slot, p in preview.get("picks", {}).items():
        assigned[slot]            = p["name"]
        assigned[f"{slot}_value"] = float(p["price"])
        assigned[f"{slot}_id"]    = str(p["id"])

    return assigned, None


# Render pitch
@app.callback(
    Output("pitch-display", "children"),
    [Input("formation-dropdown", "value"),
     Input("store-assigned-players", "data"),
     Input("store-selected-position", "data")],
)
def update_pitch(formation, assigned, selected):
    return build_pitch(formation, assigned or {}, selected)


# Select position — also resets on formation change
@app.callback(
    [Output("store-selected-position", "data"),
     Output("position-indicator", "children")],
    [Input({"type": "position-node", "index": ALL}, "n_clicks"),
     Input("formation-dropdown", "value")],
    [State("store-selected-position", "data")],
)
def select_position(n_clicks, formation, current):
    ctx = callback_context
    if not ctx.triggered:
        return current, position_indicator_text(formation, current)

    triggered_id = ctx.triggered[0]["prop_id"]

    if "formation-dropdown" in triggered_id:
        return None, "Click a position on the pitch"

    if all(n == 0 for n in (n_clicks or [])):
        # Pitch rerenders reset node click counters; preserve the active slot.
        return current, position_indicator_text(formation, current)

    try:
        trigger = json.loads(triggered_id.split(".")[0])
        slot = trigger["index"]
    except (json.JSONDecodeError, KeyError):
        return current, position_indicator_text(formation, current)

    return slot, position_indicator_text(formation, slot)


# Update rankings, weights, position data
@app.callback(
    [Output("ranking-container", "children"),
     Output("weights-container", "children"),
     Output("store-rankings-cache", "data"),
     Output("store-position-data", "data"),
     Output("reset-weights-btn", "children"),
     Output("reset-weights-btn", "disabled")],
    [Input("store-selected-position", "data"),
     Input("method-selector", "value"),
     Input("weighting-selector", "value"),
     Input({"type": "weight-slider", "index": ALL}, "value"),
     Input("reset-weights-btn", "n_clicks"),
     Input("store-assigned-players", "data"),
     Input("search-input", "value"),
     Input("budget-filter", "value"),
     Input("store-selected-player", "data")],
    [State("formation-dropdown", "value"),
     State("store-budget", "data")],
)
def update_rankings(selected_pos, method, weighting, slider_values,
                    reset_clicks, assigned, search, budget_filter_val,
                    selected_player, formation, budget_store):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    assigned = assigned or {}
    budget = budget_store or 200

    empty_rank = html.Div([
        html.Div("⚽", className="empty-state-icon"),
        html.Div("Select a position on the pitch to see player rankings.",
                 className="empty-state-text"),
    ], className="empty-state")
    empty_weight = html.Div([
        html.Div("📊", className="empty-state-icon"),
        html.Div("Weights appear when a position is selected.",
                 className="empty-state-text"),
    ], className="empty-state")

    label = "Entropy" if weighting == "entropy" else "CRITIC"
    btn_text = f"Reset to {label}"

    if not selected_pos:
        return empty_rank, empty_weight, {}, {}, btn_text, True

    # Resolve slot → specific role + eligible pool. Fall back to broad position
    # if the slot is somehow unmapped (defensive — every slot should be in SLOT_TO_ROLE).
    role_info = SLOT_TO_ROLE.get(selected_pos)
    if role_info:
        role_key = role_info["role"]
        pool = role_info["pool"]
        broad = role_info["broad"]
        criteria_config = ROLE_CRITERIA.get(role_key, {})
        players = get_role_players(PLAYER_DB, pool, broad_fallback=broad)
    else:
        broad = FORMATIONS.get(formation, {}).get(selected_pos, {}).get("pos", "Forward")
        criteria_config = POSITION_CRITERIA.get(broad, {})
        players = get_position_players(PLAYER_DB, broad)

    # Resolve slot → specific role + eligible pool. Fall back to broad position
    # if the slot is somehow unmapped (defensive — every slot should be in SLOT_TO_ROLE).
    role_info = SLOT_TO_ROLE.get(selected_pos)
    if role_info:
        role_key = role_info["role"]
        pool = role_info["pool"]
        broad = role_info["broad"]
        criteria_config = ROLE_CRITERIA.get(role_key, {})
        players = get_role_players(PLAYER_DB, pool, broad_fallback=broad)
    else:
        broad = FORMATIONS.get(formation, {}).get(selected_pos, {}).get("pos", "Forward")
        criteria_config = POSITION_CRITERIA.get(broad, {})
        players = get_position_players(PLAYER_DB, broad)

    if not criteria_config:
        return html.Div("No criteria for this position."), html.Div(), {}, {}

    # Filter out players assigned to other positions
    current_id_key = f"{selected_pos}_id"
    assigned_ids = {str(v) for k, v in assigned.items()
                    if k.endswith("_id") and k != current_id_key}
    if assigned_ids:
        players = players[~players["id"].astype(str).isin(assigned_ids)]

    if len(players) < 2:
        return html.Div("Not enough players."), html.Div(), {}, {}, btn_text, True

    criteria_order = list(criteria_config.keys())
    active = criteria_order.copy()

    slider_map = {}
    if slider_values and len(slider_values) == len(criteria_order):
        slider_map = {
            name: float(value if value is not None else 0.0)
            for name, value in zip(criteria_order, slider_values)
        }
    elif slider_values:
        try:
            ids = [json.loads(t["prop_id"].split(".")[0])["index"]
                   for t in ctx.inputs_list[3]] if ctx.inputs_list else []
            if ids and len(ids) == len(slider_values):
                slider_map = {n: float(v if v is not None else 0.0)
                              for n, v in zip(ids, slider_values)}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
            slider_map = {}

    # Custom weights from sliders.
    # Triggers that should DROP slider values and recompute fresh objective weights:
    #   - reset button, position change, weighting-scheme change.
    # Triggers that SHOULD keep current sliders (so unrelated controls don't
    # nuke a tuned weight config): budget, search, assigned-players change.
    # Slider change → rebalance with the moved slider preserved.
    fresh_triggers = ("reset-weights-btn",
                      "store-selected-position",
                      "weighting-selector")
    use_custom = not any(t in triggered_id for t in fresh_triggers)
    custom_weights = None
    if use_custom and slider_map:
        if "weight-slider" in triggered_id:
            try:
                changed_name = json.loads(triggered_id.split(".")[0])["index"]
                custom_weights = rebalance_weights_after_change(slider_map, active, changed_name)
            except (json.JSONDecodeError, KeyError, TypeError):
                custom_weights = slider_map
        else:
            custom_weights = slider_map

    active_config = {k: v for k, v in criteria_config.items() if k in active}

    try:
        ranked_df, objective_w, applied_w = rank_players(
            players, active_config, method=method,
            custom_weights=custom_weights, weighting=weighting,
        )
    except Exception as e:
        return html.Div(f"Error: {e}"), html.Div(), {}, {}, btn_text, True

    # Alternate-method stability badge comparator.
    alt_map = {
        "promethee": "vikor",
        "vikor": "promethee",
        "ahp": "topsis",
        "topsis": "ahp",
        "saw": "wp",
        "wp": "saw",
        "waspas": "promethee",
        "codas": "promethee",
        "borda_consensus": "promethee",
    }
    method_key = METHOD_ALIASES.get(str(method).strip().lower(), str(method).strip().lower())
    alt_method = alt_map.get(method_key, "promethee")
    try:
        alt_ranked, _, _ = rank_players(players, active_config, method=alt_method,
                                        custom_weights=custom_weights, weighting=weighting)
        alt_ranks = {str(row["id"]): int(row["rank"]) for _, row in alt_ranked.iterrows()}
    except Exception:
        alt_ranks = {}

    # Budget remaining for affordable filter
    formation_def = FORMATIONS.get(formation, {})
    cost = sum(assigned.get(f"{k}_value", 0) for k in formation_def)
    remaining = budget - cost
    do_budget_filter = bool(budget_filter_val)

    table = build_table(
        ranked_df,
        alt_ranks=alt_ranks,
        search=search or "",
        budget_filter=do_budget_filter,
        remaining=remaining,
        selected_player=selected_player,
    )

    # Build position-data store for radar/breakdown
    criteria_names = list(criteria_config.keys())
    columns = [criteria_config[c]["column"] for c in criteria_names]
    types   = np.array([criteria_config[c]["type"] for c in criteria_names])
    matrix  = np.nan_to_num(players[columns].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    norm = np.zeros_like(matrix, dtype=float)
    for j in range(len(criteria_names)):
        col = matrix[:, j]
        lo, hi = col.min(), col.max()
        denom = hi - lo if hi != lo else 1.0
        norm[:, j] = (col - lo) / denom if types[j] == 1 else (hi - col) / denom

    avg_norm = norm.mean(axis=0)
    pos_data = {
        "players": {},
        "avg": {k: float(avg_norm[i]) for i, k in enumerate(criteria_names)},
        "criteria_labels": {k: criteria_config[k]["label"] for k in criteria_names},
        "names": {},
    }
    for idx, (_, row) in enumerate(players.iterrows()):
        pid = str(row["id"])
        pos_data["players"][pid] = {k: float(norm[idx, i]) for i, k in enumerate(criteria_names)}
        pos_data["names"][pid] = row["display_name"]

    cache = {str(row["id"]): {"name": row["display_name"],
                               "value": float(row.get("market_value_eur_m", 0))}
             for _, row in ranked_df.iterrows()}

    weights = build_weights(criteria_config, objective_w, applied_w, active, weighting=weighting)

    is_disabled = True
    if "weight-slider" in triggered_id:
        is_disabled = False
    elif use_custom and custom_weights is not None:
        is_disabled = False

    return table, weights, cache, pos_data, btn_text, is_disabled


# (select_player callback merged into handle_player_assignment_and_selection to avoid race conditions)


# Method + weighting explanation card
@app.callback(
    Output("method-explanation", "children"),
    [Input("method-selector", "value"),
     Input("weighting-selector", "value"),
     Input("store-selected-position", "data")],
)
def update_method_explanation(method, weighting, selected_pos):
    return build_method_explanation(method, weighting)



# Render player detail panel
@app.callback(
    Output("player-detail-panel", "children"),
    [Input("store-selected-player", "data"),
     Input("store-position-data", "data")],
)
def update_player_detail(player_id, position_data):
    if not position_data:
        return None
    return build_player_detail(player_id, position_data)


# Show/hide and label the remove-player button
@app.callback(
    [Output("remove-player-btn", "style"),
     Output("remove-player-btn", "children")],
    [Input("store-selected-position", "data"),
     Input("store-assigned-players", "data")],
)
def update_remove_btn(selected_pos, assigned):
    assigned = assigned or {}
    if selected_pos and selected_pos in assigned:
        name = assigned[selected_pos][:14]
        return {"display": "inline-flex"}, f"✕ {name}"
    return {"display": "none"}, "✕ Remove"


# Assign / remove / clear players + update budget display + select player details (combined to prevent race conditions)
@app.callback(
    [Output("store-assigned-players", "data"),
     Output("store-selected-player", "data"),
     Output("spent-display", "children"),
     Output("remaining-display", "children"),
     Output("remaining-display", "className"),
     Output("squad-count", "children"),
     Output("squad-cost", "children"),
     Output("budget-bar-fill", "style"),
     Output("budget-bar-fill", "className"),
     Output("budget-percent", "children")],
    [Input({"type": "player-row", "index": ALL}, "n_clicks"),
     Input("clear-squad-btn", "n_clicks"),
     Input("remove-player-btn", "n_clicks"),
     Input("store-budget", "data"),
     Input("formation-dropdown", "value"),
     Input("store-selected-position", "data")],
    [State("store-assigned-players", "data"),
     State("store-selected-player", "data"),
     State("store-rankings-cache", "data")],
)
def handle_player_assignment_and_selection(player_clicks, clear_clicks, remove_clicks, budget, formation, selected_pos,
                                          assigned, current_player, cache):
    ctx = callback_context
    assigned = assigned or {}
    budget = budget or 200
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    new_selected_player = current_player

    if "formation-dropdown" in triggered_id:
        assigned = {}
        new_selected_player = None

    elif "clear-squad-btn" in triggered_id:
        assigned = {}
        new_selected_player = None

    elif "store-selected-position" in triggered_id:
        # Clear selected player details when moving to another position node
        new_selected_player = None

    elif "remove-player-btn" in triggered_id and selected_pos:
        assigned = {k: v for k, v in assigned.items()
                    if not k.startswith(selected_pos)}
        new_selected_player = None

    elif "player-row" in triggered_id and selected_pos and cache:
        # Table re-render re-creates player-row components with n_clicks=0,
        # which Dash treats as an input change. Ignore those phantom triggers.
        if not player_clicks or all((n or 0) == 0 for n in player_clicks):
            pass
        else:
            try:
                pid = json.loads(triggered_id.split(".")[0])["index"]
                info = cache.get(pid, {})
                if info:
                    assigned[selected_pos]            = info.get("name", "?")
                    assigned[f"{selected_pos}_value"] = info.get("value", 0)
                    assigned[f"{selected_pos}_id"]    = pid

                    # Direct click toggles details or selects player instantly:
                    if current_player is not None and str(current_player) == pid:
                        new_selected_player = None
                    else:
                        new_selected_player = pid
            except (json.JSONDecodeError, KeyError):
                pass

    formation_def = FORMATIONS.get(formation, {})
    total_slots = len(formation_def)
    filled  = sum(1 for k in formation_def if k in assigned)
    cost    = sum(assigned.get(f"{k}_value", 0) for k in formation_def)
    remaining = budget - cost
    pct = min(100, (cost / budget * 100)) if budget > 0 else 0

    if remaining < 0:
        cls     = "header-stat-value budget-over"
        bar_cls = "budget-progress-fill budget-fill-over"
    elif remaining < budget * 0.2:
        cls     = "header-stat-value budget-warn"
        bar_cls = "budget-progress-fill budget-fill-warn"
    else:
        cls     = "header-stat-value budget-ok"
        bar_cls = "budget-progress-fill budget-fill-ok"

    return (
        assigned,
        new_selected_player,
        f"€{cost:.1f}m",
        f"€{remaining:.1f}m",
        cls,
        f"{filled}/{total_slots}",
        f"€{cost:.1f}m",
        {"width": f"{pct:.0f}%"},
        bar_cls,
        f"{pct:.0f}%",
    )



# Export squad to CSV
@app.callback(
    Output("squad-download", "data"),
    Input("export-squad-btn", "n_clicks"),
    [State("store-assigned-players", "data"),
     State("formation-dropdown", "value")],
    prevent_initial_call=True,
)
def export_squad(n_clicks, assigned, formation):
    if not assigned or not formation:
        return None
    formation_def = FORMATIONS.get(formation, {})
    rows = []
    for slot, info in formation_def.items():
        if slot in assigned:
            rows.append({
                "Slot":             slot,
                "Position":         info["pos"],
                "Player":           assigned[slot],
                "Market Value (€m)": assigned.get(f"{slot}_value", 0),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return dcc.send_data_frame(df.to_csv, f"squad_{formation}.csv", index=False)


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ⚽ Transfer Window Manager")
    print("=" * 50)
    print("\n  → http://localhost:8050\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
