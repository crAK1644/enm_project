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
from dash import html, dcc, Input, Output, State, callback_context, ALL, no_update
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from mcdm.data_processor import build_player_database, get_position_players
from mcdm.criteria import POSITION_CRITERIA, FORMATIONS
from mcdm.engine import rank_players

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
            n_clicks=0,
        )
        children.append(node)

    return html.Div(children, className="pitch-container")


def build_table(ranked_df, alt_ranks=None, search="", budget_filter=False, remaining=9999):
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
                html.Td([
                    html.Span(f"{row['score']:.3f}",
                              style={"fontSize": "12px", "color": "#a0a0aa",
                                     "fontFamily": "'JetBrains Mono', monospace"}),
                    stability_el,
                ], className="score-cell"),
            ],
            id={"type": "player-row", "index": pid},
            n_clicks=0,
            className="player-tr",
            )
        )

    return html.Table([
        html.Thead(html.Tr([
            html.Th("#"), html.Th("Player"), html.Th("Value"), html.Th("Score"), html.Th("Δ"),
        ])),
        html.Tbody(rows),
    ], className="ranking-table")


def build_weights(criteria_config, critic_w, applied_w, active_criteria):
    """Build weight sliders."""
    items = []
    for name, info in criteria_config.items():
        active = name in active_criteria
        cw = critic_w.get(name, 0)
        aw = applied_w.get(name, cw)

        items.append(html.Div([
            html.Div([
                html.Div([
                    dcc.Checklist(
                        id={"type": "criteria-check", "index": name},
                        options=[{"label": info["label"], "value": name}],
                        value=[name] if active else [],
                        style={"display": "inline-block"},
                    ),
                ], style={"flex": "1"}),
                html.Div(f"{aw:.3f}", className="weight-value",
                         id={"type": "weight-display", "index": name}),
            ], className="weight-label"),
            dcc.Slider(
                id={"type": "weight-slider", "index": name},
                min=0, max=1, step=0.01, value=aw,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
                disabled=not active,
            ),
            html.Div(f"CRITIC: {cw:.3f}",
                     style={"fontSize": "9px", "color": "#6b6b76",
                            "textAlign": "right", "marginTop": "2px"}),
        ], className="weight-item"))

    return html.Div(items, className="weight-grid")


def build_player_detail(player_id, position_data):
    """Build player detail panel: radar chart + per-criterion breakdown bars."""
    if not position_data or player_id not in position_data.get("players", {}):
        return None

    player_scores = position_data["players"][player_id]
    avg_scores    = position_data.get("avg", {})
    labels        = position_data.get("criteria_labels", {})
    player_name   = position_data.get("names", {}).get(player_id, "Player")
    criteria_keys = list(labels.keys())
    criteria_names = [labels[k] for k in criteria_keys]

    p_vals  = [player_scores.get(k, 0) for k in criteria_keys]
    av_vals = [avg_scores.get(k, 0)    for k in criteria_keys]

    # Close radar polygon
    theta   = criteria_names + [criteria_names[0]]
    p_ring  = p_vals + [p_vals[0]]
    av_ring = av_vals + [av_vals[0]]

    fig = go.Figure()
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
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             linecolor="rgba(255,255,255,0.08)",
                             tickfont=dict(color="#a0a0aa", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#a0a0aa", size=10), bgcolor="rgba(0,0,0,0)",
                    orientation="h", y=-0.12),
        margin=dict(l=30, r=30, t=10, b=30),
        height=260,
    )

    # Criteria breakdown bars
    bar_items = []
    for k in criteria_keys:
        pv = player_scores.get(k, 0)
        av = avg_scores.get(k, 0)
        lbl = labels[k]
        bar_items.append(html.Div([
            html.Div(lbl, className="breakdown-label"),
            html.Div([
                html.Div(style={"width": f"{pv*100:.0f}%"}, className="breakdown-bar-fill"),
                html.Div(style={"left": f"{av*100:.0f}%"},  className="breakdown-bar-avg"),
            ], className="breakdown-bar"),
            html.Span(f"{pv:.2f}", className="breakdown-val"),
        ], className="breakdown-item"))

    return html.Div([
        html.Div([
            html.Div(player_name, className="detail-player-name"),
            html.Div("Click row again to close", className="detail-hint"),
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

    # Download
    dcc.Download(id="squad-download"),

    # ── Header ──
    html.Div([
        html.Div([
            html.Span("⚽", className="app-title-icon"),
            html.Span("Transfer Window Manager"),
        ], className="app-title"),

        html.Div([
            html.Div([
                html.Div("Method", className="header-stat-label"),
                dcc.RadioItems(
                    id="method-selector",
                    options=[
                        {"label": " PROMETHEE", "value": "promethee"},
                        {"label": " VIKOR",     "value": "vikor"},
                    ],
                    value="promethee",
                    inline=True,
                    style={"fontSize": "12px"},
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
            ], className="panel-header"),

            html.Div([
                dcc.Slider(
                    id="budget-slider",
                    min=10, max=2000, step=10, value=200,
                    marks={50: "€50m", 200: "€200m", 500: "€500m",
                           1000: "€1bn", 2000: "€2bn"},
                    tooltip={"placement": "bottom", "always_visible": False},
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
                dcc.Dropdown(
                    id="formation-dropdown",
                    options=[{"label": f, "value": f} for f in FORMATIONS.keys()],
                    value="4-3-3",
                    clearable=False,
                    style={"width": "150px"},
                ),
            ], className="panel-header", style={"gap": "16px"}),
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
                        placeholder="Search player…",
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

    # ── Weights Panel ──
    html.Div([
        html.Div([
            html.Div([
                html.Div("Criteria Weights", className="panel-title"),
                html.Button("Reset to CRITIC", id="reset-weights-btn",
                            className="btn-secondary", n_clicks=0),
            ], className="panel-header"),
            html.Div(id="weights-container", className="panel-body"),
        ], className="panel weight-panel"),
    ], style={"padding": "0 28px 28px 28px", "maxWidth": "1500px", "margin": "0 auto"}),
])


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

# Budget sync
@app.callback(
    [Output("store-budget", "data"),
     Output("budget-input", "value"),
     Output("budget-slider", "value")],
    [Input("budget-input", "value"),
     Input("budget-slider", "value")],
    [State("store-budget", "data")],
    prevent_initial_call=True,
)
def sync_budget(input_val, slider_val, current):
    ctx = callback_context
    if not ctx.triggered:
        return current, current, current
    trigger = ctx.triggered[0]["prop_id"]
    v = (input_val or 200) if "budget-input" in trigger else (slider_val or 200)
    return v, v, v


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
        return None, "Click a position on the pitch"

    triggered_id = ctx.triggered[0]["prop_id"]

    if "formation-dropdown" in triggered_id:
        return None, "Click a position on the pitch"

    if all(n == 0 for n in (n_clicks or [])):
        return None, "Click a position on the pitch"

    try:
        trigger = json.loads(triggered_id.split(".")[0])
        slot = trigger["index"]
    except (json.JSONDecodeError, KeyError):
        return current, ""

    pos = FORMATIONS.get(formation, {}).get(slot, {}).get("pos", "")
    return slot, f"{slot} · {pos}"


# Update rankings, weights, position data
@app.callback(
    [Output("ranking-container", "children"),
     Output("weights-container", "children"),
     Output("store-rankings-cache", "data"),
     Output("store-position-data", "data")],
    [Input("store-selected-position", "data"),
     Input("method-selector", "value"),
     Input({"type": "weight-slider", "index": ALL}, "value"),
     Input({"type": "criteria-check", "index": ALL}, "value"),
     Input("reset-weights-btn", "n_clicks"),
     Input("store-assigned-players", "data"),
     Input("search-input", "value"),
     Input("budget-filter", "value")],
    [State("formation-dropdown", "value"),
     State("store-budget", "data")],
)
def update_rankings(selected_pos, method, slider_values, check_values,
                    reset_clicks, assigned, search, budget_filter_val,
                    formation, budget_store):
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

    if not selected_pos:
        return empty_rank, empty_weight, {}, {}

    pos_type = FORMATIONS.get(formation, {}).get(selected_pos, {}).get("pos", "Forward")
    criteria_config = POSITION_CRITERIA.get(pos_type, {})
    if not criteria_config:
        return html.Div("No criteria for this position."), html.Div(), {}, {}

    players = get_position_players(PLAYER_DB, pos_type)

    # Filter out players assigned to other positions
    current_id_key = f"{selected_pos}_id"
    assigned_ids = {str(v) for k, v in assigned.items()
                    if k.endswith("_id") and k != current_id_key}
    if assigned_ids:
        players = players[~players["id"].astype(str).isin(assigned_ids)]

    if len(players) < 2:
        return html.Div("Not enough players."), html.Div(), {}, {}

    active = list(criteria_config.keys())

    # Custom weights from sliders
    custom_weights = None
    use_custom = "reset-weights-btn" not in triggered_id
    if use_custom and slider_values:
        try:
            ids = [json.loads(t["prop_id"].split(".")[0])["index"]
                   for t in ctx.inputs_list[2]] if ctx.inputs_list else []
            if ids and len(ids) == len(slider_values):
                custom_weights = {n: v for n, v in zip(ids, slider_values) if n in active}
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Active criteria from checkboxes
    if check_values:
        try:
            ids = [json.loads(t["prop_id"].split(".")[0])["index"]
                   for t in ctx.inputs_list[3]] if ctx.inputs_list else []
            if ids:
                active = [n for n, v in zip(ids, check_values) if v and n in v]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    if not active:
        active = list(criteria_config.keys())

    active_config = {k: v for k, v in criteria_config.items() if k in active}

    try:
        ranked_df, critic_w, applied_w = rank_players(
            players, active_config, method=method, custom_weights=custom_weights
        )
    except Exception as e:
        return html.Div(f"Error: {e}"), html.Div(), {}, {}

    # Compute alternate method ranks for stability badge
    alt_method = "vikor" if method == "promethee" else "promethee"
    try:
        alt_ranked, _, _ = rank_players(players, active_config, method=alt_method,
                                        custom_weights=custom_weights)
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

    if "weight-slider" in triggered_id:
        return table, no_update, cache, pos_data

    weights = build_weights(criteria_config, critic_w, applied_w, active)
    return table, weights, cache, pos_data


# Select player for detail panel (click again to toggle off)
@app.callback(
    Output("store-selected-player", "data"),
    [Input({"type": "player-row", "index": ALL}, "n_clicks"),
     Input("store-selected-position", "data")],
    [State("store-selected-player", "data")],
    prevent_initial_call=True,
)
def select_player(row_clicks, selected_pos, current_player):
    ctx = callback_context
    if not ctx.triggered:
        return None
    triggered_id = ctx.triggered[0]["prop_id"]

    # Position changed → clear detail panel
    if "store-selected-position" in triggered_id:
        return None

    if "player-row" not in triggered_id:
        return None

    try:
        pid = json.loads(triggered_id.split(".")[0])["index"]
    except (json.JSONDecodeError, KeyError):
        return None

    # Toggle: clicking the already-open player closes the panel
    return None if pid == current_player else pid


# Render player detail panel
@app.callback(
    Output("player-detail-panel", "children"),
    [Input("store-selected-player", "data"),
     Input("store-position-data", "data")],
)
def update_player_detail(player_id, position_data):
    if not player_id or not position_data:
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


# Assign / remove / clear players + update budget display
@app.callback(
    [Output("store-assigned-players", "data"),
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
     Input("formation-dropdown", "value")],
    [State("store-assigned-players", "data"),
     State("store-selected-position", "data"),
     State("store-rankings-cache", "data")],
)
def assign_player(player_clicks, clear_clicks, remove_clicks, budget, formation,
                  assigned, selected_pos, cache):
    ctx = callback_context
    assigned = assigned or {}
    budget = budget or 200
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if "formation-dropdown" in triggered_id:
        assigned = {}

    elif "clear-squad-btn" in triggered_id:
        assigned = {}

    elif "remove-player-btn" in triggered_id and selected_pos:
        assigned = {k: v for k, v in assigned.items()
                    if not k.startswith(selected_pos)}

    elif "player-row" in triggered_id and selected_pos and cache:
        try:
            pid = json.loads(triggered_id.split(".")[0])["index"]
            info = cache.get(pid, {})
            if info:
                assigned[selected_pos]               = info.get("name", "?")
                assigned[f"{selected_pos}_value"]    = info.get("value", 0)
                assigned[f"{selected_pos}_id"]       = pid
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
    app.run(debug=True, host="0.0.0.0", port=8050)
