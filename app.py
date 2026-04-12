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


def build_table(ranked_df):
    """Build ranking table."""
    if ranked_df is None or len(ranked_df) == 0:
        return html.Div([
            html.Div("⚽", className="empty-state-icon"),
            html.Div("Select a position on the pitch to see player rankings.",
                     className="empty-state-text"),
        ], className="empty-state")

    scores = ranked_df["score"].values
    s_min, s_max = scores.min(), scores.max()
    s_range = s_max - s_min if s_max != s_min else 1

    rows = []
    for _, row in ranked_df.head(30).iterrows():
        r = int(row["rank"])
        norm = (row["score"] - s_min) / s_range * 100

        rows.append(
            html.Tr([
                html.Td(html.Span(str(r), className=f"rank-badge {rank_class(r)}")),
                html.Td([
                    html.Div(row["display_name"], className="player-name-cell"),
                    html.Div(row.get("full_name", ""), className="team-label"),
                ]),
                html.Td(fmt_val(row.get("market_value_eur_m", 0)), className="market-value"),
                html.Td(html.Div([
                    html.Div(className="score-bar-fill", style={"width": f"{max(5, norm):.0f}%"})
                ], className="score-bar")),
                html.Td(f"{row['score']:.3f}",
                         style={"fontSize": "12px", "color": "#a0a0aa",
                                "fontFamily": "'JetBrains Mono', monospace"}),
            ],
            id={"type": "player-row", "index": str(row["id"])},
            n_clicks=0,
            )
        )

    return html.Table([
        html.Thead(html.Tr([
            html.Th("#"), html.Th("Player"), html.Th("Value"), html.Th("Score"), html.Th(""),
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


# ─────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────

app.layout = html.Div([
    # Stores
    dcc.Store(id="store-assigned-players", data={}),
    dcc.Store(id="store-selected-position", data=None),
    dcc.Store(id="store-rankings-cache", data={}),
    dcc.Store(id="store-budget", data=200),

    # ── Header ──
    html.Div([
        html.Div([
            html.Span("⚽", className="app-title-icon"),
            html.Span("Transfer Window Manager"),
        ], className="app-title"),

        html.Div([
            # Method toggle
            html.Div([
                html.Div("Method", className="header-stat-label"),
                dcc.RadioItems(
                    id="method-selector",
                    options=[
                        {"label": " PROMETHEE", "value": "promethee"},
                        {"label": " VIKOR", "value": "vikor"},
                    ],
                    value="promethee",
                    inline=True,
                    style={"fontSize": "12px"},
                ),
            ], className="header-stat"),

            html.Div(className="header-divider"),

            # Spent
            html.Div([
                html.Div("Spent", className="header-stat-label"),
                html.Div("€0.0m", id="spent-display", className="header-stat-value"),
            ], className="header-stat"),

            # Remaining
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
            # Header row: title + value input
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

            # Slider row (separate so marks don't distort header alignment)
            html.Div([
                dcc.Slider(
                    id="budget-slider",
                    min=10, max=2000, step=10, value=200,
                    marks={
                        50: "€50m", 200: "€200m", 500: "€500m",
                        1000: "€1bn", 2000: "€2bn",
                    },
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], className="budget-slider-section"),

            # Progress bar row
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
                    html.Button("Clear Squad", id="clear-squad-btn",
                                className="btn-remove", n_clicks=0),
                ], className="squad-summary"),
            ], className="panel-body"),
        ], className="panel"),

        # Right: Rankings
        html.Div([
            html.Div([
                html.Div("Player Rankings", className="panel-title"),
                html.Div(id="position-indicator",
                         style={"fontSize": "12px", "color": "#a0a0aa"}),
            ], className="panel-header"),
            html.Div(id="ranking-container", className="panel-body",
                     style={"maxHeight": "500px", "overflowY": "auto"}),
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

# Budget sync: input and slider both write to store, store updates both
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
    if "budget-input" in trigger:
        v = input_val or 200
    else:
        v = slider_val or 200
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


# Select position
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

    # Formation changed — reset selection so stale slot doesn't drive wrong rankings
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


# Update rankings & weights
@app.callback(
    [Output("ranking-container", "children"),
     Output("weights-container", "children"),
     Output("store-rankings-cache", "data")],
    [Input("store-selected-position", "data"),
     Input("method-selector", "value"),
     Input({"type": "weight-slider", "index": ALL}, "value"),
     Input({"type": "criteria-check", "index": ALL}, "value"),
     Input("reset-weights-btn", "n_clicks"),
     Input("store-assigned-players", "data")],
    [State("formation-dropdown", "value")],
)
def update_rankings(selected_pos, method, slider_values, check_values,
                    reset_clicks, assigned, formation):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
    assigned = assigned or {}

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
        return empty_rank, empty_weight, {}

    pos_type = FORMATIONS.get(formation, {}).get(selected_pos, {}).get("pos", "Forward")
    criteria_config = POSITION_CRITERIA.get(pos_type, {})
    if not criteria_config:
        return html.Div("No criteria for this position."), html.Div(), {}

    players = get_position_players(PLAYER_DB, pos_type)

    # ── Filter out players already assigned to OTHER positions ──
    assigned_ids = set()
    current_id_key = f"{selected_pos}_id"
    for key, val in assigned.items():
        if key.endswith("_id") and key != current_id_key:
            assigned_ids.add(str(val))
    if assigned_ids:
        players = players[~players["id"].astype(str).isin(assigned_ids)]

    if len(players) < 2:
        return html.Div("Not enough players."), html.Div(), {}

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
        return html.Div(f"Error: {e}"), html.Div(), {}

    table = build_table(ranked_df)

    cache = {str(row["id"]): {"name": row["display_name"],
                               "value": float(row.get("market_value_eur_m", 0))}
             for _, row in ranked_df.iterrows()}

    # Only rebuild the weights panel when position/method/reset changes,
    # NOT when a slider is dragged (which would snap it to the normalized value).
    if "weight-slider" in triggered_id:
        return table, no_update, cache

    weights = build_weights(criteria_config, critic_w, applied_w, active)
    return table, weights, cache


# Assign players & update budget
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
     Input("store-budget", "data")],
    [State("store-assigned-players", "data"),
     State("store-selected-position", "data"),
     State("store-rankings-cache", "data"),
     State("formation-dropdown", "value")],
)
def assign_player(player_clicks, clear_clicks, budget,
                  assigned, selected_pos, cache, formation):
    ctx = callback_context
    assigned = assigned or {}
    budget = budget or 200
    triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    # Clear
    if "clear-squad-btn" in triggered_id:
        assigned = {}

    # Assign
    elif "player-row" in triggered_id and selected_pos and cache:
        try:
            pid = json.loads(triggered_id.split(".")[0])["index"]
            info = cache.get(pid, {})
            if info:
                assigned[selected_pos] = info.get("name", "?")
                assigned[f"{selected_pos}_value"] = info.get("value", 0)
                assigned[f"{selected_pos}_id"] = pid
        except (json.JSONDecodeError, KeyError):
            pass

    # Compute totals
    formation_def = FORMATIONS.get(formation, {})
    total_slots = len(formation_def)
    filled = sum(1 for k in formation_def if k in assigned)
    cost = sum(assigned.get(f"{k}_value", 0) for k in formation_def)
    remaining = budget - cost
    pct = min(100, (cost / budget * 100)) if budget > 0 else 0

    # Budget status
    if remaining < 0:
        cls = "header-stat-value budget-over"
        bar_cls = "budget-progress-fill budget-fill-over"
    elif remaining < budget * 0.2:
        cls = "header-stat-value budget-warn"
        bar_cls = "budget-progress-fill budget-fill-warn"
    else:
        cls = "header-stat-value budget-ok"
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


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  ⚽ Transfer Window Manager")
    print("=" * 50)
    print("\n  → http://localhost:8050\n")
    app.run(debug=True, host="0.0.0.0", port=8050)
