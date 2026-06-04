"""UI / Dash-callback regression tests.

These guard the callback layer in ``app.py`` — the part the engine tests don't
touch. Two complementary strategies:

1. A *static* arity guard (``test_callback_return_arity_matches_outputs``) that
   parses ``app.py`` and asserts every multi-output callback returns the exact
   number of values it declares as ``Output``s on *every* return path. This is
   the test that would have caught the original bug where one branch of
   ``update_rankings`` returned 4 values against 6 declared outputs.

2. *Functional* sweeps that actually invoke the decorated callbacks with a
   mocked ``dash.callback_context`` and real data, across every formation/slot
   and every MCDM method, asserting they neither crash nor return the wrong
   shape.
"""
import ast
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402  (imported after path setup; loads the player DB once)


APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


# ─────────────────────────────────────────────────────────────
# 1. Static arity guard — every callback return matches its Output count
# ─────────────────────────────────────────────────────────────

def _is_app_callback(dec):
    return (
        isinstance(dec, ast.Call)
        and isinstance(dec.func, ast.Attribute)
        and dec.func.attr == "callback"
        and isinstance(dec.func.value, ast.Name)
        and dec.func.value.id == "app"
    )


def _count_outputs(decorator):
    return sum(
        1
        for node in ast.walk(decorator)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Output"
    )


def _top_level_returns(func_node):
    """Yield Return nodes in ``func_node``'s body, NOT descending into nested
    function/lambda scopes (whose returns belong to those inner callables)."""
    returns = []

    class _Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            if node is func_node:
                self.generic_visit(node)  # only descend into the target itself

        def visit_AsyncFunctionDef(self, node):
            if node is func_node:
                self.generic_visit(node)

        def visit_Lambda(self, node):
            pass  # never descend into lambdas

        def visit_Return(self, node):
            returns.append(node)

    _Visitor().visit(func_node)
    return returns


def _iter_callbacks():
    tree = ast.parse(open(APP_PATH).read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            decs = [d for d in node.decorator_list if _is_app_callback(d)]
            if decs:
                yield node, _count_outputs(decs[0])


def test_callback_return_arity_matches_outputs():
    """Every multi-output callback must return exactly ``n_outputs`` values on
    every explicit tuple/list return path."""
    problems = []
    for func_node, n_outputs in _iter_callbacks():
        if n_outputs <= 1:
            continue  # single-output callbacks can return a bare value
        for ret in _top_level_returns(func_node):
            val = ret.value
            if isinstance(val, (ast.Tuple, ast.List)):
                arity = len(val.elts)
                if arity != n_outputs:
                    problems.append(
                        f"{func_node.name} (line {ret.lineno}): returns {arity} "
                        f"value(s) but declares {n_outputs} outputs"
                    )
            # Non-tuple returns (bare no_update, single Name/Call) can legitimately
            # stand in for "no update to all outputs" — skip those.
    assert not problems, "Callback output-arity mismatch:\n" + "\n".join(problems)


def test_all_callbacks_have_at_least_one_output():
    """Sanity: the AST walker actually found the callbacks (guards against the
    detection silently matching nothing)."""
    callbacks = list(_iter_callbacks())
    names = {n.name for n, _ in callbacks}
    assert "update_rankings" in names
    assert "handle_player_assignment_and_selection" in names
    assert all(count >= 1 for _, count in callbacks)


# ─────────────────────────────────────────────────────────────
# 2. Functional sweeps — invoke the real callbacks
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def ctx(monkeypatch):
    """Install a mock dash.callback_context into the app module and hand back a
    setter so each test can choose the triggering component."""

    def _set(triggered_prop_id="", inputs_list=None):
        triggered = [{"prop_id": triggered_prop_id}] if triggered_prop_id else []
        fake = SimpleNamespace(triggered=triggered, inputs_list=inputs_list or [])
        monkeypatch.setattr(app, "callback_context", fake)
        return fake

    _set()  # default: no trigger
    return _set


def _all_slots():
    for formation, slots in app.FORMATIONS.items():
        for slot in slots:
            yield formation, slot


def test_update_rankings_returns_6_for_every_slot(ctx):
    """Across every formation and slot, the rankings callback must return its
    full 6-tuple and not raise."""
    for formation, slot in _all_slots():
        out = app.update_rankings(
            slot, "promethee", "critic", [], 0, {}, "", [], None, formation, 200
        )
        assert isinstance(out, tuple) and len(out) == 6, (
            f"{formation}/{slot} returned {type(out).__name__} "
            f"len={len(out) if isinstance(out, tuple) else 'n/a'}"
        )


def test_update_rankings_all_methods(ctx):
    """All nine MCDM methods drive the rankings callback without crashing."""
    methods = ["promethee", "vikor", "ahp", "topsis", "saw",
               "wp", "waspas", "codas", "borda_consensus"]
    for method in methods:
        for weighting in ("critic", "entropy"):
            out = app.update_rankings(
                "ST", method, weighting, [], 0, {}, "", [], None, "4-3-3", 200
            )
            assert isinstance(out, tuple) and len(out) == 6, (
                f"{method}/{weighting} returned bad shape"
            )


def test_update_rankings_empty_position(ctx):
    """No selected position → the empty-state 6-tuple."""
    out = app.update_rankings(
        None, "promethee", "critic", [], 0, {}, "", [], None, "4-3-3", 200
    )
    assert isinstance(out, tuple) and len(out) == 6


def test_update_rankings_no_criteria_branch(ctx, monkeypatch):
    """Regression for the fixed bug: when a slot resolves to an empty criteria
    config, the callback must still return 6 outputs (previously returned 4 and
    crashed Dash with a 'wrong number of outputs' error)."""
    # Find a slot whose role we can blank out.
    role_slot = None
    for formation, slot in _all_slots():
        info = app.SLOT_TO_ROLE.get(slot)
        if info and info["role"] in app.ROLE_CRITERIA:
            role_slot = (formation, slot, info["role"])
            break
    assert role_slot, "expected at least one role-mapped slot"
    formation, slot, role = role_slot

    monkeypatch.setitem(app.ROLE_CRITERIA, role, {})
    out = app.update_rankings(
        slot, "promethee", "critic", [], 0, {}, "", [], None, formation, 200
    )
    assert isinstance(out, tuple) and len(out) == 6


def test_assignment_callback_returns_10(ctx):
    """The assignment/selection callback declares 10 outputs; a no-trigger call
    (e.g. initial render) must still produce all 10."""
    out = app.handle_player_assignment_and_selection(
        [], 0, 0, 200, "4-3-3", None, {}, None, {}
    )
    assert isinstance(out, tuple) and len(out) == 10


def test_method_switch_does_not_rebuild_weight_sliders(ctx):
    """Regression for the method-switch freeze.

    ``update_rankings`` both takes the weight-slider values as Input and outputs
    ``weights-container`` (which builds those sliders). Rebuilding the container
    on a method switch recreates the sliders, whose value re-fires the callback —
    a cascade that, for heavy methods (Borda/PROMETHEE) under rapid switching,
    stacked until the UI froze. A method-selector trigger must therefore emit
    ``no_update`` for the weight outputs (indices 1, 4, 5) while still rebuilding
    the ranking table (index 0).
    """
    import dash

    sliders = [0.15, 0.14, 0.16, 0.14, 0.13, 0.14, 0.14]
    ctx("method-selector.value")
    out = app.update_rankings(
        "CM", "borda_consensus", "critic", sliders, 0, {}, "", [], None, "4-3-3", 200
    )
    assert len(out) == 6
    assert out[0] is not dash.no_update, "ranking table should still refresh on method switch"
    assert out[1] is dash.no_update, "weights-container must NOT rebuild on method switch"
    assert out[4] is dash.no_update and out[5] is dash.no_update


def test_weight_outputs_rebuild_on_position_and_slider(ctx):
    """The weight sliders MUST rebuild on the triggers that change weights —
    position change, weighting change, reset, and a slider drag — otherwise the
    live rebalance and objective-reset behaviour breaks."""
    import dash

    sliders = [0.15, 0.14, 0.16, 0.14, 0.13, 0.14, 0.14]
    for trig in ("store-selected-position.data",
                 "weighting-selector.value",
                 "reset-weights-btn.n_clicks",
                 '{"type":"weight-slider","index":"Goals"}.value'):
        ctx(trig)
        out = app.update_rankings(
            "CM", "saw", "critic", sliders, 0, {}, "", [], None, "4-3-3", 200
        )
        assert out[1] is not dash.no_update, f"weights must rebuild on {trig}"


def test_assignment_then_exclusion_shape(ctx):
    """Assign a player to a slot, then confirm the rankings callback still
    returns 6 outputs when that player is excluded from other positions."""
    # Pick a real player id for an ST slot from the live DB.
    st_info = app.SLOT_TO_ROLE["ST"]
    players = app.get_role_players(app.PLAYER_DB, st_info["pool"], st_info["broad"])
    assert len(players) >= 2
    pid = str(players.iloc[0]["id"])
    assigned = {"ST": "x", "ST_id": pid, "ST_value": 50}

    out = app.update_rankings(
        "LW", "promethee", "critic", [], 0, assigned, "", [], None, "4-3-3", 200
    )
    assert isinstance(out, tuple) and len(out) == 6


# ─────────────────────────────────────────────────────────────
# 3. Callback components must exist in the static layout
# ─────────────────────────────────────────────────────────────

def _collect_layout_ids(component):
    """Recursively gather every string ``id`` in the layout tree."""
    ids = set()

    def walk(c):
        if c is None:
            return
        if isinstance(c, (list, tuple)):
            for x in c:
                walk(x)
            return
        cid = getattr(c, "id", None)
        if isinstance(cid, str):
            ids.add(cid)
        children = getattr(c, "children", None)
        if children is not None:
            walk(children)

    walk(component)
    return ids


def test_callback_input_ids_exist_in_layout():
    """Every plain-string Input/State id referenced by a callback must be present
    in the initial layout.

    Regression for the optimizer Apply/Discard bug: those buttons were created
    only inside the dynamically-rendered preview, so when no preview was on
    screen the dash-renderer logged a stream of "A nonexistent object was used
    in an Input of a Dash callback" errors. Pattern-matching (dict) ids are
    skipped — they are created/destroyed dynamically by design.
    """
    def _is_pattern_id(dep_id):
        """Pattern-matching ids are stored as JSON-object strings; those
        components are created dynamically and are exempt."""
        if not isinstance(dep_id, (str, dict)):
            return True
        if isinstance(dep_id, dict):
            return True
        try:
            return isinstance(json.loads(dep_id), dict)
        except (ValueError, TypeError):
            return False

    layout_ids = _collect_layout_ids(app.app.layout)
    missing = []
    for key, spec in app.app.callback_map.items():
        for kind in ("inputs", "state"):
            for dep in spec.get(kind, []):
                dep_id = dep.get("id")
                if _is_pattern_id(dep_id):
                    continue
                if dep_id not in layout_ids:
                    missing.append(f"{kind} '{dep_id}.{dep.get('property')}' (callback {key})")
    assert not missing, (
        "Callback dependencies reference ids absent from the initial layout:\n"
        + "\n".join(sorted(set(missing)))
    )


def test_optimizer_action_buttons_are_static():
    """The Apply/Discard buttons must live in the static layout (not only in the
    dynamic preview) so their callback Inputs always resolve."""
    layout_ids = _collect_layout_ids(app.app.layout)
    assert "apply-preview-btn" in layout_ids
    assert "discard-preview-btn" in layout_ids
    assert "optimizer-actions" in layout_ids
