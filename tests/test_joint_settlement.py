"""Oct 8 outside review: the maker replay's loop stopped before its last settlement time, so the events that close
last (each run's final day) were never settled: their fills' cost and payout were both left out. The fix settles
whatever is still open at the end. These tests pin the settlement step and the end-of-run sweep."""
import collections
import types

from wxbot.backtest import joint


def _ev(close, result):
    return {"close": close, "result": result, "settled": False}


def test_settle_due_pays_the_winning_side_and_books_the_day():
    events = {"E1": _ev(1_000_000.0, {"T1": "yes", "T2": "no"})}
    positions = {"T1": types.SimpleNamespace(yes_qty=2.0, yes_cost=1.90, no_qty=0.0, no_cost=0.0),
                 "T2": types.SimpleNamespace(yes_qty=0.0, yes_cost=0.0, no_qty=1.0, no_cost=0.97)}
    by_event, settled, daily = {}, {}, collections.defaultdict(float)
    # not due yet: nothing happens
    assert joint._settle_due(events, positions, 1_000_000.0, 8.0, by_event, settled, daily) == 0.0
    assert not events["E1"]["settled"] and len(positions) == 2
    # due: $2 + $1 paid against $2.87 of cost
    pnl = joint._settle_due(events, positions, 1_000_000.0 + 8 * 3600, 8.0, by_event, settled, daily)
    assert abs(pnl - 0.13) < 1e-9 and events["E1"]["settled"] and not positions
    assert abs(sum(daily.values()) - 0.13) < 1e-9 and abs(by_event["E1"] - 0.13) < 1e-9


def test_force_settles_the_last_events_at_the_end_of_the_run():
    """The run's last events: their settle time equals the loop's end, which the loop never reaches."""
    close = 2_000_000.0
    events = {"LAST": _ev(close, {"T9": "yes"})}
    positions = {"T9": types.SimpleNamespace(yes_qty=3.0, yes_cost=2.88, no_qty=0.0, no_cost=0.0)}
    by_event, settled, daily = {}, {}, collections.defaultdict(float)
    t_last_loop = close + 8 * 3600 - 300          # the loop's final step: one step short of the settle time
    assert joint._settle_due(events, positions, t_last_loop, 8.0, by_event, settled, daily) == 0.0
    pnl = joint._settle_due(events, positions, close + 8 * 3600, 8.0, by_event, settled, daily, force=True)
    assert abs(pnl - 0.12) < 1e-9 and events["LAST"]["settled"] and "LAST" in settled


def test_run_joint_sweeps_unsettled_events_after_its_loop():
    import inspect
    src = inspect.getsource(joint.run_joint)
    assert "force=True" in src and src.index("force=True") > src.index("while t < t1")
