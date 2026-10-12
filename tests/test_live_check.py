"""scripts/live_check.py --wait (run by GO_LIVE.command right after the switch): it waits for a live status written
after the switch by a bot that has run for half a minute, and never takes an old live status or the paper one."""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(monkeypatch):
    spec = importlib.util.spec_from_file_location("live_check", os.path.join(ROOT, "scripts", "live_check.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    clock = [1_800_000_000.0]
    monkeypatch.setattr(m.time, "time", lambda: clock[0])
    monkeypatch.setattr(m.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    return m, clock


def status(path, **kw):
    json.dump({"mode": "live", "ts": 0, "uptime_min": 1.0, "halted": False, "account_fresh": True, **kw},
              open(path, "w"))


def test_waits_for_a_fresh_settled_live_status(tmp_path, monkeypatch):
    m, clock = load(monkeypatch)
    p = str(tmp_path / "live_status.json")
    assert m.wait_for_live(p, 30) is False                                       # nothing written yet
    status(p, ts=clock[0] - 600)
    assert m.wait_for_live(p, 30) is False                                       # an old live run's status
    status(p, ts=clock[0] + 100, mode="paper")
    assert m.wait_for_live(p, 30) is False                                       # the paper bot
    status(p, ts=clock[0] + 100, uptime_min=0.2)
    assert m.wait_for_live(p, 30) is False                                       # started 12 s ago
    status(p, ts=clock[0] + 100)
    assert m.wait_for_live(p, 30) is True


def test_wait_then_check(tmp_path, monkeypatch, capsys):
    m, clock = load(monkeypatch)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    status("data/live_status.json", ts=clock[0] + 5)
    monkeypatch.setattr(sys, "argv", ["live_check.py", "--wait", "60"])
    with pytest.raises(SystemExit) as e:
        m.main()
    out = capsys.readouterr().out
    assert e.value.code == 0 and "mode live" in out and out.rstrip().endswith("OK")
    os.remove("data/live_status.json")
    with pytest.raises(SystemExit) as e:
        monkeypatch.setattr(sys, "argv", ["live_check.py", "--wait", "60"])
        m.main()
    assert e.value.code == 1 and "no live status" in capsys.readouterr().out


def test_a_post_only_fill_on_arrival_needs_a_look(tmp_path, monkeypatch, capsys):
    m, clock = load(monkeypatch)
    monkeypatch.chdir(tmp_path)
    os.makedirs("data")
    status("data/live_status.json", ts=clock[0], post_only_filled_on_arrival=1)
    monkeypatch.setattr(sys, "argv", ["live_check.py", "data/live_status.json"])
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 1 and "post-only is not holding" in capsys.readouterr().out
