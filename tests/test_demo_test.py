"""The two-day demo test (Oct 4: no real money until the bot has traded on Kalshi's demo exchange for at least two
days and scripts/demo_gate.py passes). The demo bot runs beside the main bot on the same Mac with the same .env, so:
its alerts say DEMO and it sends no heartbeat; the weather cache they share is written whole; the gate judges the
demo record against the demo account with the demo key only; and GO_LIVE stays locked until the gate has passed."""
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import importlib.util  # noqa: E402

from test_stage2_gate import Account, write_record  # noqa: E402

from wxbot.app.alerts import Alerter  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ alerts
def test_the_demo_bot_sends_no_heartbeat_and_the_main_bot_still_does(monkeypatch):
    monkeypatch.setenv("WXBOT_NTFY_TOPIC", "my-topic")
    monkeypatch.delenv("WXBOT_HEARTBEAT_TOPIC", raising=False)
    assert Alerter({}).heartbeat_topic.startswith("wxbot-hb-")
    demo = Alerter({}, demo=True)
    assert demo.enabled and demo.heartbeat_topic == "" and demo.heartbeat("alive") is False
    monkeypatch.setenv("WXBOT_HEARTBEAT_TOPIC", "hb")
    assert Alerter({}, demo=True).heartbeat_topic == ""


def test_alert_titles_name_the_demo_bot_demo():
    from wxbot.app.runner import Bot
    b = Bot.__new__(Bot)
    b.cfg, b.mode = {"env": "demo", "mode": "live"}, "live"
    assert b.alert_name == "DEMO"
    b.cfg = {"env": "prod"}
    assert b.alert_name == "live"
    b.mode = "paper"
    assert b.alert_name == "paper"
    assert Bot.__new__(Bot).alert_name == "paper"          # a half-built bot (tests) still has a name


def test_a_demo_halt_alert_says_demo_not_live(tmp_path, monkeypatch):
    from test_review_round2 import Acct, make_bot
    bot = make_bot(tmp_path, monkeypatch, Acct())
    bot.cfg["env"] = "demo"
    sent = []
    bot.alerts.topic = "t"
    bot.alerts._post = lambda title, msg, prio, tags: sent.append(title) or True
    bot.halt("test halt", kind="pause", pause_s=60)
    assert sent and all("DEMO" in t and "live" not in t for t in sent)


# ------------------------------------------------------------------------------------------------ shared cache
def test_the_shared_weather_cache_is_never_seen_half_written(tmp_path, monkeypatch):
    import builtins
    import datetime as dt
    import wxbot.data.nbm as nbm
    monkeypatch.setattr(nbm, "CACHE", str(tmp_path))
    body = ("x" * 200_000).encode()
    monkeypatch.setattr(nbm.requests, "get", lambda url, timeout=60: type("R", (), {"status_code": 200,
                                                                                   "content": body})())
    cyc = dt.datetime(2026, 10, 5, 1)
    final = os.path.join(str(tmp_path), f"nbs_{cyc:%Y%m%d%H}.txt")
    opened_for_write, real_open = [], builtins.open

    def spy(path, mode="r", *a, **kw):
        if "w" in mode:
            opened_for_write.append(str(path))
        return real_open(path, mode, *a, **kw)
    monkeypatch.setattr(builtins, "open", spy)
    assert nbm.fetch("nbs", cyc) is not None
    assert final not in opened_for_write and os.path.getsize(final) == 200_000      # written beside, then renamed
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_pruning_tolerates_a_file_the_other_bot_removed_first(tmp_path, monkeypatch):
    import wxbot.data.nbm as nbm
    monkeypatch.setattr(nbm, "CACHE", str(tmp_path))
    for n in ("a.txt", "b.txt"):
        (tmp_path / n).write_text("x")
        os.utime(tmp_path / n, (0, 0))
    real = os.path.getmtime

    def gone(p):
        if p.endswith("a.txt"):
            raise FileNotFoundError(p)
        return real(p)
    monkeypatch.setattr(nbm.os.path, "getmtime", gone)
    nbm.prune_cache(keep_hours=1)
    assert not (tmp_path / "b.txt").exists()


# ------------------------------------------------------------------------------------------------ activity
def activity_bot(tmp_path):
    from collections import defaultdict
    from wxbot.app.runner import Bot
    from wxbot.store.db import Store
    b = Bot.__new__(Bot)
    b.base = str(tmp_path)
    b.store, b.next_due = Store(str(tmp_path / "demo.sqlite")), defaultdict(float)
    return b


def test_order_activity_adds_up_and_survives_a_restart(tmp_path, monkeypatch):
    clock = [1_800_000_000.0]
    import wxbot.app.runner as runner
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    (tmp_path / ".deployed_commit").write_text("abc1234 (code as abc1234)\n")
    b = activity_bot(tmp_path)
    b.add_activity({"placed": 3, "refreshed": 5, "cancelled": 1, "kept": 9})
    b.next_due["activity_save"] = clock[0] + 300
    clock[0] += 10
    b.add_activity({"placed": 1, "amended": 2})                         # not due yet: kept in memory
    assert b.activity["placed"] == 4 and b.store.kv_get("live_activity")["placed"] == 3
    assert b.activity["code"] == "abc1234" and b.activity["alive_s"] == 10.0
    b.next_due["activity_save"] = 0
    clock[0] += 10
    b.add_activity({})
    clock[0] += 3600                                                    # down for an hour (restart, asleep)
    b2 = activity_bot(tmp_path)
    b2.add_activity({"placed": 2})
    assert b2.activity["placed"] == 6 and b2.activity["refreshed"] == 5 and b2.activity["amended"] == 2
    assert b2.activity["alive_s"] == 20.0                               # the hour away does not count
    assert "kept" not in b2.activity


def test_new_code_starts_the_count_again(tmp_path, monkeypatch):
    clock = [1_800_000_000.0]
    import wxbot.app.runner as runner
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    (tmp_path / ".deployed_commit").write_text("abc1234 (code as abc1234)\n")
    b = activity_bot(tmp_path)
    for _ in range(5):
        clock[0] += 10
        b.add_activity({"placed": 1})
    b.next_due["activity_save"] = 0
    b.add_activity({})
    (tmp_path / ".deployed_commit").write_text("def5678 (code as def5678; docs)\n")
    b2 = activity_bot(tmp_path)
    clock[0] += 10
    b2.add_activity({"placed": 1})
    assert b2.activity["code"] == "def5678" and b2.activity["placed"] == 1 and b2.activity["alive_s"] == 0.0


def test_demo_alerts_send_you_to_the_demo_bot_never_the_real_one():
    from wxbot.app.runner import Bot
    b = Bot.__new__(Bot)
    b.cfg, b.mode = {"env": "demo"}, "live"
    assert b.hint("reset-halt") == ".venv/bin/python -m wxbot reset-halt --config config/demo.yaml"
    assert b.hint("stop") == "DEMO_STOP.command" and "data/demo/demo_status.json" in b.hint("check")
    b.cfg = {"env": "prod"}
    assert b.hint("reset-halt") == ".venv/bin/python -m wxbot reset-halt" and b.hint("stop") == "BACK_TO_PAPER.command"


def test_a_demo_halt_reminder_names_the_demo_config_and_stop_button(tmp_path, monkeypatch):
    from test_review_round2 import Acct, make_bot
    bot = make_bot(tmp_path, monkeypatch, Acct())
    bot.cfg["env"] = "demo"
    sent = []
    bot.alerts.topic = "t"
    bot.alerts._post = lambda title, msg, prio, tags: sent.append((title, msg)) or True
    bot.halt("drawdown test", kind="hard")
    bot.next_due["halt_reminder"] = 0
    import wxbot.app.runner as runner
    monkeypatch.setattr(runner, "to_et", lambda t: type("T", (), {"hour": 12})())
    bot.remind_halt()
    msg = [m for t, m in sent if "still halted" in t][0]
    assert "--config config/demo.yaml" in msg and "DEMO_STOP.command" in msg and "BACK_TO_PAPER" not in msg


# ------------------------------------------------------------------------------------------------ the gate
def demo_record(tmp_path, activity=None, **kw):
    """write_record's live record, moved to where config/demo.yaml keeps the demo bot's files."""
    t0 = write_record(tmp_path, **kw)
    d = tmp_path / "data"
    os.makedirs(d / "demo", exist_ok=True)
    os.replace(d / "live.sqlite", d / "demo" / "demo.sqlite")
    os.replace(d / "live.log", d / "demo" / "demo.log")
    os.replace(d / "live_status.json", d / "demo" / "demo_status.json")
    os.makedirs(tmp_path / "config", exist_ok=True)
    shutil.copy(os.path.join(ROOT, "config", "demo.yaml"), tmp_path / "config" / "demo.yaml")
    (tmp_path / ".deployed_commit").write_text("abc1234 (code as abc1234)\n")
    import sqlite3
    c = sqlite3.connect(d / "demo" / "demo.sqlite")
    c.execute("UPDATE events_log SET msg = msg || ' code=abc1234' WHERE msg LIKE 'start mode=live%'")
    c.commit()
    if activity is not None:
        from wxbot.store.db import Store
        Store(str(d / "demo" / "demo.sqlite")).kv_set("live_activity", activity)
    return t0


def run_demo_gate(tmp_path, monkeypatch, account, argv=("demo_gate.py",), key=True):
    gate = load_script("demo_gate")
    monkeypatch.chdir(tmp_path)
    import wxbot.kalshi.auth as auth
    import wxbot.kalshi.rest as rest
    urls = []
    monkeypatch.setattr(gate, "demo_key_env", lambda: key)
    monkeypatch.setattr(auth.KalshiAuth, "from_env", classmethod(lambda cls: object()))
    monkeypatch.setattr(rest, "KalshiREST", lambda **kw: urls.append(kw.get("base_url")) or account)
    with pytest.raises(SystemExit) as e:
        gate.main(list(argv))
    return e.value.code, urls


GOOD = {"code": "abc1234", "alive_s": 50 * 3600.0, "placed": 40, "refreshed": 120, "cancelled": 12, "amended": 3,
        "refused": 0}


def test_two_clean_demo_days_pass_against_the_demo_exchange_and_unlock_go_live(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity=GOOD)
    code, urls = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert code == 0, out
    from wxbot.kalshi.rest import DEMO
    assert urls == [DEMO]                                   # the demo exchange, never production
    assert "8. 48 hours running on the code GO_LIVE will run" in out and "9. the order path was exercised" in out
    rec = json.load(open(tmp_path / "data" / "demo" / "DEMO_GATE_PASSED"))
    assert rec["hours"] == 50.0 and rec["code"] == "abc1234"
    assert ["9. the order path was exercised", "PASS"] in rec["items"]


def test_too_little_activity_fails_and_leaves_go_live_locked(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity={**GOOD, "placed": 4, "refreshed": 1, "cancelled": 0})
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert code == 1 and "[FAIL ] 9. the order path was exercised" in out
    assert not (tmp_path / "data" / "demo" / "DEMO_GATE_PASSED").exists()


def test_clock_time_is_not_running_time(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity={**GOOD, "alive_s": 30 * 3600.0})    # 73 h since start, 30 h running
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert code == 1 and "[FAIL ] 8. 48 hours running on the code GO_LIVE will run: 30.0 h" in out


def test_two_days_on_other_code_do_not_count(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity={**GOOD, "code": "0ld0ld0"})
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert code == 1 and "[FAIL ] 8." in out and "restart the demo bot" in out
    (tmp_path / ".deployed_commit").write_text("def5678 (code as def5678)\n")    # deployed code it never ran
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    assert code == 1 and "has not run code def5678" in capsys.readouterr().out


def test_a_later_failure_removes_an_earlier_pass(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity=GOOD)
    assert run_demo_gate(tmp_path, monkeypatch, Account(t0))[0] == 0
    marker = tmp_path / "data" / "demo" / "DEMO_GATE_PASSED"
    assert marker.exists()
    import sqlite3
    c = sqlite3.connect(tmp_path / "data" / "demo" / "demo.sqlite")
    c.execute("INSERT INTO events_log (ts, level, msg) VALUES (?, 'ERROR', 'HALT: drawdown stop hit')", (t0 + 7200,))
    c.commit()
    capsys.readouterr()
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    assert code == 1 and not marker.exists() and "GO_LIVE is locked again" in capsys.readouterr().out


def test_the_demo_gate_judges_the_demo_bot_only(tmp_path, monkeypatch):
    demo_record(tmp_path, activity=GOOD)
    gate = load_script("demo_gate")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        gate.main(["demo_gate.py", "--config", "config/live_go.yaml"])
    assert "no --config" in str(e.value.code)


def test_no_demo_fills_yet_is_a_check_not_a_pass(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity=GOOD, with_fill=False)
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0, no_fills=True))
    out = capsys.readouterr().out
    assert code == 1 and "DEMO GATE (the 2-day demo test): CHECK" in out
    assert not (tmp_path / "data" / "demo" / "DEMO_GATE_PASSED").exists()


def one_night(tmp_path, activity, halts=()):
    """One ET day of quoting (yesterday) after a start well before it, whatever the time of day the test runs at."""
    t0 = demo_record(tmp_path, activity=activity, quote_days=1, halts=halts)
    import datetime as dt
    import sqlite3
    from zoneinfo import ZoneInfo
    # the start goes 6 hours before yesterday's ET midnight (a fixed shift fails late in the evening, when it lands
    # inside yesterday after its quoting began)
    y = (dt.datetime.now(ZoneInfo("America/New_York")) - dt.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                                                        microsecond=0)
    c = sqlite3.connect(tmp_path / "data" / "demo" / "demo.sqlite")
    c.execute("UPDATE events_log SET ts = ? WHERE msg LIKE 'start mode=live%'", (y.timestamp() - 6 * 3600,))
    c.commit()
    return t0


SHORT = ("demo_gate.py", "--accept-short")


def test_one_night_passes_only_when_shourya_accepts_the_short_demo(tmp_path, monkeypatch, capsys):
    t0 = one_night(tmp_path, {**GOOD, "alive_s": 20 * 3600.0})
    marker = tmp_path / "data" / "demo" / "DEMO_GATE_PASSED"
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert code == 1 and not marker.exists() and "[FAIL ] 1." in out and "[FAIL ] 8." in out
    code, urls = run_demo_gate(tmp_path, monkeypatch, Account(t0), argv=SHORT)
    out = capsys.readouterr().out
    from wxbot.kalshi.rest import DEMO
    assert code == 0 and urls == [DEMO], out
    assert "Short demo (--accept-short, Shourya's decision): ACCEPTED: 1 full ET day(s), 20.0 h" in out
    assert "Demo test passed (SHORT)" in out
    rec = json.load(open(marker))
    assert rec["short"] is True and rec["hours"] == 20.0 and rec["code"] == "abc1234"
    assert rec["short_note"].startswith("Shourya's decision (Oct 7): live after one night on demo")
    assert ["1. two trading days, no unexplained halt", "FAIL"] in rec["items"]     # the record says what fell short
    gate = load_script("demo_gate")
    info = gate.pass_info(str(marker))
    assert info.startswith("OK ") and "20.0 h on demo (SHORT: Shourya decided one night is enough)" in info
    (tmp_path / ".deployed_commit").write_text("def5678 (code as def5678)\n")     # new code: the pass is old
    assert gate.pass_info(str(marker)).startswith("OLD ")


def test_two_full_days_pass_without_the_short_mark(tmp_path, monkeypatch, capsys):
    t0 = demo_record(tmp_path, activity=GOOD)
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0), argv=SHORT)
    out = capsys.readouterr().out
    rec = json.load(open(tmp_path / "data" / "demo" / "DEMO_GATE_PASSED"))
    assert code == 0 and "Short demo" not in out and "short" not in rec


@pytest.mark.parametrize("activity,halts,why", [
    ({**GOOD, "alive_s": 10 * 3600.0}, (), "10.0 h of actual running; the short demo needs 18"),
    ({**GOOD, "alive_s": 20 * 3600.0, "placed": 4}, (), "not every other item passed: 9. the order path"),
    ({**GOOD, "alive_s": 20 * 3600.0}, ("HALT: drawdown stop hit",), "an unexplained halt or step failures"),
])
def test_the_short_demo_still_needs_a_clean_night(tmp_path, monkeypatch, capsys, activity, halts, why):
    t0 = one_night(tmp_path, activity, halts)
    marker = tmp_path / "data" / "demo" / "DEMO_GATE_PASSED"
    os.makedirs(marker.parent, exist_ok=True)
    marker.write_text(json.dumps({"code": "abc1234", "hours": 50.0}))        # an earlier pass does not survive it
    code, _ = run_demo_gate(tmp_path, monkeypatch, Account(t0), argv=SHORT)
    out = capsys.readouterr().out
    assert code == 1 and "not accepted" in out and why in out, out
    assert not marker.exists() and "GO_LIVE is locked again" in out


def test_the_short_demo_needs_a_full_day_and_no_step_failures():
    gate = load_script("demo_gate")
    ok = [("1. two", "FAIL", ""), ("2. x", "PASS", ""), ("8. 48", "FAIL", ""), ("9. y", "PASS", "")]
    good = {"full_days": ["2026-10-06"], "bad_halts": [], "steps": 0}
    assert gate.short_ok(ok, {"alive_h": 19.0}, good)[0] is True
    assert gate.short_ok(ok, {"alive_h": 19.0}, {**good, "full_days": []}) == (
        False, "0 ET day(s) with six hours of quoting; the short demo needs 1")
    assert gate.short_ok(ok, {"alive_h": 19.0}, {**good, "steps": 2})[0] is False
    assert gate.short_ok(ok[:1] + ok[2:3], {"alive_h": 19.0}, good)[0] is False            # nothing else was checked
    assert gate.short_ok(ok + [("6. z", "CHECK", "")], {"alive_h": 19.0}, good)[0] is False


def test_the_balance_check_wants_a_funded_readable_demo_account(capsys):
    gate = load_script("demo_gate")
    acct = type("A", (), {"balance": lambda self: {"balance": 0}})()
    assert gate.balance_check(acct) == 1 and "too little" in capsys.readouterr().out
    acct = type("A", (), {"balance": lambda self: {"balance": 62500}})()
    assert gate.balance_check(acct) == 0 and "$625.00" in capsys.readouterr().out

    def refused(self):
        raise RuntimeError("401 Unauthorized")
    assert gate.balance_check(type("A", (), {"balance": refused})()) == 2
    assert "Key ID" in capsys.readouterr().out


def test_the_demo_key_goes_in_the_environment_and_the_real_one_never_does(tmp_path, monkeypatch):
    gate = load_script("demo_gate")
    for k in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_PRIVATE_KEY", "KALSHI_BASE_URL"):
        monkeypatch.setenv(k, "")       # recorded first, so what demo_key_env sets is undone after the test
        monkeypatch.delenv(k)
    kd = tmp_path / "deploy_keys"
    kd.mkdir()
    assert gate.demo_key_env(str(kd)) is False
    (kd / "kalshi_demo.key").write_text("DEMO KEY")
    (kd / "kalshi_demo_key_id.txt").write_text("demo-id\n")
    (kd / "kalshi_live.key").write_text("REAL KEY")
    assert gate.demo_key_env(str(kd)) is True
    assert os.environ["KALSHI_KEY_ID"] == "demo-id"
    assert os.environ["KALSHI_PRIVATE_KEY_PATH"].endswith("kalshi_demo.key")
    (kd / "kalshi_live.key").write_text("DEMO KEY")                     # the "demo" file is the real key
    with pytest.raises(SystemExit):
        gate.demo_key_env(str(kd))


# ------------------------------------------------------------------------------------------------ the scripts
SCRIPTS = ["DEMO_START.command", "DEMO_STOP.command", "DEMO_RUN.command", "GO_LIVE.command", "scripts/demo_bot.sh",
           "scripts/vm_demo.sh", "scripts/vm_setup.sh", "scripts/vm_trouble.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_scripts_parse(name):
    assert subprocess.run(["bash", "-n", os.path.join(ROOT, name)]).returncode == 0


def test_the_demo_service_plist_is_valid_and_runs_the_demo_wrapper():
    xml = open(os.path.join(ROOT, "deploy", "launchd", "com.wxbot.demo.plist")).read().replace("__REPO__", "/r")
    p = plistlib.loads(xml.encode())
    assert p["Label"] == "com.wxbot.demo" and p["ProgramArguments"] == ["/bin/bash", "/r/scripts/demo_bot.sh"]
    assert p["KeepAlive"] is True and p["StandardErrorPath"].startswith("/r/data/demo/")


def fake_repo(tmp_path):
    repo = tmp_path / "weather_trader" / "kalshi-wx-bot"
    os.makedirs(repo / ".venv" / "bin")
    os.makedirs(repo / "scripts")
    os.symlink(sys.executable, repo / ".venv" / "bin" / "python")
    shutil.copy(os.path.join(ROOT, "GO_LIVE.command"), repo)
    for f in ("demo_bot.sh", "demo_gate.py"):
        shutil.copy(os.path.join(ROOT, "scripts", f), repo / "scripts")
    return repo


def test_go_live_stays_locked_until_the_demo_test_passed_on_the_deployed_code(tmp_path):
    repo = fake_repo(tmp_path)

    def go():
        return subprocess.run(["bash", str(repo / "GO_LIVE.command")], input="\n", capture_output=True, text=True,
                              timeout=60)
    r = go()
    assert r.returncode == 1 and "Not yet: nothing was changed" in r.stdout and "DEMO_START.command" in r.stdout
    assert not (repo / ".env").exists()
    os.makedirs(repo / "data" / "demo")
    json.dump({"passed_at": "2026-10-08 09:00", "hours": 50.2, "code": "abc1234"},
              open(repo / "data" / "demo" / "DEMO_GATE_PASSED", "w"))
    (repo / ".deployed_commit").write_text("fff0000 (code as def5678; docs)\n")
    r = go()
    assert r.returncode == 1 and "passed on other code" in r.stdout and "deployed now: def5678" in r.stdout
    (repo / ".deployed_commit").write_text("fff0000 (code as abc1234; docs)\n")
    r = go()
    assert "Demo test passed: 2026-10-08 09:00 after 50.2 h on demo, code abc1234" in r.stdout
    assert "No key file found" in r.stdout                  # past the lock, on to the usual steps


def test_the_demo_service_refuses_the_real_key(tmp_path):
    repo = fake_repo(tmp_path)
    kd = tmp_path / "weather_trader" / "deploy_keys"
    kd.mkdir()
    (kd / "kalshi_demo.key").write_text("SAME")
    (kd / "kalshi_demo_key_id.txt").write_text("id\n")
    (kd / "kalshi_live.key").write_text("SAME")
    r = subprocess.run(["bash", str(repo / "scripts" / "demo_bot.sh")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "REAL key" in r.stderr
    (kd / "kalshi_demo_key_id.txt").unlink()
    r = subprocess.run(["bash", str(repo / "scripts" / "demo_bot.sh")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 1 and "no demo key" in r.stderr


# ------------------------------------------------------------------------------------------------ DEMO_START / STOP
def demo_kit(tmp_path, listed=False):
    """A copy of the repo's demo scripts with a fake python (it logs each call and its key ID) and a fake launchctl."""
    root = tmp_path / "weather_trader"
    repo = root / "kalshi-wx-bot"
    for d in ("scripts", "deploy/launchd", "data/demo", ".venv/bin"):
        os.makedirs(repo / d, exist_ok=True)
    for f in ("DEMO_START.command", "DEMO_STOP.command", "deploy/launchd/com.wxbot.demo.plist"):
        shutil.copy(os.path.join(ROOT, f), repo / f)
    kd = root / "deploy_keys"
    kd.mkdir()
    (kd / "kalshi_demo.key").write_text("DEMO KEY")
    (kd / "kalshi_demo_key_id.txt").write_text("demo-id\n")
    log = tmp_path / "calls.log"
    (repo / ".venv/bin/python").write_text(
        "#!/bin/bash\n"
        f'echo "KEY=$KALSHI_KEY_ID $*" >> "{log}"\n'
        f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        'if [ "$1" = "scripts/demo_gate.py" ]; then exit "${BAL_RC:-0}"; fi\n'
        'exit 0\n')
    os.chmod(repo / ".venv/bin/python", 0o755)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "launchctl").write_text(
        "#!/bin/bash\n"
        f'echo "launchctl $*" >> "{log}"\n'
        + ('if [ "$1" = "list" ]; then echo "-\t0\tcom.wxbot.demo"; fi\n' if listed else "")
        + "exit 0\n")
    os.chmod(fakebin / "launchctl", 0o755)
    home = tmp_path / "home"
    os.makedirs(home / "Library" / "LaunchAgents")
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}", "HOME": str(home)}
    for k in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_PRIVATE_KEY", "KALSHI_BASE_URL"):
        env.pop(k, None)
    return repo, log, home, env


def run_cmd(repo, name, env, **extra):
    return subprocess.run(["bash", str(repo / name)], input="\n", capture_output=True, text=True, timeout=60,
                          env={**env, **extra})


def test_demo_start_begins_a_fresh_test_with_the_demo_key(tmp_path):
    repo, log, home, env = demo_kit(tmp_path)
    dd = repo / "data" / "demo"
    for f in ("demo.sqlite", "demo.sqlite-wal", "demo.log", "DEMO_GATE_PASSED", "QUIESCE"):
        (dd / f).write_text("old")
    r = run_cmd(repo, "DEMO_START.command", env)
    assert r.returncode == 0, r.stdout + r.stderr
    old = [d for d in os.listdir(dd) if d.startswith("old-")]
    assert len(old) == 1 and sorted(os.listdir(dd / old[0])) == ["demo.log", "demo.sqlite", "demo.sqlite-wal"]
    assert not (dd / "DEMO_GATE_PASSED").exists() and not (dd / "QUIESCE").exists()
    calls = log.read_text().splitlines()
    py = [c for c in calls if c.startswith("KEY=")]
    assert py and all(c.startswith("KEY=demo-id ") for c in py)                   # the demo key, every time
    assert any("-m wxbot smoke --config config/demo.yaml --series KXHIGHTATL" in c for c in py)
    plist = (home / "Library" / "LaunchAgents" / "com.wxbot.demo.plist").read_text()
    assert f"{repo}/scripts/demo_bot.sh" in plist and "__REPO__" not in plist
    assert any(c.startswith("launchctl load") for c in calls)


@pytest.mark.parametrize("rc,says", [(1, "Fund the demo account first"), (2, "Could not read the demo account")])
def test_demo_start_stops_before_the_order_test_when_the_account_is_not_ready(tmp_path, rc, says):
    repo, log, home, env = demo_kit(tmp_path)
    r = run_cmd(repo, "DEMO_START.command", env, BAL_RC=str(rc))
    calls = log.read_text()
    assert r.returncode == 1 and says in r.stdout and "smoke" not in calls and "launchctl load" not in calls


def test_demo_start_does_not_start_a_second_demo_bot(tmp_path):
    repo, log, home, env = demo_kit(tmp_path, listed=True)
    r = run_cmd(repo, "DEMO_START.command", env)
    assert r.returncode == 0 and "not running" in r.stdout and "launchd.err" in r.stdout      # loaded, no status
    json.dump({"ts": time.time()}, open(repo / "data" / "demo" / "demo_status.json", "w"))
    r = run_cmd(repo, "DEMO_START.command", env)
    assert "already running" in r.stdout
    assert "smoke" not in log.read_text() and "launchctl load" not in log.read_text()


def test_demo_stop_stops_the_service_and_cancels_with_the_demo_key(tmp_path):
    repo, log, home, env = demo_kit(tmp_path, listed=True)
    (home / "Library" / "LaunchAgents" / "com.wxbot.demo.plist").write_text("x")
    r = run_cmd(repo, "DEMO_STOP.command", env)
    calls = log.read_text()
    assert r.returncode == 0 and "launchctl unload" in calls
    assert "KEY=demo-id -m wxbot cancel-all --config config/demo.yaml" in calls
    assert not (home / "Library" / "LaunchAgents" / "com.wxbot.demo.plist").exists()


# ------------------------------------------------------------------------------------------------ a Linux server
def vm_kit(tmp_path, active=False):
    """scripts/vm_demo.sh with a fake python, sudo and systemctl (each logs its call)."""
    root = tmp_path / "weather_trader"
    repo = root / "kalshi-wx-bot"
    for d in ("scripts", "deploy/systemd", "data/demo", ".venv/bin"):
        os.makedirs(repo / d, exist_ok=True)
    for f in ("scripts/vm_demo.sh", "deploy/systemd/wxbot-demo.service"):
        shutil.copy(os.path.join(ROOT, f), repo / f)
    kd = root / "deploy_keys"
    kd.mkdir()
    (kd / "kalshi_demo.key").write_text("DEMO KEY")
    (kd / "kalshi_demo_key_id.txt").write_text("demo-id\n")
    log = tmp_path / "calls.log"
    (repo / ".venv/bin/python").write_text(
        "#!/bin/bash\n"
        f'echo "KEY=$KALSHI_KEY_ID $*" >> "{log}"\n'
        'if [ "$1" = "scripts/demo_gate.py" ]; then exit "${BAL_RC:-0}"; fi\n'
        'if [ "$3" = "smoke" ]; then exit "${SMOKE_RC:-0}"; fi\n'
        "exit 0\n")
    os.chmod(repo / ".venv/bin/python", 0o755)
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "systemctl").write_text(
        "#!/bin/bash\n"
        f'echo "systemctl $*" >> "{log}"\n'
        + ('if [ "$1" = "is-active" ]; then exit 0; fi\n' if active else 'if [ "$1" = "is-active" ]; then exit 3; fi\n')
        + "exit 0\n")
    unit = tmp_path / "wxbot-demo.service"
    (fakebin / "sudo").write_text(
        "#!/bin/bash\n"
        f'echo "sudo $*" >> "{log}"\n'
        f'if [ "$1" = "tee" ]; then exec cat > "{unit}"; fi\n'
        'exec "$@"\n')
    for f in ("systemctl", "sudo"):
        os.chmod(fakebin / f, 0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    for k in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_PRIVATE_KEY", "KALSHI_BASE_URL"):
        env.pop(k, None)
    return repo, log, unit, env


def vm(repo, what, env, **extra):
    return subprocess.run(["bash", str(repo / "scripts" / "vm_demo.sh"), what], capture_output=True, text=True,
                          timeout=60, env={**env, **extra})


def test_vm_demo_start_runs_a_fresh_test_as_a_service_with_the_demo_key(tmp_path):
    repo, log, unit, env = vm_kit(tmp_path)
    dd = repo / "data" / "demo"
    for f in ("demo.sqlite", "demo.log", "DEMO_GATE_PASSED"):
        (dd / f).write_text("old")
    r = vm(repo, "start", env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [d for d in os.listdir(dd) if d.startswith("old-")] and not (dd / "DEMO_GATE_PASSED").exists()
    calls = log.read_text()
    py = [c for c in calls.splitlines() if c.startswith("KEY=")]
    assert py and all(c.startswith("KEY=demo-id ") for c in py)
    assert "-m wxbot smoke --config config/demo.yaml --series KXHIGHTATL" in calls
    assert "systemctl enable --now wxbot-demo" in calls
    text = unit.read_text()
    assert f"WorkingDirectory={repo}" in text and f"ExecStart=/bin/bash {repo}/scripts/demo_bot.sh" in text
    assert "EnvironmentFile" not in [ln.split("=")[0] for ln in text.splitlines() if not ln.startswith("#")]


@pytest.mark.parametrize("rc,says", [(1, "fund the demo account first"), (2, "could not read the demo account")])
def test_vm_demo_start_stops_before_the_order_test_when_the_account_is_not_ready(tmp_path, rc, says):
    repo, log, unit, env = vm_kit(tmp_path)
    r = vm(repo, "start", env, BAL_RC=str(rc))
    assert r.returncode == 1 and says in r.stdout
    assert "smoke" not in log.read_text() and "enable --now" not in log.read_text()


def test_vm_demo_does_not_start_twice_and_stop_cancels_with_the_demo_key(tmp_path):
    repo, log, unit, env = vm_kit(tmp_path, active=True)
    r = vm(repo, "start", env)
    assert "already running" in r.stdout and "smoke" not in log.read_text()
    r = vm(repo, "stop", env)
    calls = log.read_text()
    assert r.returncode == 0 and "systemctl disable --now wxbot-demo" in calls
    assert "KEY=demo-id -m wxbot cancel-all --config config/demo.yaml" in calls


@pytest.mark.parametrize("smoke_rc", [0, 1])
def test_vm_demo_restart_keeps_the_record_and_starts_only_after_the_order_test(tmp_path, smoke_rc):
    """After a code update (Oct 6): stop, order test with the bot stopped, start; nothing archived."""
    repo, log, unit, env = vm_kit(tmp_path, active=True)
    (repo / "data" / "demo" / "demo.sqlite").write_text("record")
    r = vm(repo, "restart", env, SMOKE_RC=str(smoke_rc))
    calls = log.read_text().splitlines()
    assert (repo / "data" / "demo" / "demo.sqlite").read_text() == "record"     # the record carries on
    stop = calls.index("sudo systemctl stop wxbot-demo")
    smoke = next(i for i, c in enumerate(calls) if "-m wxbot smoke --config config/demo.yaml" in c)
    assert stop < smoke and calls[smoke].startswith("KEY=demo-id ")
    if smoke_rc == 0:
        assert r.returncode == 0 and calls.index("sudo systemctl start wxbot-demo") > smoke
    else:
        assert r.returncode == 1 and "stays stopped" in r.stdout
        assert "sudo systemctl start wxbot-demo" not in calls
