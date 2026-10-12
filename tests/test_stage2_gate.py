"""scripts/stage2_gate.py on a made-up live record: the log lines the order manager writes, the database and status
file the bot writes, and a fake account for the read-only checks. Round-5 review: the cases that used to pass when
they should fail (stacking 12 s apart, a cancel that never resolves, half an hour of trading) and the ones that used
to fail a clean run (a quick top-up, Kalshi's timestamps on Python 3.9, a stale status file)."""
import importlib.util
import json
import os
import sys
import time

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
T75 = "KXHIGHNY-26OCT05-T75"


def load_gate():
    spec = importlib.util.spec_from_file_location("stage2_gate", os.path.join(ROOT, "scripts", "stage2_gate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".123456Z"     # Kalshi's fractional seconds


def write_record(tmp_path, halts=(), uncertain_after=4.2, pending_after=2.0, pending_resolved=True, fill_price=3,
                 quote_days=3, quote_hours=8, status_age=5.0, taker_fill=False, log_extra=(), with_fill=True):
    from wxbot.store.db import Store
    os.makedirs(tmp_path / "data", exist_ok=True)
    st = Store(str(tmp_path / "data" / "live.sqlite"))
    now = time.time()
    t0 = now - quote_days * 86400 - 3600
    st.c.execute("INSERT INTO events_log (ts, level, msg) VALUES (?, 'INFO', 'start mode=live')", (t0,))
    for h in halts:
        st.c.execute("INSERT INTO events_log (ts, level, msg) VALUES (?, 'ERROR', ?)", (t0 + 3600, h))
    from wxbot.timeutil import to_et
    for d in range(1, quote_days + 1):               # quote snapshots every 5 minutes, quote_hours a day, from 8 AM
        ts = now - d * 86400                         # ET on each of the last quote_days days (never today)
        e = to_et(ts)
        day0 = ts - (e.hour - 8) * 3600 - e.minute * 60 - e.second
        for k in range(int(quote_hours * 12)):
            st.c.execute("INSERT INTO quotes (ts, mode, ticker, event, side, price_cents, count, sleeve, fair_yes, "
                         "reason) VALUES (?, 'live', ?, 'KXHIGHNY-26OCT05', 'ask', 3, 5, 'A', 0.01, '')",
                         (day0 + 300 * k, T75))
    if with_fill:
        st.c.execute("INSERT INTO fills (ts, mode, ticker, event, side, price_cents, count, sleeve, is_taker, fee, "
                     "order_ref) VALUES (?, 'live', ?, 'KXHIGHNY-26OCT05', 'ask', ?, 5, 'A', 0, 0, 'f1')",
                     (t0 + 7200, T75, fill_price))
    if taker_fill:                                   # the taker's own order (wxb-X-tk...) takes by design
        st.c.execute("INSERT INTO fills (ts, mode, ticker, event, side, price_cents, count, sleeve, is_taker, fee, "
                     "order_ref) VALUES (?, 'live', ?, 'KXHIGHNY-26OCT05', 'ask', 3, 2, 'X', 1, 0.01, 'f9')",
                     (t0 + 7300, T75))
    st.c.commit()
    lt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ",000"
    lines = [f"{lt(t0 + 100)} WARNING wxbot.execution.live: placement of {T75} ask @3c x5 has no answer: kept as "
             f"uncertain until a snapshot shows it (cid=wxb-A-aaa)",
             f"{lt(t0 + 105)} WARNING wxbot.execution.live: order o1 ({T75} ask @3c) is resting after all: adopted. "
             f"RESOLVED uncertain adopted after={uncertain_after}s cid=wxb-A-aaa",
             f"{lt(t0 + 200)} ERROR wxbot.execution.live: batch cancel: 1 of 1 not confirmed (no per-order answer); "
             f"kept as pending exposure. PENDING cids=wxb-A-bbb"]
    if pending_resolved:
        lines.append(f"{lt(t0 + 202)} INFO wxbot.execution.live: cancel of o2 confirmed on a retry. RESOLVED "
                     f"cancel_pending confirmed after={pending_after}s cid=wxb-A-bbb")
    lines += [f"{lt(t0 + 300)} {x}" for x in log_extra]
    (tmp_path / "data" / "live.log").write_text("\n".join(lines) + "\n")
    json.dump({"ts": now - status_age, "live_equity_usd": 625.0, "balance_semantics": {"excludes_resting": True}},
              open(tmp_path / "data" / "live_status.json", "w"))
    return t0


class Account:
    """o1: 10 NO at 97c, filled 5 at t0+7200. second: None, 'dup' (a second order 12 s after o1 while o1 rests,
    nothing filled yet) or 'topup' (6 s after o1 had 7 of 10 filled)."""

    def __init__(self, t0, second=None, maker_took=False, taker=False, no_fills=False):
        self.t0 = t0
        o1 = {"order_id": "o1", "client_order_id": "wxb-A-aaa", "ticker": T75, "book_side": "ask",
              "status": "canceled", "created_time": iso(t0 + 100), "last_update_time": iso(t0 + 9000),
              "yes_price_dollars": "0.0300", "remaining_count_fp": "0", "initial_count_fp": "10.00",
              "fill_count_fp": "5.00"}
        self.os, self.fs = [o1], [{"fill_id": "f1", "order_id": "o1", "yes_price_dollars": "0.0300",
                                   "count_fp": "5.00", "created_time": iso(t0 + 7200)}]
        if second == "dup":
            self.os.append({**o1, "order_id": "o2", "client_order_id": "wxb-A-dup", "created_time": iso(t0 + 112),
                            "last_update_time": iso(t0 + 400), "fill_count_fp": "0"})
        if no_fills:
            self.fs = []
            self.os[0]["fill_count_fp"] = "0"
        if second == "no_update_time":               # o1 (20, 5 filled at t0+7200) has no last_update_time;
            self.os[0]["initial_count_fp"] = "20.00" # o3 goes out after o1's last fill
            del self.os[0]["last_update_time"]
            self.os.append({**o1, "order_id": "o3", "client_order_id": "wxb-A-ccc", "created_time": iso(t0 + 8000),
                            "last_update_time": iso(t0 + 8100), "fill_count_fp": "0", "initial_count_fp": "20.00"})
        if maker_took:                               # Kalshi filled the post-only order as a taker
            self.fs[0]["is_taker"] = True
        if taker:
            self.os.append({**o1, "order_id": "o9", "client_order_id": "wxb-X-tk0123456789abcdef",
                            "created_time": iso(t0 + 7290), "last_update_time": iso(t0 + 7300),
                            "initial_count_fp": "2.00", "fill_count_fp": "2.00"})
            self.fs.append({"fill_id": "f9", "order_id": "o9", "yes_price_dollars": "0.0300", "count_fp": "2.00",
                            "created_time": iso(t0 + 7300), "is_taker": True})
        if second == "topup":
            self.fs = [{"fill_id": "f1", "order_id": "o1", "yes_price_dollars": "0.0300", "count_fp": "5.00",
                        "created_time": iso(t0 + 110)}]
            self.os[0]["fill_count_fp"] = "5.00"
            self.os[0]["initial_count_fp"] = "7.00"
            self.os.append({**o1, "order_id": "o3", "client_order_id": "wxb-A-top", "created_time": iso(t0 + 116),
                            "last_update_time": iso(t0 + 900), "fill_count_fp": "0"})

    def paged(self, path, key, params, **kw):
        return list(self.os)

    def fills(self, min_ts=None):
        return list(self.fs)

    def balance(self):
        return {"balance": 62000}

    def orders(self, status="resting"):
        return []

    def positions(self):
        return [{"ticker": T75, "position_fp": "-5", "market_exposure_dollars": "5.00"}], []


def run_gate(tmp_path, monkeypatch, account=None, args=()):
    gate = load_gate()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["stage2_gate.py", *args])
    if account is not None:
        import wxbot.kalshi.auth as auth
        import wxbot.kalshi.rest as rest
        monkeypatch.setattr(auth.KalshiAuth, "from_env", classmethod(lambda cls: object()))
        monkeypatch.setattr(rest, "KalshiREST", lambda **kw: account)
    with pytest.raises(SystemExit) as e:
        gate.main()
    return e.value.code


def test_a_clean_record_passes_with_the_account_checks(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path, halts=["HALT: daily loss stop hit (-15.20)"])
    assert run_gate(tmp_path, monkeypatch, Account(t0)) == 0
    out = capsys.readouterr().out
    assert "STAGE 2 GATE: PASS" in out and "[PASS ] 6." in out and "[PASS ] 7." in out


def test_a_maker_order_filled_as_taker_fails_but_the_takers_own_do_not(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path, taker_fill=True)
    assert run_gate(tmp_path, monkeypatch, Account(t0, taker=True)) == 0
    assert "[PASS ] 7." in capsys.readouterr().out
    assert run_gate(tmp_path, monkeypatch, Account(t0, taker=True, maker_took=True)) == 1
    assert "[FAIL ] 7." in capsys.readouterr().out


def test_a_fill_on_arrival_in_the_log_fails_even_without_the_api(tmp_path, monkeypatch, capsys):
    write_record(tmp_path, log_extra=[
        "WARNING wxbot.execution.live: post-only order KXHIGHNY-26OCT05-T75 ask @3c filled 5 on arrival: counted as "
        "exposure until the account reads catch up"])
    assert run_gate(tmp_path, monkeypatch, args=["--no-api"]) == 1
    assert "[FAIL ] 7." in capsys.readouterr().out


def test_a_quick_top_up_beside_a_mostly_filled_order_is_not_a_duplicate(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path)
    assert run_gate(tmp_path, monkeypatch, Account(t0, second="topup")) == 0
    assert "1 top-ups beside a mostly filled order" in capsys.readouterr().out


def test_without_the_api_the_verdict_is_check_not_pass(tmp_path, monkeypatch, capsys):
    write_record(tmp_path)
    assert run_gate(tmp_path, monkeypatch, args=["--no-api"]) == 1
    assert "STAGE 2 GATE: CHECK" in capsys.readouterr().out


@pytest.mark.parametrize("kw, item", [
    ({"halts": ["HALT: API error burst: pausing 15 min"]}, "1."),
    ({"quote_days": 1}, "1."),                                    # one trading day only
    ({"quote_hours": 0.5}, "1."),                                  # half an hour a day is not a trading day
    ({"uncertain_after": 75.0}, "2."),
    ({"pending_after": 90.0}, "3."),
    ({"pending_resolved": False}, "3."),                           # a cancel that never resolved
    ({"fill_price": 4}, "5.")])
def test_each_problem_fails_its_item(tmp_path, monkeypatch, capsys, kw, item):
    t0 = write_record(tmp_path, **kw)
    assert run_gate(tmp_path, monkeypatch, Account(t0)) == 1
    assert f"[FAIL ] {item}" in capsys.readouterr().out


def test_a_second_order_resting_beside_the_first_fails_even_12_s_later(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path)
    assert run_gate(tmp_path, monkeypatch, Account(t0, second="dup")) == 1
    assert "[FAIL ] 4." in capsys.readouterr().out


def test_a_stale_status_file_makes_the_equity_check_a_check(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path, status_age=3600)
    assert run_gate(tmp_path, monkeypatch, Account(t0)) == 1
    out = capsys.readouterr().out
    assert "[CHECK] 6." in out and "is the bot running?" in out


def test_since_ignores_an_earlier_aborted_attempt(tmp_path, monkeypatch, capsys):
    from wxbot.store.db import Store
    t0 = write_record(tmp_path)
    st = Store(str(tmp_path / "data" / "live.sqlite"))
    st.c.execute("INSERT INTO events_log (ts, level, msg) VALUES (?, 'INFO', 'start mode=live')", (t0 - 20 * 86400,))
    st.c.execute("INSERT INTO events_log (ts, level, msg) VALUES (?, 'ERROR', 'HALT: repeated step failures')",
                 (t0 - 20 * 86400 + 60,))
    st.c.commit()
    assert run_gate(tmp_path, monkeypatch, Account(t0)) == 1                     # judged from the first attempt
    since = time.strftime("%Y-%m-%d", time.localtime(t0 - 3600))
    assert run_gate(tmp_path, monkeypatch, Account(t0), args=["--since", since]) == 0


def test_no_fills_yet_is_a_check_not_a_pass(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path, with_fill=False)
    assert run_gate(tmp_path, monkeypatch, Account(t0, no_fills=True)) == 1
    out = capsys.readouterr().out
    assert "[CHECK] 5." in out and "STAGE 2 GATE: CHECK" in out


def test_an_order_without_an_update_time_ends_at_its_last_fill(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path)
    assert run_gate(tmp_path, monkeypatch, Account(t0, second="no_update_time")) == 0
    assert "1 orders without an update time" in capsys.readouterr().out


def test_a_cancel_and_replace_without_update_times_is_a_check_not_a_fail(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path)
    acct = Account(t0, second="no_update_time")
    o3 = acct.os[-1]
    acct.os[0].update({"expiration_time": iso(t0 + 8300)})   # o1: no update time, expiry after o3 went out
    acct.fs = []                                              # and no fills: its end can only be estimated
    acct.os[0]["fill_count_fp"] = "0"
    assert o3["order_id"] == "o3"
    assert run_gate(tmp_path, monkeypatch, acct) == 1
    out = capsys.readouterr().out
    assert "[CHECK] 4." in out and "pairs that may overlap" in out and "[FAIL ] 4." not in out


# ------------------------------------------------------------- Oct 6 demo test: a cancel that waited out a pause
def add_log(tmp_path, t0, rows):
    lt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ",000"
    with open(tmp_path / "data" / "live.log", "a") as f:
        for dt_s, msg in rows:
            f.write(f"{lt(t0 + dt_s)} INFO wxbot: {msg}\n")


START, END = "PAUSE start: Kalshi's status: trading paused; execution waits", "PAUSE end after={}s: execution resumes"


@pytest.mark.parametrize("rows, verdict", [
    ([], "FAIL"),                                                                  # 143 s, no pause
    ([(80, START), (180, END.format(100))], "PASS"),                               # 43 s with the pause left out
    ([(150, START), (180, END.format(30))], "FAIL"),                               # 113 s: a short pause is no excuse
    ([(180, END.format(100))], "PASS"),                                            # the pause began before the window
    ([(80, START), (120, "start mode=live code=abc"), (125, START), (180, END.format(55))], "PASS"),  # restart in it
    ([(80, START)], "FAIL"),                                                       # still paused: not left out yet
])
def test_a_slow_cancel_is_judged_without_the_exchanges_paused_time(tmp_path, monkeypatch, capsys, rows, verdict):
    """The cancel went unconfirmed at t0+59 and resolved at t0+202 (143 s), like the demo's at 11:53-11:56 ET."""
    t0 = write_record(tmp_path, pending_after=143.0)
    add_log(tmp_path, t0, rows)
    run_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert f"[{verdict:<5}] 3." in out
    if verdict == "PASS":
        assert "1 waited out an exchange outage" in out and "143 s with it" in out
    if rows == [(80, START)]:
        assert "run the gate again after the pause" in out


@pytest.mark.parametrize("lead, verdict", [(60, "PASS"), (200, "FAIL")])
def test_an_outage_reaches_back_to_the_first_service_unavailable_answer(tmp_path, monkeypatch, capsys, lead, verdict):
    """Oct 6 demo: Kalshi answered 503 for up to a minute before its status said paused. The cancel went unconfirmed at
    t0+59 and resolved at t0+202 (143 s); the logged pause covers t0+150-180 (30 s), leaving 113 s: a fail on its own.
    A 503 60 s before the pause starts the outage at t0+90 (90 s out, 53 s left: a pass); one 200 s before the pause
    is too early to count (the outage is the pause alone again)."""
    t0 = write_record(tmp_path, pending_after=143.0)
    add_log(tmp_path, t0, [(150 - lead, 'order error: POST /portfolio/events/orders/x/amend -> 503: '
                                        '{"error":{"code":"service_unavailable"}}'),
                           (150, START), (180, END.format(30))])
    run_gate(tmp_path, monkeypatch, Account(t0))
    assert f"[{verdict:<5}] 3." in capsys.readouterr().out


def test_an_answerless_placement_is_judged_without_the_paused_time_too(tmp_path, monkeypatch, capsys):
    t0 = write_record(tmp_path, uncertain_after=95.0)       # resolved at t0+105 after 95 s
    add_log(tmp_path, t0, [(20, START), (90, END.format(70))])
    run_gate(tmp_path, monkeypatch, Account(t0))
    out = capsys.readouterr().out
    assert "[PASS ] 2." in out and "1 waited out an exchange outage" in out
