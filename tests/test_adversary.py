"""scripts/adversary.py (the troublemaker, Oct 6) against a fake demo exchange: it only ever trades with account 2's
key on the demo exchange, never writes with the bot's key, stays inside its limits, cleans up its own orders, and
its checks catch a fill the bot did not book, a cap breach and a halted bot."""
import importlib.util
import json
import os
import random
import sqlite3
import sys
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
EV = "KXHIGHTATL-26OCT07"


def load():
    spec = importlib.util.spec_from_file_location("adversary", os.path.join(ROOT, "scripts", "adversary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pem():
    return Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                       serialization.NoEncryption())


class Exchange:
    """The demo exchange as both accounts see it: the bot's resting orders, its fills and positions; account 2's own
    orders. A take fills against the bot's order at its price."""

    def __init__(self, bot_orders=None, open_=True, close_in=6 * 3600):
        self.bot = {o["order_id"]: o for o in (bot_orders or [
            order("b1", f"{EV}-T80", "ask", 4, 5), order("b2", f"{EV}-B78.5", "ask", 6, 3),
            order("b3", "KXHIGHTDC-26OCT07-T70", "bid", 95, 2)])}
        self.open, self.close_in = open_, close_in
        self.bot_fills, self.positions, self.mine = [], {}, {}
        self.others = []        # other traders' resting orders, (ticker, side, px, n): in the books, never filled here
        self.creates, self.cancels, self.n = [], [], 0

    # ------------------------------------------------- account 2
    def create_order(self, ticker, side, price_cents, count, **kw):
        self.creates.append(dict(kw, ticker=ticker, side=side, px=price_cents, count=count))
        self.n += 1
        oid = f"a{self.n}"
        if kw.get("tif") == "immediate_or_cancel":
            got = 0.0
            for o in self.bot.values():
                px = int(round(float(o["yes_price_dollars"]) * 100))
                if o["ticker"] == ticker and o["book_side"] != side and px == price_cents and count - got > 0:
                    k = min(float(o["remaining_count_fp"]), count - got)
                    o["remaining_count_fp"] = f"{float(o['remaining_count_fp']) - k:.2f}"
                    got += k
                    self.bot_fills.append({"fill_id": f"f{len(self.bot_fills) + 1}", "order_id": o["order_id"],
                                           "count_fp": f"{k:.2f}"})
                    q = k if o["book_side"] == "bid" else -k
                    self.positions[ticker] = self.positions.get(ticker, 0.0) + q
            self.bot = {i: o for i, o in self.bot.items() if float(o["remaining_count_fp"]) > 0}
            return {"order_id": oid, "fill_count": f"{got:.2f}", "remaining_count": "0.00"}
        self.mine[oid] = {"order_id": oid, "client_order_id": kw.get("client_order_id"), "ticker": ticker}
        return {"order_id": oid, "remaining_count": f"{count:.2f}", "fill_count": "0.00"}

    def batch_cancel(self, ids, **kw):
        self.cancels += list(ids)
        for i in ids:
            self.mine.pop(i, None)
        return {"orders": [{"order_id": i, "reduced_by": "1.00", "ts_ms": 1} for i in ids]}

    def exchange_status(self):
        return {"trading_active": self.open}

    def orderbooks(self, tickers):
        """Kalshi's shape: YES bids under yes_dollars, YES asks as NO bids at 1 - price under no_dollars."""
        out = {t: {"yes_dollars": [], "no_dollars": []} for t in tickers}
        rows = [(o["ticker"], o["book_side"], int(round(float(o["yes_price_dollars"]) * 100)),
                 float(o["remaining_count_fp"])) for o in self.bot.values()] + list(self.others)
        for t, side, px, n in rows:
            if t in out:
                key, p = ("yes_dollars", px) if side == "bid" else ("no_dollars", 100 - px)
                out[t][key].append([f"{p / 100:.4f}", f"{n:.2f}"])
        return out

    def market(self, t):
        return {"close_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + self.close_in))}

    def markets(self, event_ticker=None, **kw):
        return [{"ticker": f"{event_ticker}-{s}"} for s in ("T70", "B72.5", "B74.5", "B76.5", "B78.5", "T80")]


class Adv(Exchange):
    pass


class Bot:
    """The bot's account, read-only, over the same exchange."""

    def __init__(self, ex):
        self.ex = ex

    def orders(self, status="resting", **kw):
        return [dict(o) for o in self.ex.bot.values()]

    def fills(self, min_ts=None, **kw):
        return list(self.ex.bot_fills)

    def positions(self, **kw):
        out = []
        for t, q in self.ex.positions.items():
            px = 4 if t.endswith("T80") else 6
            out.append({"ticker": t, "position_fp": f"{q:.2f}",
                        "market_exposure_dollars": f"{abs(q) * (100 - px) / 100:.2f}"})
        return out, []


class AdvView:
    """Account 2's view: its own resting orders under orders()."""

    def __init__(self, ex):
        self.ex = ex

    def __getattr__(self, k):
        return getattr(self.ex, k)

    def orders(self, status="resting", **kw):
        return [dict(o) for o in self.ex.mine.values()]


def order(oid, ticker, side, px, n):
    return {"order_id": oid, "client_order_id": f"wxb-A-{oid}", "ticker": ticker, "book_side": side,
            "yes_price_dollars": f"{px / 100:.4f}", "remaining_count_fp": f"{n:.2f}"}


CLOCK = [0.0]


def no_sleep(s):
    CLOCK[0] += s


@pytest.fixture
def kit(tmp_path, monkeypatch):
    CLOCK[0] = time.time()
    monkeypatch.setattr(time, "time", lambda: CLOCK[0])      # simulated minutes pass at once
    adv = load()
    monkeypatch.setattr(adv, "DATA", str(tmp_path))
    json.dump({"ts": time.time(), "halted": False}, open(tmp_path / "demo_status.json", "w"))
    c = sqlite3.connect(str(tmp_path / "demo.sqlite"))
    c.execute("CREATE TABLE fills (ts REAL, mode TEXT, order_ref TEXT)")
    c.commit()
    c.close()
    (tmp_path / "demo.log").write_text("")
    ex = Exchange()
    return adv, ex, AdvView(ex), Bot(ex), tmp_path


def book_all(tmp_path, ex):
    """The bot books every fill Kalshi shows for its orders."""
    c = sqlite3.connect(str(tmp_path / "demo.sqlite"))
    for f in ex.bot_fills:
        c.execute("INSERT INTO fills VALUES (?, 'live', ?)", (time.time(), f["fill_id"]))
    c.commit()
    c.close()


# --------------------------------------------------------------------------------------------- keys and safety
def test_it_refuses_the_bots_key_a_missing_key_and_never_leaves_the_demo_exchange(tmp_path, monkeypatch):
    adv = load()
    keys = tmp_path / "keys"
    keys.mkdir()
    with pytest.raises(SystemExit, match="no key for demo account 2"):
        adv.clients(str(keys))
    k = pem()
    for name, kid in (("kalshi_demo", "bot-id"), ("kalshi_demo2", "adv-id")):
        (keys / f"{name}.key").write_bytes(k)
        (keys / f"{name}_key_id.txt").write_text(kid + "\n")
    with pytest.raises(SystemExit, match="the bot's key or a real key"):
        adv.clients(str(keys))
    (keys / "kalshi_demo2.key").write_bytes(pem())
    (keys / "kalshi_live.key").write_bytes((keys / "kalshi_demo2.key").read_bytes())
    with pytest.raises(SystemExit, match="the bot's key or a real key"):
        adv.clients(str(keys))
    os.remove(keys / "kalshi_live.key")
    monkeypatch.setenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
    a, b = adv.clients(str(keys))
    from wxbot.kalshi.rest import DEMO
    assert a.base == DEMO.rstrip("/") and b.base == DEMO.rstrip("/")
    assert a.auth.key_id == "adv-id" and b.auth.key_id == "bot-id" and isinstance(b, adv.ReadOnly)


def test_the_bots_key_never_writes(monkeypatch):
    adv = load()
    import requests
    monkeypatch.setattr(requests.Session, "request", lambda *a, **kw: pytest.fail("a request went out"))
    from wxbot.kalshi.auth import KalshiAuth, load_private_key
    from wxbot.kalshi.rest import DEMO
    b = adv.ReadOnly(base_url=DEMO, auth=KalshiAuth("bot-id", load_private_key(pem())))
    for call in (lambda: b.create_order("X-1", "bid", 5, 1), lambda: b.batch_cancel(["o1"]),
                 lambda: b.amend_order("o1", "X-1", "bid", 5, 2), lambda: b.cancel_order("o1")):
        with pytest.raises(RuntimeError, match="read-only"):
            call()


# ------------------------------------------------------------------------------------------------------- moves
def test_a_take_is_immediate_or_cancel_at_the_bots_price_on_the_other_side(kit):
    adv, ex, a, b, tmp = kit
    rec = adv.play(a, b, move="swallow", rng=random.Random(1), sleep=no_sleep, used=(0, 0, {}))
    takes = [c for c in ex.creates if c.get("tif") == "immediate_or_cancel"]
    assert takes and all(c["post_only"] is False and c["client_order_id"].startswith("adv-") for c in takes)
    t = takes[0]
    bot_side = {"b1": "ask", "b2": "ask", "b3": "bid"}
    hit = [x for x in rec["actions"] if x.get("got")][0]
    assert t["side"] != bot_side[hit["bot_order"]] and t["px"] == hit["px"]


def test_a_round_never_takes_more_than_its_limits(kit):
    adv, ex, a, b, tmp = kit
    ex.bot = {"b1": order("b1", f"{EV}-T80", "ask", 4, 10)}
    rec = adv.play(a, b, move="swallow", rng=random.Random(2), sleep=no_sleep, used=(0, 0, {}))
    assert sum(x["got"] for x in rec["actions"]) == 3.0                    # 3 a round
    rec = adv.play(a, b, move="swallow", rng=random.Random(2), sleep=no_sleep, used=(0, 0, {EV: 3.5}))
    assert sum(x["got"] for x in rec["actions"]) == 0.5                    # 4 per event a day
    rec = adv.play(a, b, move="swallow", rng=random.Random(2), sleep=no_sleep, used=(11.0, 0, {}))
    assert sum(x["got"] for x in rec["actions"]) == 1.0                    # 12 a day
    for seed in range(20):                                                 # the day's takes are used up: only rests
        json.dump({"ts": time.time()}, open(tmp / "demo_status.json", "w"))     # the bot is still up
        rec = adv.play(a, b, rng=random.Random(seed), sleep=no_sleep, used=(12.0, 0, {}))
        assert rec["move"] in ("wall", "undercut")


def test_it_leaves_markets_near_their_close_and_a_closed_exchange_alone(kit):
    adv, ex, a, b, tmp = kit
    ex.close_in = 20 * 60
    rec = adv.play(a, b, move="nibble", sleep=no_sleep, used=(0, 0, {}))
    assert "far enough from a market's close" in rec["skipped"] and not ex.creates
    ex.close_in, ex.open = 6 * 3600, False
    rec = adv.play(a, b, move="nibble", sleep=no_sleep, used=(0, 0, {}))
    assert rec["skipped"] == "the exchange is not trading" and not ex.creates
    ex.open = True
    json.dump({"ts": time.time() - 600}, open(tmp / "demo_status.json", "w"))
    rec = adv.play(a, b, move="nibble", sleep=no_sleep, used=(0, 0, {}))
    assert "not running" in rec["skipped"] and not ex.creates


@pytest.mark.parametrize("move, side, px", [("wall", "bid", 3), ("undercut", "ask", 3)])
def test_wall_and_undercut_rest_one_cent_off_the_bot_and_are_cancelled(kit, move, side, px):
    adv, ex, a, b, tmp = kit
    ex.bot = {"b1": order("b1", f"{EV}-T80", "ask", 4, 5)}
    rec = adv.play(a, b, move=move, rng=random.Random(3), sleep=no_sleep, used=(0, 0, {}))
    [r] = [c for c in ex.creates]
    assert (r["side"], r["px"], r["post_only"], r["count"]) == (side, px, True, 1)
    assert r["expiration_ts"] <= time.time() + 600 and not ex.mine and ex.cancels    # cancelled at the round's end
    assert rec["actions"][0]["bot_seen"]


def test_rapid_takes_three_times_and_dust_takes_a_fraction(kit):
    adv, ex, a, b, tmp = kit
    rec = adv.play(a, b, move="rapid", rng=random.Random(4), sleep=no_sleep, used=(0, 0, {}))
    assert len([x for x in rec["actions"] if x["act"] == "take"]) == 3
    rec = adv.play(a, b, move="dust", rng=random.Random(4), sleep=no_sleep, used=(0, 0, {}))
    assert 0 < rec["actions"][0]["asked"] < 1


def test_chase_takes_a_new_order_as_soon_as_it_appears(kit):
    adv, ex, a, b, tmp = kit
    calls = {"n": 0}

    def sleep(s):
        no_sleep(s)
        calls["n"] += 1
        if calls["n"] == 3:                                  # the bot places a new order meanwhile
            ex.bot["b9"] = order("b9", f"{EV}-B74.5", "ask", 5, 4)
    rec = adv.play(a, b, move="chase", rng=random.Random(5), sleep=sleep, used=(0, 0, {}))
    assert rec["actions"][0]["bot_order"] == "b9" and rec["actions"][0]["got"] == 2.0


# ------------------------------------------------------------------------------------------------------ checks
def test_the_checks_pass_when_the_bot_booked_everything(kit):
    adv, ex, a, b, tmp = kit
    rec = adv.play(a, b, move="nibble", rng=random.Random(6), sleep=lambda s: (no_sleep(s), book_all(tmp, ex)),
                   used=(0, 0, {}))
    assert rec["ok"] is True, rec["checks"]
    assert rec["checks"]["booked"]["kalshi_fills"] >= 1 and not rec["checks"]["booked"]["missing"]


def test_a_fill_the_bot_did_not_book_fails_the_round(kit):
    adv, ex, a, b, tmp = kit
    rec = adv.play(a, b, move="nibble", rng=random.Random(6), sleep=no_sleep, used=(0, 0, {}))
    assert rec["ok"] is False and rec["checks"]["booked"]["missing"] == ["f1"]


def test_caps_are_checked_from_kalshis_numbers(kit):
    adv, ex, a, b, tmp = kit
    m = adv.Markets(a)
    limits = {"event": 10.0, "bracket": 5.0, "total": 40.0}
    ex.bot = {}
    ex.positions = {f"{EV}-T80": -5.0}                          # 5 NO at 96c: $4.80, inside every cap
    assert adv.check_caps(b, m, limits)["ok"]
    ex.positions = {f"{EV}-T80": -6.0}                          # $5.76 on one market
    r = adv.check_caps(b, m, limits)
    assert not r["ok"] and r["over"] == [f"market {EV}-T80: $5.76 > $5"]
    ex.bot = {"b1": order("b1", f"{EV}-T80", "ask", 4, 1)}      # $4.80 held and $0.96 resting on that market
    ex.positions = {f"{EV}-T80": -5.0}
    assert not adv.check_caps(b, m, limits)["ok"]
    ex.bot = {}
    ex.positions = {f"{EV}-{s}": 4.0 for s in ("T70", "B72.5", "B74.5")}   # $3.76 each, $11.28 if none wins
    r = adv.check_caps(b, m, limits)
    assert not r["ok"] and r["over"] == [f"event {EV}: worst case $11.28 > $10"]


def test_a_halted_or_silent_bot_fails_the_health_check(kit):
    adv, ex, a, b, tmp = kit
    t0 = time.time() - 5
    (tmp / "demo.log").write_text(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +
                                  ",001 ERROR wxbot: daily loss stop: realized -21.00\n")
    json.dump({"ts": time.time(), "halted": True, "halt_reason": "daily loss stop hit"},
              open(tmp / "demo_status.json", "w"))
    h = adv.check_health(t0)
    assert not h["ok"] and h["halted"] and h["new_errors"] == 1
    assert h["errors"] == ["[an error line about money]"]       # no money lines in its record (the holdout rule)


# --------------------------------------------------------------------------------------------- run and report
def test_run_stops_on_the_flag_and_cancels_its_orders(kit, monkeypatch):
    adv, ex, a, b, tmp = kit
    ex.mine["a77"] = {"order_id": "a77", "client_order_id": "adv-wall-x", "ticker": f"{EV}-T80"}
    rounds = []
    monkeypatch.setattr(adv, "play", lambda *x, **k: rounds.append(1) or {"ts": time.time(), "skipped": "test"})

    def sleep(s):
        no_sleep(s)
        (tmp / "ADVERSARY_STOP").write_text("")
    adv.run(a, b, rng=random.Random(7), sleep=sleep, watch=adv.Watch(Phone()))
    assert rounds == [1] and "a77" in ex.cancels and not ex.mine


def test_the_report_counts_rounds_and_lists_failed_checks(kit, capsys):
    adv, ex, a, b, tmp = kit
    now = time.time()
    rows = [{"ts": now, "move": "nibble", "ok": True, "actions": [
                {"act": "take", "ticker": f"{EV}-T80", "side": "bid", "px": 4, "got": 2.0}],
             "checks": {"booked": {"ok": True}}},
            {"ts": now, "move": "sweep", "ok": False, "actions": [],
             "checks": {"booked": {"ok": False, "missing": ["f9"]}, "caps": {"ok": True}}},
            {"ts": now, "skipped": "the exchange is not trading", "actions": []}]
    assert adv.report(rows) is False
    out = capsys.readouterr().out
    assert "3 rounds (1 skipped)" in out and "took 2 contracts" in out and "FAILED" in out and "f9" in out
    assert "today (ET): 2 contracts, $0.08" in out


# ------------------------------------------------------------------------------------------- the server service
def trouble_kit(tmp_path, with_key=True, active=False):
    import shutil
    root = tmp_path / "weather_trader"
    repo = root / "kalshi-wx-bot"
    for d in ("scripts", "deploy/systemd", ".venv/bin"):
        os.makedirs(repo / d, exist_ok=True)
    for f in ("scripts/vm_trouble.sh", "deploy/systemd/wxbot-adversary.service"):
        shutil.copy(os.path.join(ROOT, f), repo / f)
    kd = root / "deploy_keys"
    kd.mkdir()
    if with_key:
        (kd / "kalshi_demo2.key").write_text("KEY 2")
        (kd / "kalshi_demo2_key_id.txt").write_text("id-2\n")
    log = tmp_path / "calls.log"
    (repo / ".venv/bin/python").write_text(f'#!/bin/bash\necho "python $*" >> "{log}"\n')
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    unit = tmp_path / "unit"
    (fakebin / "systemctl").write_text(f'#!/bin/bash\necho "systemctl $*" >> "{log}"\n'
                                       + ('[ "$1" = "is-active" ] && exit 0\n' if active else
                                          '[ "$1" = "is-active" ] && exit 3\n') + "exit 0\n")
    (fakebin / "sudo").write_text(f'#!/bin/bash\necho "sudo $*" >> "{log}"\n'
                                  f'if [ "$1" = "tee" ]; then exec cat > "{unit}"; fi\nexec "$@"\n')
    for f in (repo / ".venv/bin/python", fakebin / "systemctl", fakebin / "sudo"):
        os.chmod(f, 0o755)
    env = {**os.environ, "PATH": f"{fakebin}:{os.environ['PATH']}"}
    return repo, log, unit, env


def trouble(repo, what, env):
    import subprocess
    return subprocess.run(["bash", str(repo / "scripts" / "vm_trouble.sh"), what], capture_output=True, text=True,
                          timeout=60, env=env)


def test_vm_trouble_start_installs_the_service_for_this_folder_and_clears_the_stop_flag(tmp_path):
    repo, log, unit, env = trouble_kit(tmp_path)
    os.makedirs(repo / "data" / "demo")
    (repo / "data" / "demo" / "ADVERSARY_STOP").write_text("")
    r = trouble(repo, "start", env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not (repo / "data" / "demo" / "ADVERSARY_STOP").exists()
    assert "systemctl enable --now wxbot-adversary" in log.read_text()
    text = unit.read_text()
    assert f"WorkingDirectory={repo}" in text and f"ExecStart={repo}/.venv/bin/python scripts/adversary.py run" in text


def test_vm_trouble_needs_account_2s_key_and_stop_leaves_a_flag(tmp_path):
    repo, log, unit, env = trouble_kit(tmp_path, with_key=False, active=True)
    r = trouble(repo, "start", env)
    assert r.returncode == 1 and "no key for demo account 2" in r.stdout and not log.exists()
    r = trouble(repo, "stop", env)
    assert r.returncode == 0 and (repo / "data" / "demo" / "ADVERSARY_STOP").exists()
    assert "systemctl disable --now wxbot-adversary" in log.read_text()


def test_takes_only_hit_bot_orders_that_are_the_best_price_on_their_side(kit):
    """The first live round (Oct 6): a take at the bot's 4c ask first filled 2 from another trader's 3c ask."""
    adv, ex, a, b, tmp = kit
    ex.others = [(f"{EV}-T80", "ask", 3, 2.0)]                          # someone asks 3c, below the bot's 4c
    for seed in range(12):
        json.dump({"ts": time.time()}, open(tmp / "demo_status.json", "w"))
        rec = adv.play(a, b, move="nibble", rng=random.Random(seed), sleep=no_sleep, used=(0, 0, {}))
        assert all(x["ticker"] != f"{EV}-T80" for x in rec["actions"] if x["act"] == "take")
    ex.bot = {"b1": order("b1", f"{EV}-T80", "ask", 4, 5)}             # the only bot order is not the best: rest
    json.dump({"ts": time.time()}, open(tmp / "demo_status.json", "w"))
    n = len(ex.creates)
    rec = adv.play(a, b, move="swallow", rng=random.Random(1), sleep=no_sleep, used=(0, 0, {}))
    assert rec["move"] in ("wall", "undercut") and "best price" in rec["note"]
    assert all(c.get("post_only") for c in ex.creates[n:])


def test_rests_never_cross_the_book(kit):
    """The second live round (Oct 6): a wall bid at 4c beside the bot's 5c ask was refused, another ask sat at 4c."""
    adv, ex, a, b, tmp = kit
    ex.bot = {"b1": order("b1", f"{EV}-T77", "ask", 5, 5)}
    ex.others = [(f"{EV}-T77", "ask", 4, 1.0), (f"{EV}-T77", "bid", 3, 1.0)]
    rec = adv.play(a, b, move="wall", rng=random.Random(1), sleep=no_sleep, used=(0, 0, {}))
    assert rec["actions"][0]["act"] == "skip" and not ex.creates                # a 4c bid would cross the 4c ask
    ex.others = [(f"{EV}-T77", "bid", 4, 1.0)]
    json.dump({"ts": time.time()}, open(tmp / "demo_status.json", "w"))
    rec = adv.play(a, b, move="undercut", rng=random.Random(1), sleep=no_sleep, used=(0, 0, {}))
    assert rec["actions"][0]["act"] == "skip" and not ex.creates                # a 4c ask would cross the 4c bid
    ex.others = [(f"{EV}-T77", "bid", 2, 1.0)]
    json.dump({"ts": time.time()}, open(tmp / "demo_status.json", "w"))
    rec = adv.play(a, b, move="undercut", rng=random.Random(1), sleep=no_sleep, used=(0, 0, {}))
    assert [(c["side"], c["px"]) for c in ex.creates] == [("ask", 4)]           # room: a 4c ask above a 2c bid


# ----------------------------------------------------------------------------------------------- phone alerts
class Phone:
    def __init__(self):
        self.sent = []
        self.enabled = True

    def send(self, key, title, msg, priority="default", tags=""):
        if any(k == key for k, _, _ in self.sent):
            return False                                     # the real Alerter limits each key to once in 3 h
        self.sent.append((key, title, msg))
        return True


def test_a_failed_check_pages_once_and_names_what_failed(kit):
    adv, ex, a, b, tmp = kit
    phone = Phone()
    w = adv.Watch(phone)
    rec = {"move": "sweep", "checks": {"booked": {"ok": False, "missing": ["f1", "f2"]}, "caps": {"ok": True}}}
    w.after(rec)
    w.after(rec)
    assert len(phone.sent) == 1
    key, title, msg = phone.sent[0]
    assert title == "wxbot DEMO troublemaker: a check failed" and "did not record: 2" in msg and "sweep" in msg
    w.after({"move": "nibble", "checks": {"booked": {"ok": True}, "bot": {"ok": False, "halted": True,
                                                                          "halt_reason": "API error burst"}}})
    assert phone.sent[-1][2].startswith("After a nibble round: bot: halted: API error burst")


def test_it_pages_when_the_demo_bot_goes_quiet_and_when_its_own_rounds_keep_failing(kit):
    adv, ex, a, b, tmp = kit
    phone = Phone()
    w = adv.Watch(phone)
    quiet = {"skipped": "the demo bot is not running (no fresh status)"}
    w.after(quiet)
    assert not phone.sent                                   # one stale read can be a restart
    w.after({"skipped": "the exchange is not trading"})
    w.after(quiet)
    assert not phone.sent                                   # not two in a row
    w.after(quiet)
    assert phone.sent[-1][1] == "wxbot DEMO: the demo bot is not running"
    for _ in range(2):
        w.after(error=ConnectionError("reset"))
    assert len(phone.sent) == 1
    w.after(error=ConnectionError("reset"))
    assert phone.sent[-1][1] == "wxbot DEMO troublemaker: its rounds keep failing"


def test_the_alert_topic_comes_from_the_environment_or_that_one_line_of_env(kit, monkeypatch, tmp_path):
    adv, ex, a, b, tmp = kit
    monkeypatch.setenv("WXBOT_NTFY_TOPIC", "wxbot-from-env")
    assert adv.topic() == "wxbot-from-env"
    monkeypatch.delenv("WXBOT_NTFY_TOPIC")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".env").write_text("KALSHI_KEY_ID=not-read\nWXBOT_NTFY_TOPIC='wxbot-abc'\n")
    monkeypatch.setattr(adv, "ROOT", str(repo))
    assert adv.topic() == "wxbot-abc" and "KALSHI_KEY_ID" not in os.environ
    os.remove(repo / ".env")
    assert adv.topic() == ""
