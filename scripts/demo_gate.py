"""The two-day demo test's verdict. Shourya's rule (Oct 4, 2026): no real money until the bot has traded on Kalshi's
DEMO exchange (fake money) for at least two days and this check passes.

Two days on the code GO_LIVE will run: the record is judged from the start of the latest uninterrupted run of the code
now deployed (`.deployed_commit`; the bot names its code in each start line), so a code change starts the two days
again. On that record and on the demo account (read-only, with the demo key only) it runs the Stage 2 gate's checks
(scripts/stage2_gate.py: two ET days of quoting, no unexplained halt, answerless placements and unconfirmed cancels
resolved within 60 s, no duplicate orders, the bot's fills are Kalshi's fills, settled equity matches the account,
post-only held), plus:
  8. 48 hours of actual running on that code (kv live_activity alive_s: an asleep Mac or a stopped bot does not count);
  9. the order path was exercised on that code: at least 20 orders placed, 10 expiry refreshes and 5 cancels.
PASS writes data/demo/DEMO_GATE_PASSED (when, how long, which code); GO_LIVE.command requires it, for the same code.
Anything else removes an earlier pass. CHECK (for example no demo fills yet, so nothing could be reconciled) is not a
pass: let it run longer and check again.

--accept-short (Shourya's decision, Oct 7: go live after one night on demo instead of two days): items 1 and 8 may
fall short on time only. It still needs at least one ET day with six hours of quoting and no unexplained halt or step
failure (item 1), at least 18 hours of actual running (item 8), and every other item PASS. The marker records the
shortcut ("short": true, the hours), and --pass-info shows it.

Usage (on the Mac, from the repo folder):
  .venv/bin/python scripts/demo_gate.py
  .venv/bin/python scripts/demo_gate.py --balance   (DEMO_START.command: exit 0 funded, 1 too little, 2 unreadable)
  .venv/bin/python scripts/demo_gate.py --pass-info (GO_LIVE.command: is there a pass for the code deployed now?)
  .venv/bin/python scripts/demo_gate.py --accept-short   (Shourya's call: one night on demo is enough; see above)"""
import datetime as dt
import filecmp
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

CONFIG = "config/demo.yaml"
DB = "data/demo/demo.sqlite"
MARKER = "data/demo/DEMO_GATE_PASSED"
MIN_HOURS, MIN_PLACED, MIN_REFRESHED, MIN_CANCELLED = 48.0, 20, 10, 5
SHORT_MIN_HOURS, SHORT_MIN_DAYS = 18.0, 1          # --accept-short (Shourya's decision, Oct 7)
MIN_BALANCE_USD = 50.0


def demo_key_env(keydir: str | None = None) -> bool:
    """Put the demo key, and only it, in the environment before anything reads .env (.env never overrides what is
    already set), so the real key can never reach the demo account or this check. False if no demo key is saved."""
    keydir = keydir or os.path.join(HERE, "..", "..", "deploy_keys")
    key = os.path.join(keydir, "kalshi_demo.key")
    if not os.path.exists(key) and os.path.exists(os.path.join(keydir, "kalshi_demo.pem")):
        key = os.path.join(keydir, "kalshi_demo.pem")
    idf = os.path.join(keydir, "kalshi_demo_key_id.txt")
    if not (os.path.exists(key) and os.path.exists(idf)):
        return False
    for real in ("kalshi_live.key", "kalshi_live.pem"):
        rp = os.path.join(keydir, real)
        if os.path.exists(rp) and filecmp.cmp(rp, key, shallow=False):
            raise SystemExit("the demo key file is the real key: not using it")
    os.environ["KALSHI_KEY_ID"] = open(idf).read().strip()
    os.environ["KALSHI_PRIVATE_KEY_PATH"] = key
    os.environ["KALSHI_PRIVATE_KEY"] = ""
    os.environ["KALSHI_BASE_URL"] = ""
    return True


def balance_check(rest=None) -> int:
    """0: the demo account is funded well enough to run the test (the bot keeps a $25 reserve and deploys up to $60);
    1: too little money; 2: the account could not be read (no key, a wrong Key ID, the network)."""
    try:
        if rest is None:
            from wxbot.kalshi.auth import KalshiAuth
            from wxbot.kalshi.rest import DEMO, KalshiREST
            auth = KalshiAuth.from_env()
            if auth is None:
                print("no demo key: save it as weather_trader/deploy_keys/kalshi_demo.key "
                      "(DEMO_START.command asks for its ID)")
                return 2
            rest = KalshiREST(base_url=DEMO, auth=auth)
        usd = (rest.balance().get("balance") or 0) / 100.0
    except Exception as e:
        print(f"could not read the demo account ({e!r:.200}): is the demo Key ID in "
              "deploy_keys/kalshi_demo_key_id.txt the one that goes with kalshi_demo.key?")
        return 2
    ok = usd >= MIN_BALANCE_USD
    print(f"demo balance ${usd:.2f}" + ("" if ok else f": below ${MIN_BALANCE_USD:.0f}, too little to run the test"))
    return 0 if ok else 1


def code_version(path: str = ".deployed_commit") -> str:
    """The code deployed now, as the bot names it in its start lines (wxbot/app/runner.py Bot.code_version)."""
    try:
        txt = open(path).read().strip()
    except OSError:
        return "unknown"
    m = re.search(r"code as ([0-9a-f]{6,40})", txt)
    return m.group(1) if m else (txt.split() or ["unknown"])[0]


def current_code_start(c, code: str):
    """When the latest uninterrupted run of `code` began: its first start line after the last start of other code."""
    start = None
    for ts, msg in c.execute("SELECT ts, msg FROM events_log WHERE msg LIKE 'start mode=live%' ORDER BY ts"):
        m = re.search(r"code=(\S+)", msg)
        if m and m.group(1) == code:
            start = ts if start is None else start
        else:
            start = None
    return start


def activity(c) -> dict:
    row = c.execute("SELECT v FROM kv WHERE k='live_activity'").fetchone()
    return json.loads(row[0]) if row and row[0] else {}


def make_items(code: str, state: dict):
    def demo_items(c, start: float, res):
        act = activity(c)
        same = act.get("code") == code
        hours = float(act.get("alive_s") or 0.0) / 3600 if same else 0.0
        state["alive_h"] = hours
        res("8. 48 hours running on the code GO_LIVE will run", "PASS" if hours >= MIN_HOURS else "FAIL",
            f"{hours:.1f} h of actual running on code {code} since "
            f"{dt.datetime.fromtimestamp(start):%Y-%m-%d %H:%M} (an asleep Mac or a stopped bot does not count)"
            + ("" if same else f"; the activity count is for code {act.get('code')}: restart the demo bot "
                               "(touch data/demo/RESTART) so it runs the code deployed now"))
        placed, refreshed, cancelled = ((int(act.get(k) or 0) if same else 0) for k in ("placed", "refreshed",
                                                                                         "cancelled"))
        ok = placed >= MIN_PLACED and refreshed >= MIN_REFRESHED and cancelled >= MIN_CANCELLED
        res("9. the order path was exercised", "PASS" if ok else "FAIL",
            f"{placed} placed (need {MIN_PLACED}), {refreshed} expiry refreshes (need {MIN_REFRESHED}), {cancelled} "
            f"cancels (need {MIN_CANCELLED}); {int(act.get('amended') or 0) if same else 0} amends, "
            f"{int(act.get('refused') or 0) if same else 0} refused by Kalshi")
    return demo_items


def short_ok(results: list, state: dict, facts: dict) -> tuple[bool, str]:
    """--accept-short: may the demo count as passed although items 1 and 8 fall short on time? Every other item must
    PASS; item 1 needs SHORT_MIN_DAYS full ET days and no unexplained halt or step failure; item 8 SHORT_MIN_HOURS."""
    others = [(n, v) for n, v, _ in results if not n.startswith(("1.", "8."))]
    bad = [n for n, v in others if v != "PASS"]
    if not others or bad:
        return False, "not every other item passed: " + ", ".join(bad or ["no results"])
    days, hours = len(facts.get("full_days") or []), float(state.get("alive_h") or 0.0)
    if facts.get("bad_halts") or facts.get("steps"):
        return False, "item 1 has an unexplained halt or step failures"
    if days < SHORT_MIN_DAYS:
        return False, f"{days} ET day(s) with six hours of quoting; the short demo needs {SHORT_MIN_DAYS}"
    if hours < SHORT_MIN_HOURS:
        return False, f"{hours:.1f} h of actual running; the short demo needs {SHORT_MIN_HOURS:.0f}"
    return True, f"{days} full ET day(s), {hours:.1f} h of actual running"


def make_marker_writer(code: str, state: dict, path: str = MARKER, short: bool = False, facts: dict | None = None):
    def write_marker(verdict: str, results: list, start: float):
        note = ""
        if verdict != "PASS" and short:
            ok, why = short_ok(results, state, facts or {})
            print(f"\nShort demo (--accept-short, Shourya's decision): {'ACCEPTED' if ok else 'not accepted'}: {why}")
            if ok:
                verdict, note = "PASS", why
        if verdict != "PASS":
            if os.path.exists(path):
                os.remove(path)          # an earlier pass no longer stands
                print(f"\nAn earlier pass ({path}) was removed: GO_LIVE is locked again.")
            print("\nNot passed yet: GO_LIVE stays locked. Run this again later.")
            return verdict
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        rec = {"passed_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
               "first_start": dt.datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M"),
               "hours": round(float(state.get("alive_h") or 0.0), 1),
               "clock_hours": round((time.time() - start) / 3600, 1), "code": code,
               "items": [[n, v] for n, v, _ in results]}
        if note:
            rec.update(short=True, short_note=f"Shourya's decision (Oct 7): live after one night on demo; {note}")
        json.dump(rec, open(path, "w"), indent=1)
        print(f"\nDemo test passed{' (SHORT)' if note else ''}: {path} written. GO_LIVE is unlocked for code {code} "
              "(still Stage 1, still asks you to type GO).")
        return verdict
    return write_marker


def pass_info(path: str = MARKER) -> str:
    """For GO_LIVE.command: "OK <details>" when the recorded pass is for the code deployed now, "OLD <details>" when
    it is for other code, "NONE" without one."""
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return "NONE"
    code = code_version()
    return (("OK " if d.get("code") == code else "OLD ")
            + f"{d.get('passed_at')} after {d.get('hours')} h on demo"
            + (" (SHORT: Shourya decided one night is enough)" if d.get("short") else "")
            + f", code {d.get('code')} (deployed now: {code})")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if "--pass-info" in argv:
        print(pass_info())
        return
    if "--config" in argv:
        sys.exit("demo_gate.py judges the demo bot only (config/demo.yaml): no --config")
    if not demo_key_env():
        os.environ["KALSHI_KEY_ID"] = ""      # never fall back to a key in .env (it would be the real one)
        print("no demo key in weather_trader/deploy_keys (kalshi_demo.key + kalshi_demo_key_id.txt): the account "
              "checks are skipped, so the verdict cannot be PASS")
    if "--balance" in argv:
        sys.exit(balance_check())
    code = code_version()
    state, facts = {}, {}
    short = "--accept-short" in argv
    argv = [a for a in argv if a != "--accept-short"]
    write_marker = make_marker_writer(code, state, short=short, facts=facts)
    if not os.path.exists(DB):
        print(f"no {DB}: the demo test has not run yet (DEMO_START.command)")
        write_marker("FAIL", [], 0.0)
        sys.exit(1)
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    start = current_code_start(c, code)
    c.close()
    if start is None:
        print(f"the demo bot has not run code {code} (the code deployed now) yet: restart it (touch "
              "data/demo/RESTART) or start the demo test (DEMO_START.command)")
        write_marker("FAIL", [], 0.0)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("stage2_gate", os.path.join(HERE, "stage2_gate.py"))
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    gate.main(argv=argv + ["--config", CONFIG], more_items=make_items(code, state),
              label="DEMO GATE (the 2-day demo test)", on_verdict=write_marker, start_ts=start, facts=facts)


if __name__ == "__main__":
    main()
