"""Optional phone alerts through ntfy.sh (free, no account). Install the ntfy app, subscribe to a long random topic
name, and put that name in `.env` as WXBOT_NTFY_TOPIC (kept out of the repo) or in config `alerts.ntfy_topic`.
Anyone who knows the topic can read it, so keep it random.

Sends: halts and pauses, repeated step failures, start/stop, and one daily summary. Never raises.
"""
from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger(__name__)


class Alerter:
    def __init__(self, cfg: dict | None, demo: bool = False):
        cfg = cfg or {}
        self.demo = demo           # the demo-exchange bot: no heartbeat (see heartbeat_topic)
        self.topic = (os.environ.get("WXBOT_NTFY_TOPIC") or cfg.get("ntfy_topic") or "").strip()
        self.server = (cfg.get("ntfy_server") or "https://ntfy.sh").rstrip("/")
        self.summary_hour_et = int(cfg.get("daily_summary_hour_et", 21))
        self.min_gap_s = float(cfg.get("min_gap_s", 600))
        self.last_sent: dict[str, float] = {}
        self.sent: list[tuple] = []  # (title, msg) for tests / status
        self.pending: list[tuple] = []   # alerts that could not be posted (no network): resent by flush()

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    @property
    def heartbeat_topic(self) -> str:
        """Where the bot posts an 'alive' ping every 10 minutes, for an outside check that notices when the Mac
        sleeps, loses power or the bot hangs (nothing on the Mac can report its own silence). Its own topic, so the
        phone is not pinged: WXBOT_HEARTBEAT_TOPIC, or a name derived from the alert topic (shown in status.json;
        it reveals only that the bot is alive)."""
        if self.demo:
            return ""                  # the heartbeat speaks for the main bot only
        hb = (os.environ.get("WXBOT_HEARTBEAT_TOPIC") or "").strip()
        if hb or not self.topic:
            return hb
        import hashlib
        return "wxbot-hb-" + hashlib.sha256(self.topic.encode()).hexdigest()[:20]

    def heartbeat(self, msg: str) -> bool:
        t = self.heartbeat_topic
        if not t:
            return False
        try:
            requests.post(f"{self.server}/{t}", data=msg.encode("utf-8"), headers={"Priority": "min"}, timeout=10)
            return True
        except Exception as e:
            log.debug("heartbeat failed: %s", e)
            return False

    def send(self, key: str, title: str, msg: str, priority: str = "default", tags: str = "") -> bool:
        if not self.enabled:
            return False
        now = time.time()
        if now - self.last_sent.get(key, 0) < self.min_gap_s:
            return False
        self.last_sent[key] = now
        self.sent.append((title, msg))
        if self._post(title, msg, priority, tags):
            return True
        # the network is down (the usual reason): keep it and resend when it is back, so a halt that happens while
        # the Wi-Fi is out still reaches the phone (round-7 pre-mortem)
        self.pending = (self.pending + [(now, title, msg, priority, tags)])[-20:]
        return False

    def _post(self, title: str, msg: str, priority: str, tags: str) -> bool:
        try:
            h = {"Title": title, "Priority": priority}
            if tags:
                h["Tags"] = tags
            r = requests.post(f"{self.server}/{self.topic}", data=msg.encode("utf-8"), headers=h, timeout=10)
            ok = getattr(r, "status_code", 200) < 400
            if not ok:
                log.warning("alert send failed: HTTP %s", r.status_code)
            return ok
        except Exception as e:
            log.warning("alert send failed: %s", e)
            return False

    def flush(self, max_age_s: float = 6 * 3600) -> int:
        """Resend alerts that could not be posted, oldest first (stopping at the first failure); alerts older than
        max_age_s are dropped. Returns how many went out."""
        now, sent = time.time(), 0
        self.pending = [p for p in self.pending if now - p[0] <= max_age_s]
        while self.pending:
            ts, title, msg, priority, tags = self.pending[0]
            late = time.strftime("%H:%M", time.localtime(ts))
            if not self._post(title, f"{msg} (sent late: from {late}, the network was down)", priority, tags):
                break
            self.pending.pop(0)
            sent += 1
        return sent
