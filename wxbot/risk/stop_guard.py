"""Guards on the two account-wide loss stops (Oct 11, 2026).

The stops themselves are unchanged (RiskState.check_stops in wxbot/risk/limits.py, pinned for the Dec 1 holdout): the
daily stop halts for the rest of the ET day when today's settled result reaches -daily_loss_stop_usd; the drawdown stop
halts until `python -m wxbot reset-halt` when settled equity is total_drawdown_stop_usd below its high point. Oct 11
showed two flaws in how they are judged. Each option below is off unless its key is set under `risk:`, so paper and
the older configs behave exactly as before.

  daily_resume_margin_usd: Kalshi pays out the previous day's markets by time zone (Eastern about 7:10 AM ET,
      Central 8:08, Mountain 9:07, West 10:10, rain about 1:10 PM), so the daily stop judged a partial morning. On
      Oct 11 the two losing East Coast bets paid out first: the running total hit -$20.99 at 7:11 and the stop halted
      the whole day, although by 10:10 the day stood at -$9.87. With a margin m, a daily halt is lifted the same ET day
      once today's settled result is back at -(daily_loss_stop_usd - m) or better on 3 equity readings in a row (and
      neither stop's condition holds on them), at most twice a day. A hard halt (the drawdown stop, QUIESCE, repeated
      failures) is never lifted here.
  daily_resume_hold_s: the recovery must also have lasted this many seconds, from the first of those readings to the
      latest (0, the default: the 3 readings are enough). Live reads come about every 5 seconds, so 3 readings alone
      are about 20 seconds: a phantom reading of that length (a payout in the balance while its position is still in
      the list, Oct 11) could otherwise lift a halt.
  stop_confirm_s, drawdown_confirm_s: the daily or the drawdown condition must hold on every equity reading for this
      many seconds before it halts (0, the default: at once, as before). Around payouts the balance and the positions
      list can disagree for longer than the median of three readings covers; a short phantom dip then no longer trips
      a stop.
  peak_from: "reading" (the default, as before: every reading can raise the high point) or "start_of_day": the high
      point rises only to the start-of-day equity (kv sod_equity, set from the first reading after ET midnight, when
      nothing settles), never to an intraday reading. On Oct 11 the stored high point sat about $15 above any real
      equity: a reading taken while the payout
      was already in the balance but the position still in the list raised it by about $15, which brought the $50
      hard halt $15 closer. Drawdown is still measured against the current equity (with drawdown_confirm_s).
      `python -m wxbot set-peak --to-sod` puts a high point raised that way back to the start-of-day equity.

An equity reading is one median of three account reads (live; wxbot/app/runner.py refresh_live_state) or one step
(paper). A timer starts at the first reading on which its condition holds and stops at the first on which it does not;
a gap in the account reads (an outage) does not stop it, but the count of readings back above the resume line starts
over. The timers and the count live in memory, so a restart of the bot starts them afresh, as it starts the readings
afresh. The number of automatic resumes per ET day is kept in the database (kv daily_resumes), so a restart cannot
add a third.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PEAK_FROM = ("reading", "start_of_day")


def usd(x: float) -> str:
    """A day's result in dollars, as the alerts write it: -$9.87, -$20, +$1.50 (whole dollars without cents)."""
    a = abs(float(x))
    txt = f"{a:,.0f}" if abs(a - round(a)) < 0.005 else f"{a:,.2f}"
    return ("-" if x < 0 else "+") + "$" + txt


def duration(s: float) -> str:
    """25 s, 5 min, 3.3 min."""
    if s < 120:
        return f"{s:.0f} s"
    m = s / 60
    return f"{m:.0f} min" if abs(m - round(m)) < 0.05 else f"{m:.1f} min"


@dataclass(frozen=True)
class StopGuardConfig:
    daily_resume_margin_usd: Optional[float] = None   # None: a daily halt lasts the rest of the ET day (as before)
    stop_confirm_s: float = 0.0                        # 0: the daily stop halts at once (as before)
    drawdown_confirm_s: float = 0.0                    # 0: the drawdown stop halts at once (as before)
    peak_from: str = "reading"                         # "reading" (as before) | "start_of_day"
    daily_resume_hold_s: float = 0.0                   # 0: the readings in a row are enough to resume
    resume_readings: int = 3                           # readings in a row back above the line before a resume
    max_resumes_per_day: int = 2                       # automatic resumes per ET day

    @classmethod
    def from_cfg(cls, risk: Optional[dict]) -> "StopGuardConfig":
        r = risk or {}
        m = r.get("daily_resume_margin_usd")
        secs = lambda k: float(r.get(k) or 0)   # noqa: E731
        c = cls(daily_resume_margin_usd=None if m is None else float(m), stop_confirm_s=secs("stop_confirm_s"),
                drawdown_confirm_s=secs("drawdown_confirm_s"), peak_from=str(r.get("peak_from") or "reading"),
                daily_resume_hold_s=secs("daily_resume_hold_s"))
        if c.peak_from not in PEAK_FROM:
            raise ValueError(f"risk.peak_from is {c.peak_from!r}: reading or start_of_day")
        if (m is not None and c.daily_resume_margin_usd < 0) or \
                min(c.stop_confirm_s, c.drawdown_confirm_s, c.daily_resume_hold_s) < 0:
            raise ValueError("risk.daily_resume_margin_usd, daily_resume_hold_s, stop_confirm_s and drawdown_confirm_s "
                             "cannot be negative")
        return c

    @property
    def guards_stops(self) -> bool:
        """The stops go through StopGuard (else RiskState.check_stops, exactly as before)."""
        return self.daily_resume_margin_usd is not None or self.stop_confirm_s > 0 or self.drawdown_confirm_s > 0

    @property
    def enabled(self) -> bool:
        return self.guards_stops or self.peak_from != "reading"


class StopGuard:
    """When each stop's condition was first seen on the equity readings, and how many readings in a row the day's
    result has been back above the resume line during a daily halt."""

    STOPS = ("hard", "daily")      # the drawdown stop first, as in RiskState.check_stops

    def __init__(self, cfg: StopGuardConfig):
        self.cfg = cfg
        self.since: dict = {"hard": None, "daily": None}   # first reading on which the condition held (None: not now)
        self.last_ts: Optional[float] = None               # time of the latest reading
        self.above = 0                                     # readings in a row back above the resume line
        self.above_since: Optional[float] = None           # time of the first of them

    def restart(self):
        """The account reads start over after a gap (an outage): a resume then waits for 3 readings taken after it
        (and daily_resume_hold_s from the first of them). The stops' timers keep running, as a gap is no reading on
        which the condition cleared: a stop whose condition held before the gap and still holds after it halts at the
        first reading after it, once it has held long enough counted from before the gap."""
        self.above, self.above_since = 0, None

    def need(self, kind: str) -> float:
        return self.cfg.drawdown_confirm_s if kind == "hard" else self.cfg.stop_confirm_s

    @staticmethod
    def conditions(risk) -> dict:
        """RiskState.check_stops's two conditions on the current numbers."""
        c = risk.cfg
        return {"hard": risk.peak_equity - risk.equity >= c.total_drawdown_stop_usd,
                "daily": risk.realized_today <= -c.daily_loss_stop_usd}

    def resume_line(self, risk) -> Optional[float]:
        m = self.cfg.daily_resume_margin_usd
        return None if m is None else -(risk.cfg.daily_loss_stop_usd - m)

    def held(self, kind: str) -> float:
        """Seconds of readings the condition has held, 0 when it does not hold."""
        s = self.since[kind]
        return 0.0 if s is None or self.last_ts is None else self.last_ts - s

    def observe(self, ts: float, risk, today: str) -> list:
        """One equity reading. Returns lines for the log: a stop's condition starting to hold while it must hold a
        while before it halts, or clearing before it halted."""
        notes = []
        cond = self.conditions(risk)
        name = {"hard": "drawdown stop", "daily": "daily loss stop"}
        num = {"hard": f"{risk.peak_equity - risk.equity:.2f} below the high point",
               "daily": f"{risk.realized_today:.2f} today"}
        for k in self.STOPS:
            need = self.need(k)
            if cond[k] and self.since[k] is None:
                self.since[k] = ts
                if need > 0 and not risk.halted:
                    notes.append(f"{name[k]} reached ({num[k]}): it halts if it still holds after {duration(need)} "
                                 "of readings")
            elif not cond[k] and self.since[k] is not None:
                if need > 0 and not risk.halted:
                    notes.append(f"{name[k]} no longer reached ({num[k]}) after {duration(ts - self.since[k])} of "
                                 "readings: not halted")
                self.since[k] = None
        self.last_ts = ts
        line = self.resume_line(risk)
        ok = (line is not None and risk.halted and risk.halt_kind == "daily" and risk.halt_day == today
              and risk.realized_today >= line and not cond["daily"] and not cond["hard"])
        self.above = self.above + 1 if ok else 0
        self.above_since = (ts if self.above_since is None else self.above_since) if ok else None
        return notes

    def above_for(self) -> float:
        """Seconds of readings the day's result has been back above the resume line (0 when it is not)."""
        return 0.0 if self.above_since is None or self.last_ts is None else self.last_ts - self.above_since

    def due(self, risk) -> Optional[str]:
        """The stop that halts now: 'hard' (the drawdown stop), 'daily' or None. As RiskState.check_stops (the
        drawdown stop first), except that each condition halts only once it has held for its confirmation time."""
        cond = self.conditions(risk)
        for k in self.STOPS:
            if cond[k] and self.held(k) >= self.need(k):
                return k
        return None

    def reason(self, kind: str, risk, resumes_left: int) -> str:
        """The halt's reason, worded as RiskState.check_stops words it (dashboards and the Stage 2 gate read it), plus
        how long the condition held and when a daily halt can lift."""
        held = f" for {duration(self.held(kind))}" if self.need(kind) > 0 else ""
        if kind == "hard":
            return f"drawdown stop hit ({risk.peak_equity - risk.equity:.2f}){held} - manual review needed"
        why = f"daily loss stop hit ({risk.realized_today:.2f}){held}"
        line = self.resume_line(risk)
        if line is not None and resumes_left > 0:
            hold = self.cfg.daily_resume_hold_s
            why += f"; trading resumes today if the day's settled result recovers to {usd(line)}"
            why += f" for {duration(hold)}" if hold > 0 else ""
        elif line is not None:
            why += f"; no automatic resume left today ({self.cfg.max_resumes_per_day} used)"
        return why

    def resume_due(self, risk, today: str, resumes_today: int) -> bool:
        """A daily halt of today that the day's result has climbed back out of (daily_resume_margin_usd): back above
        the line on resume_readings readings in a row, and for daily_resume_hold_s from the first of them."""
        return (self.cfg.daily_resume_margin_usd is not None and risk.halted and risk.halt_kind == "daily"
                and risk.halt_day == today and self.above >= self.cfg.resume_readings
                and self.above_for() >= self.cfg.daily_resume_hold_s
                and resumes_today < self.cfg.max_resumes_per_day)

    def status(self, risk, resumes_today: int) -> dict:
        held = lambda k: None if self.since[k] is None else round(self.held(k), 1)   # noqa: E731
        return {"daily_resume_margin_usd": self.cfg.daily_resume_margin_usd, "resume_line_usd": self.resume_line(risk),
                "daily_resume_hold_s": self.cfg.daily_resume_hold_s, "readings_above": self.above,
                "above_s": round(self.above_for(), 1) if self.above else None, "resumes_today": resumes_today,
                "stop_confirm_s": self.cfg.stop_confirm_s, "daily_held_s": held("daily"),
                "drawdown_confirm_s": self.cfg.drawdown_confirm_s, "drawdown_held_s": held("hard"),
                "peak_from": self.cfg.peak_from}
