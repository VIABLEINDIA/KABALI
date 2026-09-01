"""The limit that stops a slow bleed, which the daily limit cannot.

WHY THE DAILY LIMIT IS NOT ENOUGH
=================================
`risk.circuit` caps a single session at 2% of capital, and across 66 replayed
sessions it never once fired. That is not the limit working -- it is the limit
being irrelevant to the failure mode that actually occurs.

This system does not blow up. It bleeds. Its worst session was -Rs 763 against an
Rs 800 stop, and its AVERAGE session was -Rs 176. A daily limit answers "how bad
can today be"; nothing answered "how many days like this before I stop", and the
answer was: indefinitely.

A per-session cap on a negative-expectancy strategy is a speed limit on a road
that goes off a cliff. It controls the rate of loss and nothing else.

WHAT THIS MEASURES
==================
Drawdown from the equity PEAK, not cumulative loss from the start. A system that
made Rs 5,000 and gave back Rs 4,000 has lost the same money as one that simply
lost Rs 4,000, and an operator would want stopping in both cases. Measuring from
the start would let early profit fund an arbitrarily long decline.

The peak is of realised session equity, so it moves only on closed sessions --
there is no intraday mark to argue with.

STOPPING IS STICKY, AND CLEARING IT IS MANUAL
=============================================
When the limit trips, the bot refuses to start and keeps refusing. It does not
un-trip because the next session would have been profitable, and it does not
reset at month end. Clearing requires a human to write an acknowledgement into
the halt file, for the same reason arming the live gate does: the decision to
keep going after losing a tenth of the account is exactly the decision that
should cost somebody thirty seconds of deliberate typing.

That is the whole design. The mechanism is trivial; the property that matters is
that it cannot clear itself.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kabali.config import STATE_DIR

#: Written by hand into the halt file to resume after a trip.
RESUME_PHRASE = "I HAVE REVIEWED THE LOSSES AND CHOOSE TO CONTINUE"

HALT_PATH = STATE_DIR / "CUMULATIVE_HALT.json"


class CumulativeHalt(RuntimeError):
    """Trading is stopped by the cumulative loss limit. Never caught internally."""


@dataclass(frozen=True)
class CumulativeVerdict:
    tripped: bool
    peak: float
    equity: float
    drawdown: float
    limit: float
    sessions: int
    worst_session: float
    detail: str

    def render(self) -> str:
        head = "CUMULATIVE LIMIT: TRIPPED" if self.tripped else "CUMULATIVE LIMIT: ok"
        return (f"{head}\n"
                f"  sessions        {self.sessions}\n"
                f"  realised equity Rs {self.equity:+,.0f}  (peak Rs {self.peak:+,.0f})\n"
                f"  drawdown        Rs {self.drawdown:,.0f} of Rs {self.limit:,.0f} "
                f"allowed ({self.drawdown / self.limit * 100:.0f}%)\n"
                f"  worst session   Rs {self.worst_session:,.0f}\n"
                f"  {self.detail}")


def evaluate(record: pd.DataFrame | None, capital: float,
             limit_pct: float) -> CumulativeVerdict:
    """Score a session record against the drawdown limit.

    An empty record is NOT a trip: a bot that has never traded has lost nothing.
    """
    limit = capital * limit_pct / 100.0
    if record is None or record.empty or "net_pnl" not in record:
        return CumulativeVerdict(False, 0.0, 0.0, 0.0, limit, 0, 0.0,
                                 "no sessions recorded yet")

    df = record.copy()
    if "trade_date" in df:
        df = df.sort_values("trade_date")
    curve = df["net_pnl"].astype(float).cumsum()
    peak = float(max(curve.max(), 0.0))          # never below the starting point
    equity = float(curve.iloc[-1])
    drawdown = peak - equity
    worst = float(df["net_pnl"].min())

    if drawdown >= limit:
        detail = (f"drawdown of Rs {drawdown:,.0f} from the peak has reached the "
                  f"Rs {limit:,.0f} limit ({limit_pct}% of Rs {capital:,.0f}). "
                  f"Trading is stopped until a human clears it.")
        return CumulativeVerdict(True, peak, equity, drawdown, limit,
                                 len(df), worst, detail)

    room = limit - drawdown
    at_rate = df["net_pnl"].mean()
    sessions_left = int(room / abs(at_rate)) if at_rate < 0 else None
    detail = (f"Rs {room:,.0f} of room remains"
              + (f"; at the current average of Rs {at_rate:,.0f}/session that is "
                 f"about {sessions_left} more session(s)." if sessions_left is not None
                 else "."))
    return CumulativeVerdict(False, peak, equity, drawdown, limit,
                             len(df), worst, detail)


# ------------------------------------------------------------------ the halt

def read_halt(path: Path | None = None) -> dict:
    p = Path(path or HALT_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # An unreadable halt file is treated as a halt. The failure mode of
        # guessing wrong in the other direction is trading through a stop.
        return {"tripped": True, "reason": "halt file is unreadable"}


def trip(verdict: CumulativeVerdict, path: Path | None = None) -> Path:
    """Record the halt so it survives the process that discovered it."""
    p = Path(path or HALT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "tripped": True,
        "tripped_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "drawdown": round(verdict.drawdown, 2),
        "limit": round(verdict.limit, 2),
        "peak": round(verdict.peak, 2),
        "equity": round(verdict.equity, 2),
        "sessions": verdict.sessions,
        "reason": verdict.detail,
        "resume": "",
        "how_to_resume": (
            f"Set 'resume' to exactly {RESUME_PHRASE!r} after reviewing the "
            f"losses. Deleting this file also works and records nothing, which "
            f"is the point of not doing it that way."),
    }, indent=2), encoding="utf-8")
    return p


def cleared(path: Path | None = None) -> bool:
    """True when a human has written the acknowledgement."""
    blob = read_halt(path)
    if not blob or not blob.get("tripped"):
        return True
    return str(blob.get("resume", "")).strip() == RESUME_PHRASE


def require_clear(record: pd.DataFrame | None, capital: float, limit_pct: float,
                  path: Path | None = None) -> CumulativeVerdict:
    """Raise `CumulativeHalt` unless trading may continue.

    Checked BEFORE a session is built, so a stopped bot never reaches the point
    of holding a position it would then have to manage.
    """
    v = evaluate(record, capital, limit_pct)
    standing = read_halt(path)

    if standing.get("tripped") and not cleared(path):
        raise CumulativeHalt(
            "trading is stopped by a standing cumulative halt.\n"
            f"  {standing.get('reason', '')}\n"
            f"  tripped at {standing.get('tripped_at', '?')}\n"
            f"  clear it by writing the acknowledgement into "
            f"{Path(path or HALT_PATH)}")

    if v.tripped:
        p = trip(v, path)
        raise CumulativeHalt(
            "CUMULATIVE LOSS LIMIT REACHED -- trading stopped.\n"
            + v.render() + f"\n  halt recorded at {p}")
    return v
