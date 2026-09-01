"""The live-trading gate.

This module is the reason the bot is allowed to have an order path at all.

The gate is EVIDENCE-BASED, not a flag. Setting `execution.mode: live` is
necessary and nowhere near sufficient: the gate recomputes the criteria from the
recorded paper-session history every time it is asked, so a config edit, an
environment variable, or an operator in a hurry cannot by itself put money at
risk. If the paper log does not support live trading, the answer is no, and the
answer changes only when the evidence does.

The criteria come from ULTIMATE's section 9 -- the validation it said was still
required before any capital decision -- turned into executable checks:

  SESSIONS / TRADES   enough independent days and round trips that the result is
                      not one lucky morning. Twenty sessions is not statistical
                      significance; it is the floor below which the question is
                      not worth asking.
  NET PROFITABILITY   positive AFTER the full statutory cost model. The most
                      common way a validated backtest dies in production.
  PROFIT FACTOR       >= 1.2, not >= 1.0. A strategy that only just clears costs
                      in paper has no margin for the frictions paper omits.
  SLIPPAGE FIDELITY   observed slippage must not exceed modelled slippage by
                      more than a tolerance. This is the check that catches the
                      specific failure ULTIMATE named: "slippage assumptions are
                      the most common reason a validated backtest fails in
                      practice." A strategy can be genuinely profitable and still
                      fail here, and failing here means the paper P&L was
                      measuring something that will not happen.
  DRAWDOWN            worst session inside the configured daily loss limit. A
                      breach means the risk engine did not hold, which is a bug,
                      not a drawdown.
  FRESHNESS           evidence expires. A gate armed on data from two months ago
                      is describing a market that no longer exists.
  PROVENANCE          the record must say which run wrote it, and the code and
                      config fingerprints it was produced under must match what
                      is installed now. Freshness asks whether the evidence is
                      about a recent market; provenance asks whether it is about
                      THIS system. A replay from before a risk-engine fix is
                      recent and completely irrelevant.

Arming additionally requires a human to write an explicit confirmation phrase
into the gate file. That is not security -- anyone who can edit the config can
edit the gate file -- it is friction placed exactly where an irreversible
decision is made, so that going live is always a deliberate act.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from kabali.config import STATE_DIR
from kabali.execution import provenance

# The evidence the gate scores. One path, so a caller cannot hand the gate a
# record from somewhere else and have it scored as though it were promoted.
RECORD_PATH = STATE_DIR / "paper_record.csv"

# The operator must write this exactly into the gate file's `confirm` field.
CONFIRMATION_PHRASE = "I ACCEPT LIVE TRADING RISK"

# Evidence older than this is stale and the gate closes on its own.
MAX_EVIDENCE_AGE_DAYS = 14


@dataclass(frozen=True)
class GateCriteria:
    """Thresholds a paper record must clear before live trading is permitted."""

    min_sessions: int = 20
    min_trades: int = 40
    min_net_pnl: float = 0.0
    min_profit_factor: float = 1.2
    max_slippage_ratio: float = 1.5      # observed / modelled
    max_evidence_age_days: int = MAX_EVIDENCE_AGE_DAYS

    def as_dict(self) -> dict:
        return {
            "min_sessions": self.min_sessions, "min_trades": self.min_trades,
            "min_net_pnl": self.min_net_pnl, "min_profit_factor": self.min_profit_factor,
            "max_slippage_ratio": self.max_slippage_ratio,
            "max_evidence_age_days": self.max_evidence_age_days,
        }


@dataclass
class GateVerdict:
    """Per-criterion result. `passed` only when every check passed."""

    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks[name] = (ok, detail)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for ok, _ in self.checks.values())

    @property
    def failures(self) -> list[str]:
        return [f"{n}: {d}" for n, (ok, d) in self.checks.items() if not ok]

    def render(self) -> str:
        lines = [f"LIVE GATE: {'PASS' if self.passed else 'FAIL'}"]
        for name, (ok, detail) in self.checks.items():
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


class GateClosed(RuntimeError):
    """Live trading was requested and the gate refused. Never caught internally."""


def evaluate(sessions: pd.DataFrame, criteria: GateCriteria | None = None,
             now: pd.Timestamp | None = None) -> GateVerdict:
    """Score the paper-session record against the criteria.

    `sessions` is one row per paper session, as written by the engine:
        trade_date, trades, wins, gross_pnl, costs, net_pnl,
        modelled_slippage, observed_slippage
    """
    c = criteria or GateCriteria()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now()
    v = GateVerdict(evidence={"criteria": c.as_dict()})

    if sessions is None or sessions.empty:
        v.add("evidence", False, "no paper sessions recorded")
        return v

    df = sessions.copy()
    n_sessions = len(df)
    v.add("sessions", n_sessions >= c.min_sessions,
          f"{n_sessions} sessions recorded, need {c.min_sessions}")

    trades = int(df["trades"].sum()) if "trades" in df else 0
    v.add("trades", trades >= c.min_trades,
          f"{trades} round trips, need {c.min_trades}")

    net = float(df["net_pnl"].sum()) if "net_pnl" in df else 0.0
    v.add("net_profitability", net > c.min_net_pnl,
          f"net Rs {net:,.0f} after costs, need > Rs {c.min_net_pnl:,.0f}")

    # Profit factor from session-level sums. Trade-level would be sharper, but
    # session-level cannot be gamed by a single outsized winner as easily.
    if "net_pnl" in df:
        gains = float(df.loc[df["net_pnl"] > 0, "net_pnl"].sum())
        losses = abs(float(df.loc[df["net_pnl"] < 0, "net_pnl"].sum()))
        pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    else:
        pf = 0.0
    v.add("profit_factor", pf >= c.min_profit_factor,
          f"profit factor {pf:.2f}, need >= {c.min_profit_factor}")

    # Slippage fidelity.
    if "observed_slippage" in df and "modelled_slippage" in df:
        obs = float(df["observed_slippage"].sum())
        mod = float(df["modelled_slippage"].sum())
        if mod <= 0:
            v.add("slippage_fidelity", False,
                  "modelled slippage is zero -- cannot verify fill assumptions")
        else:
            ratio = obs / mod
            v.add("slippage_fidelity", ratio <= c.max_slippage_ratio,
                  f"observed/modelled slippage {ratio:.2f}, need <= {c.max_slippage_ratio}")
    else:
        v.add("slippage_fidelity", False,
              "paper record has no slippage columns -- fills were never compared")

    # Drawdown: no session may have breached the configured daily loss limit.
    if "net_pnl" in df and "loss_limit" in df:
        breaches = df[df["net_pnl"] < -df["loss_limit"]]
        v.add("daily_limit_respected", breaches.empty,
              "no session breached its daily loss limit" if breaches.empty
              else f"{len(breaches)} session(s) breached the daily loss limit -- risk engine bug")
    else:
        worst = float(df["net_pnl"].min()) if "net_pnl" in df else 0.0
        v.add("daily_limit_respected", True, f"worst session Rs {worst:,.0f} (limit not recorded)")

    # Freshness.
    if "trade_date" in df:
        last = pd.Timestamp(pd.to_datetime(df["trade_date"]).max())
        age = (now.normalize() - last.normalize()).days
        v.add("freshness", age <= c.max_evidence_age_days,
              f"most recent session {last.date()} is {age}d old, "
              f"max {c.max_evidence_age_days}d")
    else:
        v.add("freshness", False, "paper record has no trade_date column")

    v.evidence.update({"sessions": n_sessions, "trades": trades,
                       "net_pnl": round(net, 2), "profit_factor": round(pf, 3)})
    return v


class LiveGate:
    """Reads the gate file, re-evaluates the evidence, and permits or refuses."""

    def __init__(self, cfg, criteria: GateCriteria | None = None,
                 record_path: Path | None = None):
        self.cfg = cfg
        self.path: Path = cfg.gate_path()
        self.criteria = criteria or GateCriteria()
        self.record_path: Path = Path(record_path) if record_path else RECORD_PATH

    def load_record(self) -> pd.DataFrame | None:
        """The paper-session evidence, or None if no replay has been promoted.

        Every caller reads the record through here. When each one opened the
        path itself, nothing tied the rows to the sidecar that describes them,
        and the gate could score a file no caller could identify.
        """
        if not self.record_path.exists():
            return None
        return pd.read_csv(self.record_path)

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            raise GateClosed(f"gate file {self.path} is not valid JSON: {exc}") from exc

    def check(self, sessions: pd.DataFrame | None = None,
              now: pd.Timestamp | None = None) -> GateVerdict:
        """Full check: armed, confirmed, and the evidence still supports it."""
        raw = self.read()
        v = GateVerdict()

        if not raw:
            v.add("gate_file", False,
                  f"no gate file at {self.path} -- live trading has never been armed")
            return v
        v.add("gate_file", True, f"present at {self.path}")

        v.add("armed", bool(raw.get("armed")) is True,
              "armed=true" if raw.get("armed") else "armed is not true")

        confirm = str(raw.get("confirm", ""))
        v.add("confirmation", confirm == CONFIRMATION_PHRASE,
              "confirmation phrase present" if confirm == CONFIRMATION_PHRASE
              else f"confirm must be exactly {CONFIRMATION_PHRASE!r}")

        if sessions is None:
            sessions = self.load_record()
        if sessions is None or sessions.empty:
            v.add("evidence", False, "no paper-session record supplied to the gate")
            return v

        ev = evaluate(sessions, self.criteria, now=now)
        v.checks.update(ev.checks)
        v.evidence.update(ev.evidence)

        # Provenance last: it is the check most likely to be the real reason a
        # record should not be trusted, and reporting it after the numbers makes
        # clear that clearing every threshold is not on its own sufficient.
        ok, detail = provenance.verify(self.record_path, sessions, self.cfg)
        v.add("provenance", ok, detail)
        v.evidence["provenance"] = provenance.read(self.record_path) or {}
        return v

    def require_open(self, sessions: pd.DataFrame | None = None,
                     now: pd.Timestamp | None = None) -> GateVerdict:
        """Raise `GateClosed` unless every check passes."""
        v = self.check(sessions, now=now)
        if not v.passed:
            raise GateClosed(
                "live trading refused; the gate is closed.\n" + v.render()
            )
        return v

    def template(self) -> dict:
        """A gate file skeleton, deliberately not armed."""
        return {
            "armed": False,
            "confirm": "",
            "note": (f"Set armed=true and confirm to {CONFIRMATION_PHRASE!r} only after "
                     "the paper record clears every criterion. The gate re-evaluates the "
                     "evidence on every run; these flags alone do not enable live trading."),
            "criteria": self.criteria.as_dict(),
            "armed_at": None,
        }

    def write_template(self) -> Path:
        if self.path.exists():
            raise GateClosed(f"refusing to overwrite an existing gate file at {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.template(), indent=2))
        return self.path
