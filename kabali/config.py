"""Typed configuration for the KABALI bot.

Config is loaded once and frozen. The risk block is validated on load rather
than at the point of use: a typo that removed the daily loss limit should stop
the process at startup, not be discovered when the limit fails to fire.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
STATE_DIR = ROOT / "state"
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"

for _d in (STATE_DIR, RUNS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class ConfigError(RuntimeError):
    """Configuration is absent, malformed, or internally inconsistent."""


def _time(raw: str, field_name: str) -> dt.time:
    try:
        hh, mm = str(raw).split(":")
        return dt.time(int(hh), int(mm))
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"{field_name}: expected HH:MM, got {raw!r}") from exc


@dataclass(frozen=True)
class AccountConfig:
    capital: float
    currency: str = "INR"
    assumed_mis_leverage: float = 4.0


@dataclass(frozen=True)
class RiskConfig:
    daily_loss_limit_pct: float
    daily_profit_lock_pct: float
    per_trade_risk_pct: float
    max_concurrent_positions: int
    max_gross_exposure_mult: float
    max_position_notional_pct: float
    max_trades_per_day: int
    max_consecutive_losses: int
    min_position_notional: float
    #: Drawdown from peak realised equity, across sessions, at which the bot
    #: stops entirely. The daily limit caps one bad day; this caps a run of
    #: mediocre ones, which is the shape a losing system actually takes.
    cumulative_loss_limit_pct: float = 10.0

    def validate(self) -> None:
        if self.daily_loss_limit_pct <= 0:
            raise ConfigError("risk.daily_loss_limit_pct must be > 0; there is no unlimited mode")
        if self.per_trade_risk_pct <= 0:
            raise ConfigError("risk.per_trade_risk_pct must be > 0")
        if self.max_concurrent_positions < 1:
            raise ConfigError("risk.max_concurrent_positions must be >= 1")
        if self.cumulative_loss_limit_pct <= 0:
            raise ConfigError(
                "risk.cumulative_loss_limit_pct must be > 0; there is no unlimited mode")
        if self.cumulative_loss_limit_pct <= self.daily_loss_limit_pct:
            raise ConfigError(
                f"risk.cumulative_loss_limit_pct ({self.cumulative_loss_limit_pct}%) must "
                f"exceed the daily limit ({self.daily_loss_limit_pct}%), or one bad day "
                f"stops the bot permanently and the cumulative limit means nothing.")
        # The structural check: if every open position stopped out at once, the
        # loss must not exceed the daily budget. Otherwise the daily limit is
        # decorative -- it could only fire after it had already been breached.
        worst_case = self.per_trade_risk_pct * self.max_concurrent_positions
        if worst_case > self.daily_loss_limit_pct * 1.0001:
            raise ConfigError(
                f"risk config is incoherent: {self.max_concurrent_positions} positions at "
                f"{self.per_trade_risk_pct}% each risks {worst_case}% simultaneously, above "
                f"the {self.daily_loss_limit_pct}% daily limit. Lower per_trade_risk_pct or "
                f"max_concurrent_positions."
            )

    def daily_loss_limit(self, capital: float) -> float:
        return capital * self.daily_loss_limit_pct / 100.0

    def cumulative_loss_limit(self, capital: float) -> float:
        return capital * self.cumulative_loss_limit_pct / 100.0

    def daily_profit_lock(self, capital: float) -> float:
        return capital * self.daily_profit_lock_pct / 100.0

    def risk_per_trade(self, capital: float) -> float:
        return capital * self.per_trade_risk_pct / 100.0

    def gross_exposure_cap(self, capital: float) -> float:
        return capital * self.max_gross_exposure_mult

    def position_notional_cap(self, capital: float) -> float:
        return self.gross_exposure_cap(capital) * self.max_position_notional_pct / 100.0


@dataclass(frozen=True)
class SessionConfig:
    timezone: str
    market_open: dt.time
    market_close: dt.time
    opening_range_end: dt.time
    entry_cutoff: dt.time
    square_off: dt.time

    def validate(self) -> None:
        order = [
            ("market_open", self.market_open),
            ("opening_range_end", self.opening_range_end),
            ("entry_cutoff", self.entry_cutoff),
            ("square_off", self.square_off),
            ("market_close", self.market_close),
        ]
        for (an, a), (bn, b) in zip(order, order[1:]):
            if not a < b:
                raise ConfigError(f"session.{an} ({a}) must be before session.{bn} ({b})")


@dataclass(frozen=True)
class UniverseConfig:
    size: int
    candidate_series: tuple[str, ...]
    history_years: int
    min_price: float
    max_price: float
    min_median_turnover: float
    min_history_bars: int
    max_participation_pct: float


@dataclass(frozen=True)
class RegimeConfig:
    benchmark: str
    breadth_sample: int
    trend_fast: int
    trend_slow: int
    trend_anchor: int
    atr_window: int
    high_vol_percentile: float
    low_vol_percentile: float


@dataclass(frozen=True)
class CostsConfig:
    profile: str
    slippage_bps: float


@dataclass(frozen=True)
class SwingRiskConfig:
    """Risk limits for the swing venue.

    Deliberately a separate block from `risk:`, not a reuse of it. Two of the
    intraday limits are meaningless here and one is actively harmful:

    `daily_loss_limit` would flatten a multi-week book because of one bad
    session, which does not reduce the risk being taken -- it converts an open
    idea into a realised loss and pays the round trip to do it. The swing
    analogue is `drawdown_pause_pct`: stop OPENING positions when equity is far
    enough below its peak, and let the existing stops do their job.

    `max_gross_exposure_mult` defaults to 1.0 because CNC delivery is cash. The
    intraday block assumes MIS leverage, and carrying that assumption across
    would silently authorise a book this account cannot fund.

    The method names match `RiskConfig` so `size_position` can take either.
    """

    per_trade_risk_pct: float = 1.2
    max_concurrent_positions: int = 5
    max_gross_exposure_mult: float = 1.0
    max_position_notional_pct: float = 30.0
    min_position_notional: float = 5000.0
    drawdown_pause_pct: float = 12.0

    def validate(self) -> None:
        if self.per_trade_risk_pct <= 0:
            raise ConfigError("swing.risk.per_trade_risk_pct must be > 0")
        if self.max_concurrent_positions < 1:
            raise ConfigError("swing.risk.max_concurrent_positions must be >= 1")
        if self.max_gross_exposure_mult > 1.0:
            raise ConfigError(
                f"swing.risk.max_gross_exposure_mult is {self.max_gross_exposure_mult}, "
                "above 1.0. CNC delivery is settled in cash -- there is no leverage to "
                "borrow, so a gross book larger than capital cannot be funded."
            )
        if self.drawdown_pause_pct <= 0:
            raise ConfigError("swing.risk.drawdown_pause_pct must be > 0; there is no unlimited mode")
        worst_case = self.per_trade_risk_pct * self.max_concurrent_positions
        if worst_case > self.drawdown_pause_pct:
            raise ConfigError(
                f"swing risk config is incoherent: {self.max_concurrent_positions} positions "
                f"at {self.per_trade_risk_pct}% risks {worst_case}% at once, above the "
                f"{self.drawdown_pause_pct}% drawdown pause. The pause could then only fire "
                f"after a single simultaneous stop-out had already passed it."
            )

    def risk_per_trade(self, capital: float) -> float:
        return capital * self.per_trade_risk_pct / 100.0

    def gross_exposure_cap(self, capital: float) -> float:
        return capital * self.max_gross_exposure_mult

    def position_notional_cap(self, capital: float) -> float:
        return self.gross_exposure_cap(capital) * self.max_position_notional_pct / 100.0

    def drawdown_pause(self, peak_equity: float) -> float:
        return peak_equity * self.drawdown_pause_pct / 100.0


@dataclass(frozen=True)
class SwingRulesConfig:
    """The entry/exit rules, pre-registered in docs/swing_hypothesis.md.

    Every value here comes from Clenow, *Stocks on the Move* (2015). They are
    recorded in config so a run is reproducible and so a change to any of them
    is a visible edit rather than a quiet one -- but they are NOT knobs. The
    reason to adopt a published parameter set wholesale is that parameters
    chosen against this data would carry the selection pressure that produced
    zero robust regions from ULTIMATE's 6,036 intraday configurations.
    """

    rank_window: int = 90
    trend_ma: int = 100
    index_ma: int = 200
    gap_window: int = 90
    max_gap_pct: float = 15.0
    exit_rank_pct: float = 20.0
    atr_window: int = 20
    stop_atr_mult: float = 3.0
    rebalance_weekday: int = 2          # 0=Mon .. 4=Fri; Wednesday is Clenow's
    min_history_bars: int = 100
    turnover_window: int = 100

    def validate(self) -> None:
        if not 0 <= self.rebalance_weekday <= 4:
            raise ConfigError("swing.rules.rebalance_weekday must be 0 (Mon) .. 4 (Fri)")
        if self.rank_window < 3:
            raise ConfigError("swing.rules.rank_window must be >= 3")
        if self.stop_atr_mult <= 0:
            raise ConfigError("swing.rules.stop_atr_mult must be > 0; every position needs a stop")
        if not 0 < self.exit_rank_pct <= 100:
            raise ConfigError("swing.rules.exit_rank_pct must be in (0, 100]")
        if self.min_history_bars < max(self.rank_window, self.trend_ma):
            raise ConfigError(
                f"swing.rules.min_history_bars ({self.min_history_bars}) is below the longest "
                f"window it must support ({max(self.rank_window, self.trend_ma)}); names would "
                f"be ranked on NaN"
            )


@dataclass(frozen=True)
class SwingConfig:
    risk: SwingRiskConfig = field(default_factory=SwingRiskConfig)
    rules: SwingRulesConfig = field(default_factory=SwingRulesConfig)
    costs: CostsConfig = field(
        default_factory=lambda: CostsConfig(profile="equity_delivery", slippage_bps=3.0))

    def validate(self) -> None:
        self.risk.validate()
        self.rules.validate()
        if self.costs.profile == "equity_intraday":
            raise ConfigError(
                "swing.costs.profile is equity_intraday, but positions are held overnight. "
                "Delivery STT is 0.1% PER SIDE against intraday's 0.025% on the sell only; "
                "the intraday profile would understate the round trip by about half."
            )

    def summary(self, capital: float) -> str:
        r = self.risk
        return (f"swing: risk/position Rs {r.risk_per_trade(capital):,.0f} "
                f"({r.per_trade_risk_pct}%) | max {r.max_concurrent_positions} held | "
                f"gross cap Rs {r.gross_exposure_cap(capital):,.0f} | pause at "
                f"-{r.drawdown_pause_pct}% | costs {self.costs.profile}")


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str
    gate_file: str
    order_product: str = "INTRADAY"
    order_type: str = "MARKET"

    def validate(self) -> None:
        if self.mode not in ("paper", "live"):
            raise ConfigError(f"execution.mode must be paper or live, got {self.mode!r}")


@dataclass(frozen=True)
class BotConfig:
    account: AccountConfig
    risk: RiskConfig
    session: SessionConfig
    universe: UniverseConfig
    regime: RegimeConfig
    costs: CostsConfig
    execution: ExecutionConfig
    swing: SwingConfig = field(default_factory=SwingConfig)
    source: Path = field(default=CONFIG_DIR / "bot.yaml")

    @property
    def capital(self) -> float:
        return self.account.capital

    def gate_path(self) -> Path:
        p = Path(self.execution.gate_file)
        return p if p.is_absolute() else ROOT / p

    def summary(self) -> str:
        r, c = self.risk, self.capital
        return (
            f"capital Rs {c:,.0f} | daily stop Rs {r.daily_loss_limit(c):,.0f} "
            f"({r.daily_loss_limit_pct}%) | risk/trade Rs {r.risk_per_trade(c):,.0f} "
            f"| max {r.max_concurrent_positions} positions | gross cap "
            f"Rs {r.gross_exposure_cap(c):,.0f} | mode {self.execution.mode.upper()}"
        )


REQUIRED_SECTIONS = ("account", "risk", "session", "universe", "regime", "costs", "execution")


def load_config(path: Path | str | None = None) -> BotConfig:
    path = Path(path) if path else CONFIG_DIR / "bot.yaml"
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    missing = [k for k in REQUIRED_SECTIONS if k not in raw]
    if missing:
        raise ConfigError(f"config {path} is missing section(s): {', '.join(missing)}")

    a, rk, s = raw["account"], raw["risk"], raw["session"]
    u, rg = raw["universe"], raw["regime"]
    co, ex = raw["costs"], raw["execution"]

    account = AccountConfig(
        capital=float(a["capital"]),
        currency=a.get("currency", "INR"),
        assumed_mis_leverage=float(a.get("assumed_mis_leverage", 4.0)),
    )
    if account.capital <= 0:
        raise ConfigError("account.capital must be > 0")

    risk = RiskConfig(
        daily_loss_limit_pct=float(rk["daily_loss_limit_pct"]),
        daily_profit_lock_pct=float(rk.get("daily_profit_lock_pct", 1e9)),
        per_trade_risk_pct=float(rk["per_trade_risk_pct"]),
        max_concurrent_positions=int(rk["max_concurrent_positions"]),
        max_gross_exposure_mult=float(rk["max_gross_exposure_mult"]),
        max_position_notional_pct=float(rk["max_position_notional_pct"]),
        max_trades_per_day=int(rk["max_trades_per_day"]),
        max_consecutive_losses=int(rk["max_consecutive_losses"]),
        min_position_notional=float(rk.get("min_position_notional", 0.0)),
    )
    risk.validate()

    session = SessionConfig(
        timezone=s.get("timezone", "Asia/Kolkata"),
        market_open=_time(s["market_open"], "session.market_open"),
        market_close=_time(s["market_close"], "session.market_close"),
        opening_range_end=_time(s["opening_range_end"], "session.opening_range_end"),
        entry_cutoff=_time(s["entry_cutoff"], "session.entry_cutoff"),
        square_off=_time(s["square_off"], "session.square_off"),
    )
    session.validate()

    universe = UniverseConfig(
        size=int(u["size"]),
        candidate_series=tuple(str(x).upper() for x in u.get("candidate_series", ["EQ"])),
        history_years=int(u.get("history_years", 5)),
        min_price=float(u["min_price"]),
        max_price=float(u["max_price"]),
        min_median_turnover=float(u["min_median_turnover"]),
        min_history_bars=int(u.get("min_history_bars", 500)),
        max_participation_pct=float(u.get("max_participation_pct", 1.0)),
    )

    regime = RegimeConfig(
        benchmark=rg.get("benchmark", "NIFTY"),
        breadth_sample=int(rg.get("breadth_sample", 100)),
        trend_fast=int(rg["trend_fast"]),
        trend_slow=int(rg["trend_slow"]),
        trend_anchor=int(rg["trend_anchor"]),
        atr_window=int(rg.get("atr_window", 14)),
        high_vol_percentile=float(rg.get("high_vol_percentile", 80)),
        low_vol_percentile=float(rg.get("low_vol_percentile", 20)),
    )

    costs = CostsConfig(profile=co["profile"], slippage_bps=float(co.get("slippage_bps", 5.0)))
    execution = ExecutionConfig(
        mode=str(ex.get("mode", "paper")).lower(),
        gate_file=ex.get("gate_file", "state/LIVE_GATE.json"),
        order_product=ex.get("order_product", "INTRADAY"),
        order_type=ex.get("order_type", "MARKET"),
    )
    execution.validate()

    # `swing:` is optional. A config written before the swing venue existed
    # still loads, and gets the pre-registered defaults rather than an error --
    # the intraday paths do not read this block at all.
    sw = raw.get("swing") or {}
    sw_risk, sw_rules, sw_costs = sw.get("risk") or {}, sw.get("rules") or {}, sw.get("costs") or {}
    swing = SwingConfig(
        risk=SwingRiskConfig(
            per_trade_risk_pct=float(sw_risk.get("per_trade_risk_pct", 1.2)),
            max_concurrent_positions=int(sw_risk.get("max_concurrent_positions", 5)),
            max_gross_exposure_mult=float(sw_risk.get("max_gross_exposure_mult", 1.0)),
            max_position_notional_pct=float(sw_risk.get("max_position_notional_pct", 30.0)),
            min_position_notional=float(
                sw_risk.get("min_position_notional", risk.min_position_notional)),
            drawdown_pause_pct=float(sw_risk.get("drawdown_pause_pct", 12.0)),
        ),
        rules=SwingRulesConfig(
            rank_window=int(sw_rules.get("rank_window", 90)),
            trend_ma=int(sw_rules.get("trend_ma", 100)),
            index_ma=int(sw_rules.get("index_ma", 200)),
            gap_window=int(sw_rules.get("gap_window", 90)),
            max_gap_pct=float(sw_rules.get("max_gap_pct", 15.0)),
            exit_rank_pct=float(sw_rules.get("exit_rank_pct", 20.0)),
            atr_window=int(sw_rules.get("atr_window", 20)),
            stop_atr_mult=float(sw_rules.get("stop_atr_mult", 3.0)),
            rebalance_weekday=int(sw_rules.get("rebalance_weekday", 2)),
            min_history_bars=int(sw_rules.get("min_history_bars", 100)),
            turnover_window=int(sw_rules.get("turnover_window", 100)),
        ),
        costs=CostsConfig(
            profile=sw_costs.get("profile", "equity_delivery"),
            slippage_bps=float(sw_costs.get("slippage_bps", 3.0)),
        ),
    )
    swing.validate()

    return BotConfig(account=account, risk=risk, session=session, universe=universe,
                     regime=regime, costs=costs, execution=execution, swing=swing,
                     source=path)
