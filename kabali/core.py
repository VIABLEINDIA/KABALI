"""Bridge to the validated ULTIMATE research core.

KABALI is the live/paper trading layer. It deliberately does NOT reimplement the
things ULTIMATE already got right and pinned with tests:

    research.dhan_client   authenticated, throttled, retrying Dhan access
    research.datastore     bar fetching, caching, and the integrity audit
    research.costs         the Indian statutory cost model (STT, exchange, GST,
                           stamp) -- the single most common thing a naive bot
                           gets wrong
    research.instruments   scrip-master resolution
    research.backtest      the fill/stop/target engine the research ran on
    research.robustness    neighbourhood region labelling
    research.metrics       Sharpe / drawdown / profit factor

Importing rather than copying is a correctness decision. The cost model in
particular is fingerprinted into every stored research result; a divergent second
copy here would silently invalidate any comparison between what KABALI trades and
what ULTIMATE validated.

The read-only guarantee travels with the import. `research.dhan_client` wraps the
SDK in `ReadOnlyDhan`, which raises `OrderPlacementBlocked` on any method outside
its allowlist. KABALI's order path therefore cannot go through this bridge at all
-- it constructs its own client in `kabali.execution.dhan_live`, behind the arming
gate. That separation is the point: research code physically cannot place an order.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_ULTIMATE_ROOT = Path("D:/ULTIMATE")


class CoreUnavailable(RuntimeError):
    """The ULTIMATE research core could not be located or imported."""


def ultimate_root() -> Path:
    """Resolve the ULTIMATE checkout, env override first."""
    raw = os.environ.get("KABALI_ULTIMATE_ROOT")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_ULTIMATE_ROOT


def ensure_core_importable() -> Path:
    """Put ULTIMATE on sys.path so `import research.*` resolves.

    Raises rather than falling back to a stub: a bot that silently ran without
    the audited cost model would produce plausible, wrong numbers.
    """
    root = ultimate_root()
    if not (root / "research" / "__init__.py").exists():
        raise CoreUnavailable(
            f"ULTIMATE research core not found at {root}. "
            f"Set KABALI_ULTIMATE_ROOT to the checkout containing research/."
        )
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


ensure_core_importable()

from research.costs import PRESETS as COST_PROFILES           # noqa: E402
from research.costs import CostModel, get_cost_model           # noqa: E402
from research.datastore import audit_bars, load_bars, participation  # noqa: E402
from research.dhan_client import (                            # noqa: E402
    DhanBridge,
    DhanDataError,
    OrderPlacementBlocked,
    normalise_candles,
)
from research.instruments import Instrument, ScripMaster, SymbolNotFound  # noqa: E402
from research.metrics import Metrics, compute_metrics, max_drawdown_pct, sharpe_ratio  # noqa: E402
from research.backtest import BacktestResult, Trade, run_backtest  # noqa: E402
from research.indicators import (  # noqa: E402
    assert_causal, atr, bollinger_bands, ema, realised_vol,
    rolling_percentile, rsi, sma, true_range,
)

__all__ = [
    "BacktestResult",
    "COST_PROFILES",
    "CostModel",
    "CoreUnavailable",
    "DhanBridge",
    "DhanDataError",
    "Instrument",
    "OrderPlacementBlocked",
    "ScripMaster",
    "SymbolNotFound",
    "Metrics",
    "Trade",
    "assert_causal",
    "atr",
    "audit_bars",
    "bollinger_bands",
    "ema",
    "compute_metrics",
    "get_cost_model",
    "ensure_core_importable",
    "load_bars",
    "normalise_candles",
    "max_drawdown_pct",
    "participation",
    "realised_vol",
    "rolling_percentile",
    "rsi",
    "sma",
    "true_range",
    "run_backtest",
    "sharpe_ratio",
    "ultimate_root",
]
