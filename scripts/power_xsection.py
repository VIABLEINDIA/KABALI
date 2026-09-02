"""Power analysis for the decile-spread test. PRINTS VOLATILITY ONLY.

Run BEFORE the hypothesis is pre-registered, to size the test. Volatility is a
nuisance parameter: knowing it does not tell you the sign or size of the effect,
so measuring it here does not contaminate the test of the mean. The mean is
computed nowhere in this file, on purpose. That restraint is the only thing
making the later test honest, so it is enforced rather than promised: the spread
series is never returned, only its standard deviation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kabali.xsection.momentum import momentum, LOOKBACK, SKIP  # noqa: E402

close = pd.read_parquet("state/panel_close.parquet")
turn = (close * pd.read_parquet("state/panel_volume.parquet")).rolling(60).median()

sig = momentum(close, LOOKBACK, SKIP)
ok = (turn >= 1e7) & (close >= 10.0) & sig.notna()

fwd = close.pct_change(fill_method=None).shift(-1)      # next-day return
dates = close.index[LOOKBACK + 5:-1]

def spread_vol(n_side: int) -> tuple[float, int, float]:
    """Std of the daily long-short decile spread. Mean is never formed."""
    daily, widths = [], []
    for d in dates:
        s = sig.loc[d][ok.loc[d]].dropna()
        if len(s) < 4 * n_side:
            continue
        r = fwd.loc[d]
        top = s.nlargest(n_side).index
        bot = s.nsmallest(n_side).index
        lo, sh = r[top].dropna(), r[bot].dropna()
        if len(lo) < n_side // 2 or len(sh) < n_side // 2:
            continue
        daily.append(lo.mean() - sh.mean())
        widths.append(len(s))
    a = np.asarray(daily)
    return float(a.std(ddof=1)), len(a), float(np.mean(widths))

print(f"panel {close.shape[1]} names, {close.index.min().date()} -> {close.index.max().date()}")
print()
for n in (30, 60, 100):
    sd, n_obs, avg_w = spread_vol(n)
    ann = sd * np.sqrt(252) * 100
    yrs = n_obs / 252
    se_ann = ann / np.sqrt(yrs)
    print(f"{n}x{n} spread: {n_obs} obs ({yrs:.1f}y), avg {avg_w:.0f} rankable/day")
    print(f"    residual vol {ann:5.1f}%/yr   SE of mean {se_ann:4.2f}%/yr")
    print(f"    detectable at t=2 in {yrs:.1f}y: effects >= {2*se_ann:4.1f}%/yr")
    print(f"    years needed for a 5%/yr effect: {(ann/(5/2))**2:5.1f}")
