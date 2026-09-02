# Pre-registration: cross-sectional momentum, second design

Written 2026-09-02, **after** the first cross-sectional run returned alpha
t = 0.46 and **before** any part of the test below was executed. The power
analysis in `scripts/power_xsection.py` was run first, by deliberate design: it
prints the *volatility* of the candidate test statistic and never forms its
mean. Volatility is a nuisance parameter -- knowing it sizes the experiment
without revealing the answer. The mean stays unmeasured until this file is
committed.

## Why the first result was not an answer

The first run reported, over 3.98 years and 606 names:

| | Strategy | Equal-weight benchmark |
|---|---|---|
| Total return | +215.2% | +145.1% |
| CAGR | 33.5% | 25.3% |
| Sharpe | 1.24 | **1.40** |
| Max drawdown | 38.0% | 22.4% |
| Beta vs benchmark | 1.18 (t vs 1 = **5.99**) | 1.00 |
| Alpha | +3.9%/yr (t = **0.46**) | -- |

Two things in that table matter more than the +70pp of excess return.

**The benchmark has the higher Sharpe.** Holding all 606 names equal-weight
delivered more return per unit of risk than picking ten by momentum. The
strategy's advantage came from carrying more risk in a market that rose, and it
was not compensated for the extra risk it took.

**Beta is 1.18 with t = 5.99.** That is the most statistically solid finding in
the entire run -- far stronger than anything about alpha. The portfolio's
outperformance is well explained by holding higher-beta names during a bull
market. This is the textbook signature of momentum in a trending market and it
is not evidence of selection skill.

But the null result is *also* not evidence of absence, and this is the part the
first design got wrong.

## The instrument was too coarse to read the quantity

Back out the standard error from the reported numbers: alpha 3.90%/yr at
t = 0.46 implies SE = 8.50%/yr, implying an idiosyncratic residual volatility of
**17.0%/yr** on a ten-name portfolio.

At that noise level, over 3.98 years:

- the smallest alpha detectable at t = 2 is **17.0%/yr**
- detecting a *realistic* 3.9%/yr alpha at t = 2 would take **75 years**

No momentum system in the literature produces 17%/yr of net alpha. The first
test could only ever have returned "significant" by getting lucky. Its null
result is the expected output of an underpowered design, and reports almost
nothing about whether the signal exists.

The diagnosis is concentration: a ten-name portfolio discards ~98% of the
cross-sectional information in a 606-name panel, then measures what is left
through the noisiest possible estimator.

## What more power actually buys, measured

From `scripts/power_xsection.py`, on the real panel (565 rankable names/day):

| Test statistic | Residual vol | SE over 3.9y | Detectable at t=2 | Years for a 5%/yr effect |
|---|---|---|---|---|
| 10-name long-only (v1) | 17.0%/yr | 8.50%/yr | 17.0%/yr | 75 |
| 30x30 long-short | 16.3%/yr | 8.29%/yr | 16.6%/yr | 43 |
| 60x60 long-short | 13.7%/yr | 6.96%/yr | 13.9%/yr | 30 |
| 100x100 long-short | 11.8%/yr | 6.01%/yr | 12.0%/yr | 22 |

**The honest conclusion is that this hypothesis cannot be settled on 3.9 years
of data by any test design.** Better statistics take the requirement from 75
years to 22. They do not take it to four. A design that pretends otherwise is
the same error as the first one, wearing better notation.

That fact determines everything below: the primary deliverable of this test is
an **effect size with a confidence interval**, not a trade/no-trade verdict.

## The precondition: the panel must get longer before the verdict runs

The binding constraint is data, not method. The current panel starts
2021-08-30 because that is the Dhan API's practical reach, and it holds only
names listed *today* -- so every company delisted between 2021 and 2026 is
missing from both strategy and benchmark.

Both problems have the same fix. NSE publishes daily bhavcopy archives going
back to the 1990s, as-of-the-date snapshots that include names later delisted.
Reconstructing the panel from bhavcopy would deliver roughly 20 years and
survivorship-free membership, moving the 100x100 test from "needs 22 years, has
3.9" to adequately powered.

**Pre-committed: the confirmatory test below does not run on the 3.9-year
panel.** Running it now would burn the hypothesis on a sample that cannot
answer it, and the result -- almost certainly "inconclusive" -- would be quoted
later as if it meant something.

## The test, fixed in advance

### Stage 1 -- does the signal exist? (research question, not a strategy)

Long-short, equal-weight, monthly formation, using the *existing* pre-committed
12-1 signal from `kabali/xsection/momentum.py`. No parameter is re-chosen.

- **Estimator A, decile spread.** Top 100 minus bottom 100 by momentum rank,
  among names passing the standing point-in-time screens (turnover >= Rs 1cr
  median over 60d, price >= Rs 10). Daily returns of that spread; test the mean
  against zero with Newey-West standard errors at 21 lags, because overlapping
  formation windows induce autocorrelation that plain OLS understates.
- **Estimator B, Fama-MacBeth.** Each month, cross-sectionally regress forward
  monthly return on momentum rank across all rankable names. Test the mean of
  the monthly slopes. This uses every name rather than the tails, and is the
  more efficient estimator if the relationship is monotonic.

Both are reported. They answer slightly different questions -- B assumes
monotonicity, A does not -- and disagreement between them is itself informative,
so neither is designated the winner after the fact. **A is primary** for the
decision rule below; B is reported alongside.

### Stage 2 -- is it tradable? (runs only if Stage 1 passes)

Long-only, because CNC has no shorting and stock-lending on Indian mid-caps is
too thin and expensive to treat as available. Long-only implementation of a
signal validated long-short, with:

- the beta problem addressed at the benchmark, not the portfolio: the
  comparison is against an equal-weight hold of the same names **levered to the
  strategy's realised beta**, so that beta exposure is no longer creditable as
  skill
- **the bar is Sharpe, not total return.** v1 beat the benchmark on return and
  lost on Sharpe. Pre-committed: the strategy must exceed benchmark Sharpe after
  costs, at the `equity_delivery` profile already fingerprinted into results.

### Robustness, reported as a spread and never selected from

Formation 12-1 (primary), 6-1, 9-1. Holding 21d (primary), 63d. Position counts
per the table above. Industry-neutral and risk-parity weighted variants.
**These are reported as a distribution. The primary specification is 12-1 /
21d / 100x100 and is fixed now.** If the primary fails and a variant passes,
the hypothesis has failed; the variant is a new hypothesis needing its own
sample.

## The decision rule, with all three outcomes named

Stated in advance so that "inconclusive" cannot be quietly rounded into either
of the other two.

**PASS** -- Estimator A's mean spread is positive with Newey-West t >= 2.0 on
the extended panel, *and* the sign holds in both halves of the sample split at
its midpoint, *and* Stage 2 beats the beta-matched benchmark's Sharpe after
costs.

**FAIL** -- A's mean spread is negative, or Stage 2 trails the beta-matched
benchmark on Sharpe. The hypothesis is dead and is not retried with different
parameters on this data.

**INCONCLUSIVE** -- |t| < 2.0. Expected outcome if run on anything shorter than
~20 years. Obliges: report the point estimate and its 95% interval, change
nothing about live trading, and treat the panel extension as the open work.
**Inconclusive is not a soft pass.** No capital is committed on it.

## Known biases, stated before the result

**Survivorship** -- the current panel holds today's listings. Delisted names are
absent from strategy and benchmark alike. Momentum tends to *sell* deteriorating
names before delisting, so their absence plausibly hurts the benchmark more than
the strategy, meaning the bias may flatter the strategy. Bhavcopy reconstruction
is the fix, and is the reason it is a precondition rather than a nicety.

**Listing recency** -- the 2021-2026 window contains an unusual number of recent
small-cap IPOs in a strong bull market, exactly the population where spurious
momentum appears. Pre-committed sub-test: repeat the primary with names
requiring >= 2 years of listed history at formation, reported alongside, not
instead.

**One regime** -- 2021-2026 on Indian equities is close to a single extended
bull market. Even 20 years of bhavcopy gives perhaps three genuinely distinct
regimes. Any PASS from this test is a statement about a small number of
independent macro episodes, and should be read that way.

**Momentum crash risk** -- cross-sectional momentum's known failure mode is
sharp losses at market rebounds off a bottom, when prior losers rally hardest.
A test window without such a rebound will overstate the strategy. Whether the
extended sample contains one is to be reported explicitly.

## What this file commits me to

That the next artefact produced on this hypothesis is a **longer panel**, not a
better backtest; that the confirmatory run happens once, after that, against
the rules above; and that its result is published here whatever it says --
including, most likely, that four years was never going to be enough and twenty
may not be either.
