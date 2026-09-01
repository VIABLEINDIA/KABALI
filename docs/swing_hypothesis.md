# Pre-registration: cross-sectional swing momentum

Written **before** the first backtest was run, 2026-08-31. Nothing below was
chosen after seeing a result on this data. That is the whole point of the file:
the intraday venue died of a process that looked at 6,036 configurations and
found zero robust regions, and the only defence against repeating it is to fix
the rules in advance and publish whatever they produce.

## Why a new hypothesis rather than a repair of the old one

KABALI's intraday result is negative *before costs* (gross −₹3,078 over 64
sessions, 403 round trips), so it is not a friction problem with a good signal
underneath. Every rule sat below its own breakeven win rate. Tuning it is
fitting to one quarter of one regime era.

What changes here is the edge, the holding period, and the sample:

| | Intraday venue | This hypothesis |
|---|---|---|
| Edge | five intraday price patterns | cross-sectional relative strength |
| Hold | minutes to hours, flat by 15:10 | weeks, held overnight |
| Product | MIS (margin, intraday) | CNC delivery |
| Round trips | 403 per quarter | ~15–25 per quarter, expected |
| Cost per round trip | 14.6 bps | 28.2 bps (delivery STT is 0.1% *per side*) |
| Sample | 64 sessions, one era | ~4.2 years, multiple eras |

Delivery costs **twice as much per round trip**. The hypothesis is that trading
roughly 25× less often more than pays for it. That claim is arithmetic and is
settled by the run, not by argument.

## The rules

Implemented from Clenow, *Stocks on the Move* (2015), a published and widely
replicated system. Using someone else's fully specified parameters is
deliberate: parameters I picked myself, on this data, would carry exactly the
selection pressure this file exists to avoid.

**Ranking** — annualised exponential-regression slope over 90 trading days,
multiplied by the regression R². Highest score ranks first.

**Per-stock eligibility, all measured at the rebalance close:**
- close above its 100-day simple moving average
- no single-day gap larger than 15% in the last 90 days
- passes the standing liquidity floors already in `config/bot.yaml`
  (price ≥ ₹50, 100-day median turnover ≥ ₹5 crore)
- at least 100 bars of history

**Market filter** — no new entries while NIFTY is below its 200-day SMA. Open
positions are still managed normally.

**Rebalance** — weekly, on Wednesday. Wednesday is Clenow's own choice and is
kept for that reason. The five weekday variants are reported as a robustness
spread, *not* selected from.

**Exit** — a position leaves when any of: it falls out of the top 20% of the
ranked list, it closes below its 100-day SMA, it prints a >15% gap, or its
trailing stop is hit.

**Entry/exit timing** — every decision uses bars up to and including the
decision day's close; every fill happens at the *next* session's open, with
slippage. This is the sequencing the intraday replay had to be corrected to.

## The risk numbers, and why they are a different kind of choice

Edge parameters (what to buy) are fixed above and come from the published
system. Risk parameters (how much to lose) are properties of the account, not
of the market, and are set to fit ₹40,000:

- 1.2% of equity risked per position, entry to stop, costs included
- at most 5 concurrent positions → 6% of equity at risk if all stop at once
- cash only: gross exposure ≤ 1.0× capital (CNC has no MIS leverage)
- ₹5,000 minimum position notional, unchanged — below it fixed costs dominate
- trailing stop at 3× ATR20, which is also the sizing unit

**The intraday 2% daily loss circuit deliberately does not apply.** Flattening a
multi-week book because of one bad session would destroy the strategy being
tested. The swing analogue is a drawdown circuit: stop opening new positions
when equity is more than 12% below its peak. Existing positions keep their
stops.

## What would count as a result

Reported against two benchmarks, both bought and sold through the same cost
model: NIFTY buy-and-hold, and an equal-weight buy-and-hold of the same
eligible universe. **A long-only momentum system in a rising market will show a
profit that is not an edge.** The question is not whether net P&L is positive,
it is whether it beats the equal-weight benchmark after costs, and whether it
does so in more than one era.

Falsified if: net-of-cost return trails equal-weight buy-and-hold, or the
result depends on a single year, or the weekday-rebalance spread is wider than
the margin over the benchmark.

## Known bias, stated in advance

The bar cache holds names listed today. Companies delisted between 2021 and
2026 are absent from both the strategy and the benchmarks. This inflates both
sides; it does not obviously favour either, and the strategy-minus-benchmark
comparison is the number to read because of it.
