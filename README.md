# KABALI — intraday equity bot for NSE cash, on Dhan

Selects a daily 250-name high-momentum universe from five years of history,
classifies the market regime, runs a regime-gated multi-strategy intraday stack,
and executes through a risk engine sized to a fixed rupee budget.

**Paper by default. The live order path exists but is gated on evidence, and the
gate recomputes that evidence on every run.**

> Research and engineering output. Historical and simulated behaviour under the
> assumptions written down below. Not investment advice.

---

## Read this first

This bot was built on top of [`D:\ULTIMATE`](../ULTIMATE), a research engine that
already tested these hypotheses on Dhan data. What it found should set your
expectations:

| Finding | Value |
|---|---|
| Configurations tested across 8 venues | 6,036 |
| Parameter regions reaching ROBUST | **0** |
| Walk-forward folds positive | **84/212 (39.6%)** — a coin flip is 50% |
| Intraday 15m venue, by regime | negative in **every** regime bucket |
| Positive control (planted edge) | found it — 20 ROBUST regions |
| Negative control (pure noise) | rejected all 243 |

The controls matter: the engine finds a real edge when one is planted and
rejects noise completely, so those negatives are evidence that the hypotheses
failed, not that the pipeline is broken.

KABALI is therefore built as an **apparatus for finding out**, not as a machine
that assumes the edge is there. Every component that could manufacture a fake
edge — lookahead, optimistic fills, uncosted trades, unbounded risk — is closed
off deliberately, and the live gate will not open on hope.

## Install

```bash
pip install -r requirements.txt
```

Requires the ULTIMATE research core. Default path `D:/ULTIMATE`, override with
`KABALI_ULTIMATE_ROOT`. Credentials come from the environment only:
`DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` (plus `DHAN_PIN` / `DHAN_TOTP_SECRET` for
token refresh).

```bash
python cli.py health
```

## Use

```bash
python scripts/build_universe.py                      # 5y ingest + 250 selection (~35 min cold)
python cli.py regime                                  # today's regime, and which rules it enables
python cli.py universe --show 25                      # the current universe
python scripts/run_paper.py --days 90 --symbols 120   # walk-forward paper replay
python scripts/analyze.py runs/paper_<stamp>          # why the P&L came out that way
python cli.py gate                                    # can this go live? (it will say no)
python scripts/run_live.py                            # forward session, paper, on live data
python scripts/daily.py                               # refresh + session + gate, one command

python scripts/run_xsection.py --random 200 --grid --controls   # cross-sectional study
```

`scripts/daily.py` is the once-a-morning entry point. Windows Task Scheduler:

```bash
schtasks /create /tn KABALI /tr "python D:\KABALI\scripts\daily.py" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:45
```

Sizing note: `run_paper` holds every session in memory, so `--symbols 120` is
about the ceiling on a 6 GB machine. The *universe* is still 250 — the replay
scans a subset of it.

**Do not run two of these at once.** They share one Dhan rate limit, and the
penalty for breaching it is far worse than a brief backoff. Measured on a real
build: ingest ran at ~2.4 min per 100 symbols for the first thousand, then one
concurrent CLI call triggered `DH-904` — and the *next* hundred symbols took
**346 minutes**. The build finished 6.5 hours late with 265 symbols never
downloaded. Defaults are now 2.5 req/s across 3 workers, and `daily.py`
sequences its stages for this reason.

The live loop waits for **bar boundaries**, not a fixed timer. Polling every 30s
would re-download the scan list ten times per 5-minute bar — ~3.3 req/s all
session at 100 symbols, right back at the rate that caused the throttle above —
and an unfinished bar is not actionable at any polling frequency.

**Tokens expire in ~24 hours.** `DH-901` on every call means the token is stale,
not that anything is broken. Refresh from the PIN/TOTP already in your
environment:

```bash
python D:\ULTIMATE\cli.py login
```

## How it works

```
5y daily history (2,636 NSE EQ names)
        |
        v
  liquidity gates ............ turnover, price band, zero-volume, PARTICIPATION
        |
        v
  factor scoring ............. relative strength / trend cleanness / intraday
        |                      range / liquidity / volume expansion
        v
  regime classifier .......... NIFTY trend + ATR percentile + efficiency + breadth
        |                      -> direction (bull/bear/neutral) x family
        |                         (trend/range/volatile)
        v
  universe: top 250 .......... weights change WITH the regime; RS is signed in a
        |                      trend and unsigned in a range
        v
  strategy router ............ only rules valid in this regime family run
        |                      trend    -> ORB, VWAPPullback, MomentumBurst
        |                      range    -> VWAPReversion, GapFade
        |                      volatile -> MomentumBurst only
        v
  risk engine ................ stop-derived sizing, 4 caps, circuit breakers
        |
        v
  broker ..................... PaperBroker  |  DhanBroker (gated)
```

### The 250 are a scan list, not a portfolio

At ₹40,000 with a ₹800 daily loss budget and ₹200 of risk per trade, the book
holds at most **4 concurrent positions**. The width of the universe exists so the
strategies have enough candidates to find a few clean setups — not so the bot
holds 250 names.

### What "high momentum" means here

Not trailing return. A stock that rose 80% in a year via ten gap-ups and 240 flat
sessions has excellent momentum and nothing an intraday rule can capture. The
score blends four things an intraday strategy can actually use: **range** (enough
travel to clear ~15 bps of cost), **cleanness** (Kaufman efficiency and
regression fit, so the travel trends rather than chops), **liquidity**, and a
**directional tilt** that the regime points long or short.

Overnight gap is explicitly *subtracted* from the range factor. A bot flat by
15:10 cannot capture it, and ranking on raw ATR would favour names that look
volatile and sit still during the session.

### The regime test is a percentile, not a constant

The family test asks whether the index is travelling *more directionally than it
usually does*, measured as the percentile of its own trailing 50-bar efficiency
ratio, plus an absolute floor of 0.10 to reject dead chop outright.

That is a correction. The first version used a flat efficiency floor of 0.30,
which turned out to sit at NIFTY's **90th percentile** — median 0.126, p75 0.202,
p90 0.298. Only 2.3% of 556 real sessions classified as `trend`, and since three
of the five strategies are gated to that family, most of the stack could never
fire. The router looked complete and was mostly unreachable.

| Family | Flat 0.30 floor | Percentile (p65) |
|---|---|---|
| range | 67.8% | 47.8% |
| volatile | 29.9% | 29.9% |
| **trend** | **2.3%** | **22.3%** |

A percentile survives a change of index, window length, or volatility regime. A
hand-picked constant survives none of them. `TestRegimeCalibration` asserts every
family holds at least 5% of real sessions, so this cannot silently regress.

## Risk model

| Limit | Value at ₹40,000 |
|---|---|
| Daily loss limit (hard halt, flattens) | ₹800 (2%) |
| Daily profit lock (stops new entries) | ₹1,200 (3%) |
| Risk per trade (entry → stop) | ₹200 (0.5%) |
| Max concurrent positions | 4 |
| Gross exposure cap | ₹120,000 (3×) |
| Max single position | ₹36,000 |
| Max trades/day | 12 |
| Consecutive-loss halt | 4 |
| Min position notional | ₹5,000 |

Config load *rejects* an incoherent risk block: positions × per-trade risk may
not exceed the daily limit, or the daily limit could only fire after it had
already been breached. 4 × 0.5% = 2.0% exactly.

Position size is derived from the **stop**, never the price, so two setups on
different stocks carry the same rupee risk:

```
quantity = risk_per_trade / (|entry - stop| + entry * round_trip_cost_rate)
```

The cost term is not a refinement — it is what makes the daily limit hold. Costs
leave the account exactly as adverse price movement does, and a paper replay
proved it: sizing on price risk alone put 4 × ₹200 of risk plus 4 × ~₹21 of costs
on the book, settled at −₹884 against an ₹800 limit, and the gate flagged it as a
risk-engine bug. It was one.

Four independent caps can only ever *reduce* the result — risk budget, notional,
gross exposure, and participation (our order as a share of median daily volume,
the gate ULTIMATE found matters most).

Positions are **re-sized at the fill price**. The sizer runs on the signal bar's
close but the fill happens at the next bar's open; on a 5-minute bar those differ,
and an unadjusted quantity carries more than its share of the daily budget.

The daily limit is measured on **liquidation value**, not marks — open positions
are valued net of what it would cost to close them. On marks the breaker trips at
exactly −₹800 and then pays ~₹24 per position to get out, settling past the limit
it was meant to enforce.

A position below ₹5,000 is **rejected, not shrunk**: it pays the same fixed
brokerage as a ₹20,000 one, so the small version needs several times the move to
break even.

## What stops this from lying to itself

| Failure | How it is closed off |
|---|---|
| Lookahead on the signal bar | entries fill at the **next bar's open**, never the signal bar's close |
| Lookahead in indicators | all via `research.indicators`, causality pinned by truncation tests |
| Lookahead in selection | regime **and** universe recomputed as-of each replay date |
| Forming-bar lookahead (live) | `LiveSession._confirmed` drops any candle whose interval has not elapsed |
| Optimistic fills | stop wins when one bar contains both stop and target |
| Uncosted trades | full statutory model from `research.costs`, charged at entry and exit |
| Symmetric slippage | charged **against** the order on both legs |
| Unbounded risk | circuit breakers on equity P&L, including unrealised |
| Overnight drift | unconditional square-off; a test asserts the book is empty |
| Risk limits that do not bind | sizing is cost-inclusive; equity is liquidation value; reservations consume budget within a timestep |
| Silent live trading | `DhanBroker.__init__` demands a passing `GateVerdict` |
| Evidence from code that no longer exists | the record carries code and config fingerprints; the gate refuses a mismatch |

## The live gate

`execution.mode: live` is necessary and nowhere near sufficient. The gate
recomputes its criteria from `state/paper_record.csv` on **every** run, so a
config edit cannot by itself risk money.

| Criterion | Threshold |
|---|---|
| Sessions | ≥ 20 |
| Round trips | ≥ 40 |
| Net P&L after costs | > 0 |
| Profit factor | ≥ 1.2 (not 1.0 — no margin otherwise) |
| Observed ÷ modelled slippage | ≤ 1.5, and measured against the bars, not the model |
| Daily limit breaches | 0 (a breach is a risk-engine bug) |
| Evidence age | ≤ 14 days |
| Provenance | sidecar present, fingerprints match the installed code and loaded config |

### Promotion

A replay writes only into its own `runs/` directory. Making it the gate's
evidence is a separate, explicit act:

```
python scripts/run_paper.py --days 60 --symbols 250 --promote
```

Promotion writes `state/paper_record.csv` together with
`state/paper_record.meta.json`, which records the source run, the digest of the
trading code (`kabali/risk`, `kabali/strategies`, `kabali/regime`,
`kabali/universe`, the session engine and the paper broker) and the digest of
the config sections that shape a session.

This exists because `runs/` held both `paper_final` and `paper_v3`, and nothing
said which one the gate was reading. It was `paper_v3` — the later run, despite
the name. `paper_final` had been replayed one minute before the
daily-loss-limit fix landed in `risk/circuit.py`, so its record described a risk
engine that no longer existed and reported a breach that was already fixed. Both
files looked equally authoritative. The fingerprints make that distinction
mechanical: evidence produced by code you have since changed fails the gate
instead of arguing for going live.

Plus `armed: true` and the exact phrase `I ACCEPT LIVE TRADING RISK` written by
hand into `state/LIVE_GATE.json`. That is friction placed where an irreversible
decision is made, not security.

**What the slippage check actually measures.** It used to compare `fill.slippage`
against the model that produced it: `PaperBroker` applies exactly the modelled
slippage, so the ratio was 1.00 by construction and the check could not fail —
yet it reported PASS, which reads as "fills were verified" at the moment someone
is deciding to risk money.

Observed is now measured against the bars instead. An entry is decided on a bar's
close and cannot fill until the next bar opens; that move is real, it is in the
data, and it is precisely what a flat `slippage_bps` is standing in for. The bars
are independent of the cost model, so they can falsify it.

    modelled = reference x qty x slippage_bps / 10,000
    observed = (fill_price - decision_price) x qty      # signed against the order

Exits measure nothing extra — a stop's trigger price *is* its reference, so there
is no decision-to-fill lag — and favourable moves are not clamped away, since
keeping only adverse ones would bias the ratio and fail a sound model. Each
session records a `slippage_source` of `modelled`, `market` or `broker`, and the
gate refuses to credit `modelled`.

**Measured, over 426 entries:** observed ₹2,906 against ₹7,613 modelled — a
ratio of **0.38**. The real move between deciding on a bar's close and filling at
the next open cost ₹6.82 per entry where the 5bps assumption charges ₹17.87, so
the model is roughly 2.6x conservative on this component. In 21 of 66 sessions
the gap was net *favourable* — the next bar opened in the trade's direction.

That makes the paper P&L slightly pessimistic, not optimistic, and it changes
nothing about the verdict: the stack loses ₹4,495 **before** costs, and no
slippage assumption fixes a system that loses gross.

A passing ratio establishes one thing: that the assumption covers the
decision-to-fill move. It does **not** cover what `PaperBroker` openly does not
simulate — market impact, partial fills, queue position, rejection, or a scrip
frozen at a circuit limit. Those still need live fills, so forward paper trading
on a live feed remains a genuine prerequisite.

## Layout

```
cli.py                     health / regime / universe / gate / signals
config/bot.yaml            every risk decision, in one file
kabali/
  core.py                  bridge to ULTIMATE (costs, data, indicators, backtest)
  config.py                typed config, validated on load
  data/       ingest.py    staged parallel history ingest, rate-limited
              cache.py     offline parquet reads (post-ingest stages never fetch)
              intraday.py  session splitting, as-of daily context
  universe/   factors.py   per-symbol factors, causal
              selector.py  gates + regime-aware cross-sectional scoring
  regime/     classifier.py  direction x family, from NIFTY + breadth
  strategies/ base.py      Signal (stop mandatory), SymbolContext, registry
              opening_range.py vwap_pullback.py momentum_burst.py
              vwap_reversion.py gap_fade.py registry.py (router)
  risk/       book.py sizing.py circuit.py
  execution/  broker.py paper.py gate.py dhan_live.py
  xsection/   panel.py     aligned daily panel + data-quality screen
              momentum.py  12-1 ranking, parameters pre-committed
              portfolio.py monthly rebalance, delivery costs, whole shares
              evaluate.py  alpha/beta split and the random-ranking null
              controls.py  synthetic panels with a planted edge, or none
  engine/     session.py   one day, replay-driven
              live.py      same stages, clock-driven
scripts/    build_universe.py   5y ingest + daily selection
            run_paper.py        walk-forward replay -> the paper record
            run_live.py         one forward session (paper, or gated live)
            daily.py            refresh + session + gate status
            analyze.py          payoff / breakeven diagnosis of a run
            run_xsection.py     cross-sectional momentum study
tests/      test_kabali.py      82 tests
            test_xsection.py    28 tests, weighted to lookahead and cash safety
```

## Reading a result

`scripts/analyze.py` reports, per strategy, the only three numbers that decide
whether a rule can work:

```
payoff ratio        average win / average loss, as realised
breakeven win rate  1 / (1 + payoff) -- the hit rate the rule NEEDS
actual win rate     the hit rate it GOT
```

The gap between the last two separates two very different diagnoses that a P&L
total conflates. A rule whose designed 1.7 reward:risk realises 1.05 is being
bled by square-offs closing positions mid-idea — an exit problem. A rule whose
payoff holds but whose win rate sits far below breakeven is wrong about
direction, and no exit tuning will save it.

**Reading that output and then tuning parameters until the number improves is
how 6,036 configurations produced zero robust regions.** If a rule fails, the
honest options are to retire it or to test a different hypothesis on data it has
not seen.

## What it actually did

Walk-forward paper replay, 66 sessions (2026-06-01 → 2026-09-01), 250 names
scanned daily, regime and universe recomputed as-of each morning. This is the
promoted record (`runs/paper_promote`, code `3d2f8a4b776d`) — the evidence the
gate scores. Reproduce with
`python scripts/run_paper.py --days 60 --symbols 250 --promote`.

| | |
|---|---|
| Sessions | 66 |
| Round trips | 426 |
| Win rate | 36.9% |
| **Gross P&L** | **−₹4,495** |
| Costs | ₹8,072 |
| **Net P&L** | **−₹12,568 (−31.4% on ₹40,000)** |
| Worst session | −₹755 (limit ₹800 — held) |
| Sessions breaching the daily limit | **0 / 66** |
| Halted sessions | 16 |

**The risk engine works. The strategies do not.**

Two separate readings, and conflating them is the mistake to avoid:

**It loses before costs.** Gross is −₹4,495, so this is not a fee problem with a
good signal underneath. Costs then nearly triple the loss — ₹8,072 over 426 round
trips is 20% of capital in friction across one quarter — but removing them
entirely still leaves a losing system.

**Every rule sits below its own breakeven win rate:**

| Strategy | Trades | Payoff | Needs | Got | Shortfall |
|---|---|---|---|---|---|
| VWAPReversion | 346 | 1.27 | 44.0% | 35.3% | −8.8pp |
| OpeningRangeBreakout | 53 | 1.06 | 48.6% | 39.6% | −9.0pp |
| VWAPPullback | 6 | 0.45 | 69.2% | 16.7% | −52.5pp |
| GapFade | 21 | 0.86 | 53.7% | 61.9% | +8.2pp, sample too small |

The whole stack needs 45.7% and gets 36.9%. GapFade is the only rule above its
own line and it has 21 trades; at that sample a coin flip clears 62% about one
time in ten. MomentumBurst did not fire at all in this replay, which is itself
the point — a rule whose sample vanishes when the universe changes was never
measuring anything stable.

**By regime** — negative in three of four buckets, the fourth being 3 sessions:

| Regime | Sessions | Net | Per session |
|---|---|---|---|
| neutral_range | 33 | −₹7,599 | −₹230 |
| bear_range | 20 | −₹3,094 | −₹155 |
| neutral_trend | 10 | −₹1,966 | −₹197 |
| bear_trend | 3 | +₹91 | +₹30 |

This reproduces ULTIMATE's intraday finding — *"every regime bucket is negative;
not regime-dependent, absent"* — on a different timeframe, a different universe,
and an independently written engine.

**Nothing here has been tuned to improve these numbers, and nothing should be.**
Reading the table and adjusting parameters until it turns positive is fitting to
66 sessions. That is exactly the process that produced zero robust regions from
6,036 configurations.

### Bugs this replay caught

Every one was in the risk engine, found by running it rather than by reading it:

| Bug | Symptom |
|---|---|
| Sizing ignored costs | 4 × ₹200 risk + 4 × ₹21 costs settled at −₹884 vs an ₹800 limit |
| Reservations reported zero risk | four entries on one bar all passed the budget check against zero open risk |
| Sized at signal price, filled at next open | a gap made real risk exceed sized risk |
| Equity measured on marks | breaker tripped at −₹800, then paid ₹72 to flatten |
| **Unrealised profit counted as budget** | three shorts up ₹230 read as room for a fourth; all four reversed; settled −₹934 |

The last one is the subtlest: paper gains are not budget, they are the thing that
disappears on the way to the stop. Entry admission is now judged only on realised
P&L plus the full entry-to-stop risk of everything open.

## The second hypothesis: cross-sectional momentum

The intraday result closed one question and opened another. Intraday lost *before*
costs and then paid 19% of capital a quarter in friction, so the natural next test
had to change both things at once: a different signal construction, and a holding
period long enough that friction stops dominating.

```bash
python scripts/run_xsection.py --random 200 --grid --controls
```

**What is different about it.** Every one of ULTIMATE's 6,036 configurations and
all five of KABALI's intraday rules are *time-series* rules -- each symbol judged
against its own history. This ranks symbols against *each other* and holds the
top ten, rebalanced monthly, long-only, on delivery. Nothing in either codebase
had tested a cross-sectional construction, and `research.backtest` cannot express
one: it takes a single instrument's bars. So `kabali/xsection/` is new.

**The frequency arithmetic, which was the point.** Delivery costs *more* per
trade than intraday -- 28bps round trip against 19bps -- but a monthly rebalance
does 311 round trips over four years where the intraday stack did 426 in one
quarter. Friction falls from 20% of capital per quarter to 5.6% over four years.

**The data was already on disk.** 608 NSE names carry the full five years from the
universe builder's staged ingest. No new fetch, no rate limit, and a window
spanning several regime eras rather than one quarter.

### Result: no demonstrable edge, and the reason is instructive

| | Strategy | Equal-weight panel | Difference |
|---|---|---|---|
| Total return | 215.2% | 145.1% | +70.1pp |
| CAGR | 33.5% | 25.3% | +8.2pp |
| Sharpe | 1.24 | **1.40** | **−0.16** |
| Max drawdown | 38.0% | **22.4%** | **+15.6pp** |
| Costs | ₹2,253 (5.6% over 4y) | — | |

The benchmark is an equal-weight hold of the *identical* 606 names over the
*identical* window, so it carries exactly the same survivorship bias and
subtracting it cancels most of that bias rather than arguing about its size.

Read only the first row and this looks like a win. The next two rows say
otherwise: more return, worse risk-adjusted return, and nearly double the
drawdown. Two further tests separate the explanations.

**Random-ranking null** — the same engine, the same eligibility mask, the same
costs, ranking names at random, 200 times:

| p5 | p25 | median | p75 | p95 | strategy |
|---|---|---|---|---|---|
| 41.7% | 64.9% | 89.1% | 122.0% | 171.1% | **215.2%** |

Momentum beats all 200 draws. Concentration alone does not explain it, so the
ranking is doing something.

**What the ranking is doing is buying beta.** Regressing daily strategy returns
on the benchmark splits the outperformance in two:

| | beta | alpha %/yr | t(alpha) |
|---|---|---|---|
| Negative control (no edge planted) | 0.98 | −12.3 | −2.40 |
| Positive control (edge 0.02) | 0.98 | +16.5 | **3.17** |
| Positive control (edge 0.03) | 0.98 | +26.1 | **5.01** |
| **Real NSE panel** | **1.18** | +3.9 | **0.46** |

Beta is 1.18 with t = 5.99 against 1.0 — overwhelming. Alpha is +3.9%/yr with
t = 0.46 — indistinguishable from zero. The strategy systematically picks
higher-beta names, and in a four-year bull market high beta beats random
selection without any skill being involved.

The controls are what make that null mean something. The same method finds a
*modest* planted edge at t = 3.17 and correctly reports negative alpha on pure
noise, so it is powered to see an edge of the size that would matter. Note also
that a planted edge leaves beta at 0.98 — real alpha does not inflate beta. Only
the real panel's does, which is the tell.

**Every configuration, not the best one:**

| N | rebal | total % | Sharpe | beta | alpha %/yr | t |
|---|---|---|---|---|---|---|
| 5 | 21d | 221.8 | 1.13 | 1.16 | 6.2 | 0.53 |
| 5 | 63d | 373.2 | 1.44 | 1.18 | 15.7 | 1.34 |
| 10 | 21d | 215.2 | 1.24 | 1.18 | 3.9 | 0.46 |
| 10 | 63d | 169.7 | 1.08 | 1.21 | −0.8 | −0.09 |
| 20 | 21d | 202.2 | 1.30 | 1.18 | 2.2 | 0.35 |
| 20 | 63d | 237.6 | 1.44 | 1.14 | 5.9 | 0.94 |
| 30 | 21d | 226.1 | 1.43 | 1.16 | 4.3 | 0.78 |
| 30 | 63d | 264.0 | 1.56 | 1.15 | 7.4 | 1.32 |
| 50 | 21d | 191.1 | 1.49 | 1.01 | 4.4 | 1.01 |
| 50 | 63d | 211.1 | 1.57 | 1.03 | 5.7 | 1.38 |

Alpha is positive in nine of ten cells and significant in none. The largest
t-statistic is 1.38, which is unremarkable as the best of ten attempts — that is
roughly what the null produces. Parameters were fixed before the first run and
this grid is reported whole, precisely so the 373% cell cannot be quietly
promoted into a headline.

### How this differs from the intraday verdict

The intraday result was **clearly negative** — it lost before costs and the
stack sat 8.8 points below its breakeven win rate. This one is **not proven**, which is a
different thing and should not be rounded to either "it works" or "it failed":

- the point estimate is positive and consistently so across configurations;
- it never reaches significance over this sample;
- survivorship bias flatters it, so the true figure is likely lower;
- four years, and one bull market, is one regime era again.

**The live gate stays closed, and no cross-sectional path to it exists.** Nothing
in `kabali/xsection/` can place an order; it is research code with no execution
surface at all.

What would actually settle it: a survivorship-corrected panel including delisted
names (a multi-hour ingest against the Dhan rate limit), and a window covering at
least one full bear market. Until then this is a hypothesis with a positive prior
and insufficient evidence — not an edge.

### Data defects this study caught

Both were found by looking at the data rather than by reading code, and both
would have manufactured signal rather than crashing:

| Defect | What it would have done |
|---|---|
| HATSUN prints ₹468 between two ₹900 sessions | `research.corporate_actions` reads it as a 2:1 split — its open-confirms check passes because the corrupt bar's *open* is corrupt too — and back-adjusts a year of history, showing the ranking a stock that doubled overnight |
| MOTHERSON and LLOYDSENGG interleave whole sessions at two price scales, 62 and 31 times | A name that periodically jumps 50% sits at the top of a momentum sort on the strength of a data defect |

The fix is a persistence test: a split is permanent, a bad print reverts. Names
with more than three reverting prints are dropped entirely; isolated ones are
masked to NaN. Across 606 names this drops 2 and masks 23 bars, and it must run
*before* corporate-action detection. 584 of 608 names were clean.

## Known limitations

- **Costs are modelled, not observed.** Verify `research/costs.py` rates against
  a current contract note before trusting any P&L here.
- **No market impact, partial fills, or exchange rejections** in paper. Circuit
  limits and scrip-level bans are not simulated at all.
- **MIS leverage is assumed, not queried.** Dhan sets it per scrip and changes it.
- **Survivorship.** The universe is drawn from names listed today.
- **Selection pressure.** Every parameter in `config/bot.yaml` is a choice that
  was not swept. Sweeping them would raise the chance the best-looking setting is
  noise — which is what ULTIMATE's 6,036 configurations demonstrated.
- **The replay window is one regime era.** June–August 2026 was `range` on ~78%
  of sessions and `volatile` on none. The volatile family — the one ULTIMATE
  found most expensive — is therefore **completely untested here**, and the trend
  family has a thin sample. A result from one era is not a result about the rule.
- **Intraday history is ~90 days at a practical fetch rate.** Testing an older
  regime era means a much longer ingest, and the rate limit makes that a
  multi-hour job, not a quick re-run.
- **A negative paper result is weaker evidence than a positive one would be.**
- **The cross-sectional panel is survivorship-biased.** Its 606 names are those
  listed and liquid today. For a momentum rule the bias runs upward, so the
  reported alpha is an overestimate and the null verdict is, if anything,
  generous to the strategy.
- **Cross-sectional t-statistics are plain OLS.** Daily returns are mildly
  autocorrelated and heteroskedastic, so the true standard errors are wider than
  reported and the alpha t of 0.46 is an optimistic reading, not a conservative
  one.
- **No dividends in the panel.** Total returns are understated by roughly the
  yield -- close to neutral across a cross-sectional sort, but not zero.
- **Four years is one regime era.** The cross-sectional window (2022-2026) was
  predominantly a bull market in Indian mid caps; 82% of panel names ended
  higher. Momentum has historically failed hardest at sharp reversals, and this
  sample contains none.
