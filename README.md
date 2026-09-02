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

**Both hypotheses it was built to test have now been answered, and neither
supports trading this account.**

| Hypothesis | Verdict |
|---|---|
| Intraday multi-strategy stack | **Negative.** −₹3,923 *before* costs over 66 sessions and 424 round trips. Every rule sits below its own breakeven win rate, so it is not a friction problem with a good signal underneath. |
| Cross-sectional momentum | **Real, but unreachable.** The signal is strong long-short (+28.12%/yr, t = 5.92 over 16.6 survivorship-free years) and fails long-only after costs (Sharpe 0.64 against the benchmark's 0.66). |

The benchmark in that second row is the finding worth carrying: an equal-weight
hold of every liquid NSE name returned **13.12%/yr at Sharpe 0.66** over the same
period, with no ranking, no turnover and no costs. It is the best-performing
strategy anywhere in this repository.

The live gate is closed. Nothing here recommends opening it.

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

python scripts/run_xsection.py --random 200 --grid --controls   # first cross-sectional study
```

The cross-sectional line, which is settled — see
[`docs/xsection_hypothesis.md`](docs/xsection_hypothesis.md):

```bash
python scripts/build_bhavcopy_panel.py --start 2006-01-01   # 20y panel (~1h cold, then seconds)
python scripts/power_xsection.py                            # how long a sample the test needs
python scripts/run_stage1.py                                # does the signal exist
python scripts/run_stage2.py --spread                       # is it tradable long-only
```

The panel download is one hour once and fully resumable; every day-file and
action window is cached, so rebuilding afterwards takes about four minutes with
`--no-fetch`. `run_stage1.py` **refuses** a panel shorter than 15 years and exits
2, because the power analysis shows a shorter one cannot settle the question.

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
docs/       swing_hypothesis.md      pre-registration: swing momentum
            xsection_hypothesis.md   pre-registration + result: cross-sectional
kabali/
  core.py                  bridge to ULTIMATE (costs, data, indicators, backtest)
  config.py                typed config, validated on load
  data/       ingest.py    staged parallel history ingest, rate-limited
              cache.py     offline parquet reads (post-ingest stages never fetch)
              intraday.py  session splitting, as-of daily context
              bhavcopy.py  NSE daily archive, three schemas, survivorship-free
              corpactions.py  split/bonus factors, parsed AND price-confirmed
              panel.py     identity across renames + ISIN changes; wide panels
  universe/   factors.py   per-symbol factors, causal
              selector.py  gates + regime-aware cross-sectional scoring
  regime/     classifier.py  direction x family, from NIFTY + breadth
  strategies/ base.py      Signal (stop mandatory), SymbolContext, registry
              opening_range.py vwap_pullback.py momentum_burst.py
              vwap_reversion.py gap_fade.py registry.py (router)
  risk/       book.py sizing.py circuit.py
  execution/  broker.py paper.py gate.py dhan_live.py provenance.py
  xsection/   panel.py     aligned daily panel + data-quality screen
              momentum.py  12-1 ranking, parameters pre-committed
              portfolio.py monthly rebalance, delivery costs, whole shares
              evaluate.py  alpha/beta split and the random-ranking null
              controls.py  synthetic panels with a planted edge, or none
              stage1.py    decile spread + Fama-MacBeth, Newey-West errors
              stage2.py    long-only tradability against a beta-matched hold
              neural.py    a neural cross-sectional model, and the null it returns
  swing/      strategy.py panel.py portfolio.py benchmark.py
  fundamentals/ store.py   valuation data for the traded head of the universe
  engine/     session.py   one day, replay-driven
              live.py      same stages, clock-driven
scripts/    build_universe.py       5y ingest + daily selection
            build_bhavcopy_panel.py 20y survivorship-free panel from NSE archive
            power_xsection.py       volatility-only power analysis (never the mean)
            run_paper.py            walk-forward replay -> the paper record
            run_live.py             one forward session (paper, or gated live)
            daily.py                refresh + session + gate status
            analyze.py              payoff / breakeven diagnosis of a run
            render_results.py       regenerate the README from the record
            run_xsection.py         first cross-sectional study
            run_stage1.py           does the signal exist (sample-length guarded)
            run_stage2.py           is it tradable long-only
            run_swing.py run_neural.py diagnose.py repair_cache.py
tests/      test_kabali.py          104 tests
            test_xsection.py         28 tests, weighted to lookahead and cash safety
            test_bhavcopy_panel.py   26 tests, weighted to silent-failure modes
            test_stage1.py           10 tests, estimators against known answers
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

<!-- BEGIN GENERATED RESULTS -- scripts/render_results.py -->

Walk-forward paper replay, 66 sessions (2026-06-01 → 2026-09-01), 250 names scanned daily, regime and universe recomputed as-of each morning. This is the promoted record (`runs\paper_final_repromote`, code `49c0b130e100`) — the evidence the gate scores. Reproduce with
`python scripts/run_paper.py --days 60 --symbols 250 --promote`.

| | |
|---|---|
| Sessions | 66 |
| Round trips | 424 |
| Win rate | 39.2% |
| **Gross P&L** | **−₹3,923** |
| Costs | ₹7,718 |
| **Net P&L** | **−₹11,641 (-29.1% on ₹40,000)** |
| Profit factor | 0.35 |
| Worst session | −₹763 (limit ₹800 — held) |
| Sessions breaching the daily limit | **0 / 66** |
| Halted sessions | 17 |

**It loses before costs.** Gross is −₹3,923, so this is not a fee problem with a good signal underneath. Costs then more than double the loss — ₹7,718 over 424 round trips is 19% of capital in friction across one quarter — but removing them entirely still leaves a losing system.

**Breakeven diagnosis — payoff against hit rate:**

| Strategy | Trades | Payoff | Needs | Got | Gap |
|---|---|---|---|---|---|
| VWAPReversion | 343 | 1.13 | 47.0% | 38.2% | -8.8pp |
| OpeningRangeBreakout | 54 | 1.30 | 43.4% | 38.9% | -4.5pp |
| GapFade | 17 | 0.48 | 67.5% | 64.7% | -2.8pp |
| VWAPPullback | 6 | 0.61 | 62.1% | 33.3% | -28.7pp |
| MomentumBurst | 4 | 2.91 | 25.6% | 25.0% | -0.6pp |
| **All strategies** | **424** | **1.10** | **47.6%** | **39.2%** | **-8.5pp** |

**By exit reason:**

| Reason | Trades | Net |
|---|---|---|
| stop | 190 | −₹35,142 |
| squareoff | 115 | −₹717 |
| target | 119 | ₹24,218 |

**By regime:**

| Regime | Sessions | Net | Per session |
|---|---|---|---|
| neutral_range | 33 | −₹7,444 | −₹226 |
| bear_range | 20 | −₹3,093 | −₹155 |
| neutral_trend | 10 | −₹1,195 | −₹119 |
| bear_trend | 3 | +₹91 | +₹30 |

<sub>Generated from `state/paper_record.csv` (promoted 2026-09-02T00:05, config `9fb4863b88c9`). Do not edit by hand — run `scripts/render_results.py --write`.</sub>

<!-- END GENERATED RESULTS -->

**The risk engine works. The strategies do not.** Two separate readings, and
conflating them is the mistake to avoid.

**Every rule sits below its own breakeven win rate.** The payoff column above is
the ratio of the average win to the average loss; `Needs` is the hit rate that
ratio requires to break even. Every rule is short of it, and the stack is short
by more than eight points. That gap does not close with parameter tuning: it is a
property of where these rules place their stops relative to their targets.

Two rules deserve a caveat rather than a reading. GapFade has clawed its way to a
65% win rate and still loses money, because its payoff is 0.48 — it needs 67.5%.
MomentumBurst fires four times in 66 sessions and vanished entirely on a slightly
different universe. Neither has a sample worth interpreting, and the first is a
standing reminder that a win rate on its own says nothing.

**By regime, the loss is not concentrated anywhere.** It is negative in three of
four buckets, the fourth being three sessions. This reproduces ULTIMATE's
intraday finding — *"every regime bucket is negative; not regime-dependent,
absent"* — on a different timeframe, a different universe, and an independently
written engine.

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

**What is different about it.** Every one of ULTIMATE's 6,036 configurations and
all five of KABALI's intraday rules are *time-series* rules — each symbol judged
against its own history. This ranks symbols against *each other*. Nothing in
either codebase had tested a cross-sectional construction, and `research.backtest`
cannot express one: it takes a single instrument's bars. So `kabali/xsection/` is
new.

This question is now **settled**, and settling it took three steps: a first run
whose null turned out to mean nothing, a twenty-year panel built to fix that, and
a two-stage pre-registered test. The pre-registration and its result are in
[`docs/xsection_hypothesis.md`](docs/xsection_hypothesis.md).

### Step 1: the first run, and why its null meant nothing

On 606 names over 3.9 years, a ten-name monthly-rebalanced book returned 215.2%
against 145.1% for an equal-weight hold of the identical names — but with a
*worse* Sharpe (1.24 against 1.40) and nearly double the drawdown. Regression
split the outperformance cleanly:

| | beta | alpha %/yr | t(alpha) |
|---|---|---|---|
| Negative control (no edge planted) | 0.98 | −12.3 | −2.40 |
| Positive control (edge 0.02) | 0.98 | +16.5 | **3.17** |
| **Real panel** | **1.18** | +3.9 | **0.46** |

Beta 1.18 with t = 5.99; alpha t = 0.46. It was recorded as "not proven".

**That reading gave the number more credit than it deserved.** Back out the
standard error: alpha 3.90%/yr at t = 0.46 implies SE 8.50%/yr, so a ten-name
portfolio carries 17.0%/yr of residual volatility. The smallest alpha that design
could ever have called significant was **17%/yr**, and detecting the alpha it
actually measured would have taken **75 years**. No momentum system in the
literature produces 17%/yr of net alpha. The test could only have passed by luck.

`scripts/power_xsection.py` measures the ceiling on fixing that statistically —
reporting the *volatility* of candidate estimators while never forming their mean,
so the experiment can be sized without revealing the answer. Even a beta-neutral
100×100 long-short spread needs **22 years** for a 5%/yr effect. Better statistics
take the requirement from 75 years to 22; they do not take it to four.

The binding constraint was data, not method.

### Step 2: the panel that made a test possible

`kabali/data/bhavcopy.py` builds a **survivorship-free 20-year panel** from NSE's
daily archive: 5,100 sessions × 3,847 companies, 2006-01-02 to 2026-09-01. It is
an as-of-the-date snapshot, so a company that vanished in 2009 is present in every
file printed before it vanished — **470 companies present in 2006 and gone by 2026
are retained**, exactly the names a today's-listings panel silently drops.

Four things had to be established by probing rather than assumed:

| Finding | Why it mattered |
|---|---|
| History reaches 2005, not 2019 | The archive answers unprimed clients with `403` and an empty body, indistinguishable from "no such file". Taking it at face value caps the panel at seven years. |
| Bhavcopy is unadjusted, and `prev_close` does not help | Measured: across 90,525 day-pairs it equals the raw prior close 99.98% of the time, including on every confirmed split. A 1:10 split enters a ranking as a −90% return. |
| A split changes the ISIN | FEDERALBNK is `INE171A01011` before its 2015 bonus and `INE171A01029` after. Keying on ISIN severs history at the actions being corrected for; symbols fail too (AEGISLOG → AEGISCHEM). Identity resolves as connected components over the (symbol, ISIN) graph. |
| NSE files an action under the ISIN it retires | So an ISIN-only join drops precisely the splits that matter most. Resolving to entity *before* confirming took the confirmation rate from 38.9% to 100% on the test quarter. |

A parsed corporate-action factor is treated as a hypothesis and confirmed against
the price jump actually observed on the ex-date; 95.3% confirm, 60 are
contradicted and **refused rather than applied**. A jump no action explains has the
name's prior history blanked rather than a factor guessed from the jump itself —
because a 1:2 split and a stock halving are the same number.

### Step 3: the pre-registered test

Two stages, both fixed in writing before either ran, with all three outcomes
(PASS / FAIL / INCONCLUSIVE) named in advance.

**Stage 1 — does the signal exist? PASS.**

```
decile spread 100x100  +28.12%/yr  t=+5.92 (Newey-West)  95% CI [+18.8, +37.4]
Fama-MacBeth           +1.32%/period  t=+2.05
halves                 first +33.14%/yr (t=+4.97), second +28.47%/yr (t=+4.63)
```

Long-short, so market beta is removed by construction rather than regressed away.
Against an equal-weight hold of the eligible universe (+14.46%/yr), the long leg
beats by **+12.05pp** and the short leg lags by **−16.07pp** — roughly symmetric,
on liquid names (median turnover ₹9.2cr and ₹6.4cr at formation). The tested span
is 16.6 years, not 20.7: a rebalance needs 400 eligible names and NSE does not
supply them before 2010.

**Stage 2 — is it tradable long-only? FAIL.**

| | Total | CAGR | Sharpe | maxDD |
|---|---|---|---|---|
| Strategy (N=10, ₹40k) | +1104.9% | +13.51% | **0.64** | 72.8% |
| Equal-weight eligible | +1027.3% | +13.12% | **0.66** | 74.3% |
| Benchmark × beta 0.75 | +585.8% | +10.30% | 0.66 | 63.3% |

Alpha +4.51%/yr at t = 1.11 — positive and insignificant, the identical shape the
first run produced. 1,507 fills cost ₹22,225, **55.6% of capital** over 16.6 years.

Variants at N=20 pass. The pre-registration fixed the rule before any of it ran:
**if the primary fails and a variant passes, the hypothesis has failed** and the
variant is a new hypothesis needing its own sample. The passing cells do not
survive scrutiny anyway — Sharpe rises with N because the ~600-name benchmark sits
at 0.66 and the strategy only approaches it as N grows, which is diversification
recovering rather than selection skill; no alpha t reaches 2 except one, the best
of eight cells; and the ordering is not even monotonic.

### What this settles

**The edge is real and the money cannot reach it.** Sixteen of the spread's
twenty-eight points are the short leg, and CNC cannot short. Of the twelve that
remain, costs take 3.3%/yr, and what survives does not clear a Sharpe bar against
simply holding the eligible universe equal-weight — which returned 13.12%/yr at
Sharpe 0.66 with no ranking, no turnover and no costs.

That last comparison is the useful one, and it is the best-performing strategy
anywhere in this repository.

**The live gate is untouched, and no cross-sectional path to it exists.** Nothing
in `kabali/xsection/` can place an order; it is research code with no execution
surface at all.

### Data defects these studies caught

Found by looking at the data rather than by reading code, and all would have
manufactured signal rather than crashing:

| Defect | What it would have done |
|---|---|
| HATSUN prints ₹468 between two ₹900 sessions | `research.corporate_actions` reads it as a 2:1 split — its open-confirms check passes because the corrupt bar's *open* is corrupt too — and back-adjusts a year of history |
| MOTHERSON and LLOYDSENGG interleave whole sessions at two price scales, 62 and 31 times | A name that periodically jumps 50% sits at the top of a momentum sort on the strength of a data defect |
| A bhavcopy parse failure cached as a *holiday* | Holidays are never retried, so 2006–2010 would have vanished from the panel silently. Only genuine absence is cached now. |
| A quarantine threshold of −55%, set below the −50% a 1:2 split produces | It protected sub-rupee names already excluded by the ₹10 price floor, while letting every unadjusted 1:2 split through — 15 of them on eligible names, including MASTEK 834→380 and DABUR 73.65→36.02 |

The first two are fixed by a persistence test: a split is permanent, a bad print
reverts. The last two only appeared at full scale — both were correct on the
64-day slice they were first validated against, which is the pattern worth
remembering.

## Known limitations

- **Costs are modelled, not observed.** Verify `research/costs.py` rates against
  a current contract note before trusting any P&L here.
- **No market impact, partial fills, or exchange rejections** in paper. Circuit
  limits and scrip-level bans are not simulated at all.
- **MIS leverage is assumed, not queried.** Dhan sets it per scrip and changes it.
- **The intraday universe is survivorship-biased.** It is drawn from names listed
  today. The bhavcopy panel is not — it retains 470 companies that delisted
  between 2006 and 2026 — but nothing in the intraday path uses it.
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
- **The cross-sectional test spans 16.6 years, not the panel's 20.7.** A
  rebalance needs 400 eligible names and NSE does not supply them before 2010, so
  2006–2010 contributes history to the formation windows but no rebalances.
- **Even 16.6 years is a small number of independent macro episodes.** The window
  contains one severe drawdown (2020) and no prolonged bear market. Momentum
  fails hardest at sharp reversals off a bottom, and this sample is thin in
  exactly that regime.
- **No dividends in any panel.** Total returns are understated by roughly the
  yield — close to neutral across a cross-sectional sort, but not zero.
- **The quarantine rule removes real events along with bad data.** A one-day move
  beyond −42% or +90% that no confirmed corporate action explains has the name's
  prior history blanked. Genuine crashes of that size are rare but real, and they
  are discarded. The choice is deliberate: it removes names from the sample
  rather than inventing returns for them.
- **Corporate-action coverage is NSE's.** 95.3% of priceable actions are
  price-confirmed and 60 are contradicted and refused, but an action the feed
  never carried can only be caught by the quarantine rule, not adjusted.
