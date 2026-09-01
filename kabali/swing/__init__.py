"""The swing venue: multi-week positions on daily bars, CNC delivery.

Separate from `kabali.engine` on purpose. The intraday engine's core assumption
-- that the book is flat by 15:10 and every position is MIS -- is false here,
and a shared engine that had to branch on it would make both harder to reason
about. What IS shared is everything that does not depend on holding period: the
cost model, the sizing discipline, the closed-trade accounting, and the live
gate.
"""
