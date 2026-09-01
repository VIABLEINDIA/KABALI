"""Company fundamentals: identity, size and sector for the traded universe.

WHERE THIS DATA COMES FROM, AND WHY THAT MATTERS
================================================
Everything else in KABALI is computed from bars the bot fetches itself. This is
not: it comes from a document/company connector that is available to an agent
session, not to a scheduled process. The bot at 09:00 cannot refresh it.

That constraint is recorded here rather than hidden behind a helper, because a
fundamentals file that silently ages is worse than none. A market cap from four
months ago will still join, still rank, and still look like a fact. So every
record carries `fetched_at`, the store reports its own staleness, and callers
that filter on it are expected to check.

WHAT IT IS FOR
==============
Price momentum says a name is moving. It says nothing about whether the company
behind it is a 500-crore shell or a 80,000-crore steelmaker, and those two names
respond to the same technical setup very differently -- one has a book deep
enough to absorb an order, the other does not. The universe's turnover screen is
a proxy for that; market cap is the thing itself.

THE SYMBOL MUST MATCH, NOT JUST THE SEARCH
==========================================
The connector resolves a free-text query to a company, and it will happily return
a DIFFERENT one rather than nothing. Querying "MANINDS Man Industries India"
returned Solar Industries -- a Rs 183,000 crore explosives maker filed under a
Rs 1,000 crore pipe manufacturer's symbol. Nothing about the response says it is
wrong; it is a well-formed record for a real company.

So a result is accepted only when its `nse_code` equals the symbol asked for.
Three of the first forty names could not be resolved that way and are absent
rather than wrong, which is the correct outcome: a missing record is visible in
the coverage report, and a wrong one ranks and filters exactly like a right one.

WHAT IT IS NOT
==============
Not an edge. Adding a size filter to a strategy that loses before costs makes it
lose slightly differently. Its honest use is as a NEW hypothesis tested on its
own terms -- quality-filtered momentum, pre-registered and run once -- not as
another knob on the existing one.
"""
