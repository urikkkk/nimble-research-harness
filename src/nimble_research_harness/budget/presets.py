"""Time budget preset utilities."""

from ..models.enums import TimeBudget
from ..models.session import TimeBudgetPolicy


def get_policy(budget: TimeBudget) -> TimeBudgetPolicy:
    return TimeBudgetPolicy.from_budget(budget)


def suggest_budget(query: str) -> TimeBudget:
    """Heuristic budget suggestion based on query complexity signals."""
    q = query.lower()
    deep_signals = ["comprehensive", "deep dive", "thorough", "exhaustive", "detailed report"]
    if any(s in q for s in deep_signals):
        return TimeBudget.DEEP_30M

    medium_signals = ["compare", "analysis", "research", "investigate", "market", "competitive"]
    if any(s in q for s in medium_signals):
        return TimeBudget.STANDARD_10M

    quick_signals = ["what is", "who is", "quick", "fast", "brief", "lookup"]
    if any(s in q for s in quick_signals):
        return TimeBudget.SHORT_2M

    return TimeBudget.MEDIUM_5M
