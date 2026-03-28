"""
Financial Scoring Utilities

Shared financial analysis logic across multiple agents to eliminate duplication.
"""

from typing import Any

from ashare_analyzer.models import SignalType


def analyze_financial_health(data: dict[str, Any]) -> int:
    """
    Analyze financial health metrics (0-10 scale).

    Scoring:
    - Debt-to-Equity < 0.3: +3, < 0.5: +2, < 1.0: +1
    - Current Ratio > 2.0: +3, > 1.5: +2, > 1.0: +1
    - Interest Coverage > 5: +2, > 3: +1
    - Free Cash Flow positive: +2
    """
    score = _score_balance_sheet_strength(data)

    fcf = data.get("free_cash_flow", 0)
    if fcf > 0:
        score += 2

    return min(score, 10)


def analyze_safety(data: dict[str, Any]) -> int:
    """
    Analyze balance sheet safety (0-10 scale).

    Scoring:
    - Debt-to-Equity < 0.3: +3, < 0.5: +2, < 1.0: +1
    - Current Ratio > 2.0: +3, > 1.5: +2, > 1.0: +1
    - Interest Coverage > 5: +2, > 3: +1
    - Net cash position: +2
    """
    score = _score_balance_sheet_strength(data)

    if data.get("net_cash_position", False):
        score += 2

    return min(score, 10)


def _score_balance_sheet_strength(data: dict[str, Any]) -> int:
    """
    Score balance sheet strength metrics (shared logic).

    Returns:
        Base score (0-8) from debt-to-equity, current ratio, and interest coverage
    """
    score = 0

    dte = data.get("debt_to_equity", 0)
    if dte < 0.3:
        score += 3
    elif dte < 0.5:
        score += 2
    elif dte < 1.0:
        score += 1

    current_ratio = data.get("current_ratio", 0)
    if current_ratio > 2.0:
        score += 3
    elif current_ratio > 1.5:
        score += 2
    elif current_ratio > 1.0:
        score += 1

    interest_coverage = data.get("interest_coverage", 0)
    if interest_coverage > 5:
        score += 2
    elif interest_coverage > 3:
        score += 1

    return score


def analyze_profitability(data: dict[str, Any]) -> int:
    """
    Analyze profitability metrics (0-10 scale).

    Scoring:
    - ROE > 15%: +3, > 10%: +2, > 5%: +1
    - Net Margin > 20%: +3, > 15%: +2, > 10%: +1
    - Gross Margin > 40%: +2, > 30%: +1
    - ROA > 8%: +2, > 5%: +1
    """
    score = 0

    roe = data.get("roe", 0)
    if roe > 15:
        score += 3
    elif roe > 10:
        score += 2
    elif roe > 5:
        score += 1

    net_margin = data.get("net_margin", 0)
    if net_margin > 20:
        score += 3
    elif net_margin > 15:
        score += 2
    elif net_margin > 10:
        score += 1

    gross_margin = data.get("gross_margin", 0)
    if gross_margin > 40:
        score += 2
    elif gross_margin > 30:
        score += 1

    roa = data.get("roa", 0)
    if roa > 8:
        score += 2
    elif roa > 5:
        score += 1

    return min(score, 10)


class FinancialScorer:
    """
    Backward-compatible wrapper for module functions.

    Deprecated: Use module-level functions directly instead.
    """

    @staticmethod
    def analyze_financial_health(data: dict[str, Any]) -> int:
        return analyze_financial_health(data)

    @staticmethod
    def analyze_safety(data: dict[str, Any]) -> int:
        return analyze_safety(data)

    @staticmethod
    def analyze_profitability(data: dict[str, Any]) -> int:
        return analyze_profitability(data)


def score_to_signal(
    score: float,
    buy_threshold: float = 70.0,
    sell_threshold: float = 30.0,
    neutral_point: float = 50.0,
) -> tuple[SignalType, int]:
    """
    Convert numeric score to signal with confidence.

    Args:
        score: Numeric score (typically 0-100)
        buy_threshold: Score >= this value triggers BUY (default 70)
        sell_threshold: Score <= this value triggers SELL (default 30)
        neutral_point: Center point for HOLD confidence calculation (default 50)

    Returns:
        Tuple of (SignalType, confidence)

    Example:
        signal, confidence = score_to_signal(75)  # -> (SignalType.BUY, 67)
        signal, confidence = score_to_signal(25)  # -> (SignalType.SELL, 67)
        signal, confidence = score_to_signal(55)  # -> (SignalType.HOLD, 55)
    """
    if score >= buy_threshold:
        confidence = min(100, int(50 + (score - buy_threshold) * 1.5))
        return (SignalType.BUY, confidence)
    elif score <= sell_threshold:
        confidence = min(100, int(50 + (sell_threshold - score) * 1.5))
        return (SignalType.SELL, confidence)
    else:
        distance = abs(score - neutral_point)
        confidence = max(30, int(50 + distance))
        return (SignalType.HOLD, confidence)
