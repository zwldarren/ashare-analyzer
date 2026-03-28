"""
ReportBuilder - Transform AnalysisResult objects into Report objects.

This module provides the ReportBuilder class that converts raw analysis results
into structured Report objects for unified output formatting.
"""

from datetime import datetime
from typing import Any

from ashare_analyzer.constants import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, normalize_signal
from ashare_analyzer.models import AnalysisResult

from .models import AgentOpinion, MarketSnapshot, Report, ReportSummary, StockReport


class ReportBuilder:
    """
    Build Report objects from AnalysisResult objects.

    Transforms raw analysis results into structured reports suitable for
    both console display and markdown notification.
    """

    def build(self, results: list[AnalysisResult], report_date: str | None = None) -> Report:
        """
        Build a Report from a list of AnalysisResults.

        Args:
            results: List of AnalysisResult objects
            report_date: Optional report date string (defaults to today)

        Returns:
            Report object with sorted stocks and summary
        """
        if report_date is None:
            report_date = datetime.now().strftime("%Y-%m-%d")

        if not results:
            return Report(
                report_date=report_date,
                stocks=[],
                summary=ReportSummary(total_count=0, buy_count=0, hold_count=0, sell_count=0),
            )

        # Sort results by sentiment_score descending (highest score first)
        sorted_results = sorted(results, key=lambda r: r.sentiment_score, reverse=True)

        # Build stock reports
        stock_reports = [self._build_stock_report(result) for result in sorted_results]

        # Count actions
        buy_count = sum(1 for s in stock_reports if s.action == SIGNAL_BUY)
        hold_count = sum(1 for s in stock_reports if s.action == SIGNAL_HOLD)
        sell_count = sum(1 for s in stock_reports if s.action == SIGNAL_SELL)

        summary = ReportSummary(
            total_count=len(stock_reports),
            buy_count=buy_count,
            hold_count=hold_count,
            sell_count=sell_count,
        )

        return Report(report_date=report_date, stocks=stock_reports, summary=summary)

    def _build_stock_report(self, result: AnalysisResult) -> StockReport:
        """
        Build a StockReport from an AnalysisResult.

        Args:
            result: AnalysisResult object

        Returns:
            StockReport object
        """
        # Extract name - use default if empty or placeholder
        name = result.name.strip()
        if not name or name.startswith("股票") or name.startswith("Stock"):
            name = f"股票{result.code}"

        # Extract dashboard data
        dashboard = result.dashboard or {}
        agent_consensus = dashboard.get("agent_reports", {})
        key_factors = dashboard.get("key_considerations", [])

        # Ensure key_factors is a list
        if not isinstance(key_factors, list):
            key_factors = []

        # Extract agent opinions
        agent_opinions = self._extract_agent_opinions(agent_consensus)

        # Extract consensus_level
        consensus_level = agent_consensus.get("consensus_level", "N/A")

        # Use sentiment_score as confidence
        confidence = result.sentiment_score

        # Build market snapshot
        market_snapshot = MarketSnapshot.from_dict(result.market_snapshot)

        # Determine action from final_action or decision_type
        action = normalize_signal(result.final_action or result.decision_type)

        # Extract action quantity and position action from result
        action_quantity = result.action_quantity
        position_action = result.position_action

        return StockReport(
            code=result.code,
            name=name,
            action=action,
            confidence=confidence,
            position_ratio=result.position_ratio,
            action_quantity=action_quantity,
            position_action=position_action,
            trend_prediction=result.trend_prediction,
            decision_reasoning=result.decision_reasoning,
            agent_opinions=agent_opinions,
            consensus_level=consensus_level,
            key_factors=key_factors[:3] if key_factors else [],
            risk_warning=result.risk_warning if result.risk_warning else None,
            market_snapshot=market_snapshot,
            has_position=result.has_position,
            position_quantity=result.position_quantity,
            position_cost_price=result.position_cost_price,
            success=result.success,
            error_message=result.error_message,
            data_sources=result.data_sources,
        )

    def _extract_agent_opinions(self, agent_consensus: dict[str, Any]) -> list[AgentOpinion]:
        """
        Extract agent opinions from consensus data.

        Args:
            agent_consensus: Dict with signals, confidences, reasonings

        Returns:
            List of AgentOpinion objects
        """
        opinions: list[AgentOpinion] = []

        if not agent_consensus:
            return opinions

        # Get the three dictionaries
        signals = agent_consensus.get("signals", {})
        confidences = agent_consensus.get("confidences", {})
        reasonings = agent_consensus.get("reasonings", {})

        # All three should have the same keys (agent names)
        if not signals:
            return opinions

        for agent_name in signals:
            signal = normalize_signal(signals.get(agent_name, SIGNAL_HOLD))
            confidence = confidences.get(agent_name, 0)
            reasoning = reasonings.get(agent_name, "")

            # Ensure confidence is an integer
            if not isinstance(confidence, int):
                try:
                    confidence = int(confidence)
                except (ValueError, TypeError):
                    confidence = 0

            opinions.append(
                AgentOpinion(
                    name=agent_name,
                    signal=signal,
                    confidence=confidence,
                    reasoning=str(reasoning),
                )
            )

        return opinions
