"""
AI Analyzer Module - Multi-Agent Architecture

This module implements the AI analysis layer using a multi-agent system.
It coordinates specialized agents to analyze stocks and generate trading decisions.

Responsibilities:
1. Coordinate multiple specialized agents for parallel analysis
2. Provide analysis reports as context to DecisionMaker
3. Generate decision dashboard reports
4. Support multiple LLM providers

Architecture:
    Analysis Agents (parallel) -> RiskManagerAgent -> DecisionMakerAgent (LLM-driven decision)
"""

import logging
from typing import Any

from ashare_analyzer.ai.interface import IAIAnalyzer
from ashare_analyzer.analysis.context import build_portfolio_context
from ashare_analyzer.config import get_config
from ashare_analyzer.constants import normalize_signal
from ashare_analyzer.data.stock_name_resolver import StockNameResolver
from ashare_analyzer.dependencies import get_portfolio_service
from ashare_analyzer.exceptions import AnalysisError
from ashare_analyzer.models import AnalysisResult
from ashare_analyzer.utils import get_display

logger = logging.getLogger(__name__)


class AIAnalyzer(IAIAnalyzer):
    """
    AI Analyzer based on multi-agent architecture.

    This class coordinates multiple specialized agents to analyze stocks
    and generates trading decisions through the DecisionMakerAgent.

    Execution Flow:
        1. RiskManagerAgent calculates position limits (runs first)
        2. Analysis agents run in parallel (Technical, Fundamental, etc.)
        3. DecisionMakerAgent makes final LLM-driven decision respecting risk limits

    Example:
        analyzer = AIAnalyzer()
        result = analyzer.analyze(context)
    """

    def __init__(self):
        """Initialize AI analyzer with multi-agent coordinator."""
        self._init_agent_coordinator()
        self._portfolio_service = get_portfolio_service()
        logger.debug("AI分析器初始化成功 (LLM决策模式)")

    def _init_agent_coordinator(self) -> None:
        """Initialize multi-agent coordinator with decision layer."""
        from ashare_analyzer.ai.agents import (
            AgentCoordinator,
            ChipAgent,
            DecisionMakerAgent,
            FundamentalAgent,
            NewsSentimentAgent,
            RiskManagerAgent,
            StyleAgent,
            TechnicalAgent,
            ValuationAgent,
        )

        self._agent_coordinator = AgentCoordinator()

        # Register analysis agents (provide analysis reports, not votes)
        # Technical & A-share specific
        self._agent_coordinator.register_agent(TechnicalAgent())
        self._agent_coordinator.register_agent(ChipAgent())

        # Fundamental analysis
        self._agent_coordinator.register_agent(FundamentalAgent())
        self._agent_coordinator.register_agent(ValuationAgent())

        # News & sentiment
        self._agent_coordinator.register_agent(NewsSentimentAgent())

        # Investment style (merged Value/Growth/Momentum into single agent)
        self._agent_coordinator.register_agent(StyleAgent())

        # Risk manager (calculates position limits, not a trading signal)
        self._risk_manager_agent = RiskManagerAgent()

        # Decision maker (LLM-driven final decision)
        self._decision_maker = DecisionMakerAgent()

        agent_count = len(self._agent_coordinator.agents)
        logger.debug(f"Agent协调器初始化完成，已注册{agent_count}个分析Agent + RiskManager + DecisionMaker")

    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self._agent_coordinator is not None

    async def analyze(self, context: dict[str, Any]) -> AnalysisResult:
        """
        Analyze a single stock using multi-agent architecture (async).

        Analysis flow:
        1. RiskManagerAgent calculates position limits (runs first)
        2. Execute parallel multi-agent analysis (Technical/Fundamental/Chip)
        3. DecisionMakerAgent makes final LLM-driven decision respecting risk limits
        4. Build decision dashboard
        5. Return structured result

        Args:
            context: Analysis context with stock data

        Returns:
            AnalysisResult with trading decision
        """
        code = context.get("code", "Unknown")
        name = context.get("stock_name", "")

        # Get stock name from context
        if not name or name.startswith("股票") or name.startswith("Stock"):
            if "realtime" in context and context["realtime"].get("name"):
                name = context["realtime"]["name"]
            else:
                name = StockNameResolver.from_context(code, context)

        # Final fallback
        if not name or name.startswith("股票") or name.startswith("Stock"):
            name = f"股票{code}"

        logger.debug(f"多Agent分析 {name}({code}) 开始")

        # Ensure coordinator is initialized
        if self._agent_coordinator is None:
            logger.debug("多Agent协调器未初始化，正在初始化...")
            self._init_agent_coordinator()

        if self._agent_coordinator is None:
            raise AnalysisError("Agent协调器初始化失败")

        try:
            display = get_display()

            # Step 1: RiskManagerAgent calculates position limits (runs first)
            display.start_agent("RiskManagerAgent")
            risk_manager_signal = await self._risk_manager_agent.analyze(context)
            display.complete_agent("RiskManagerAgent", "neutral", risk_manager_signal.confidence)
            max_pos = risk_manager_signal.metadata.get("max_position_size", 0.25) * 100
            logger.debug(f"[{code}] 风险管理完成: 仓位上限={max_pos:.0f}%")

            # Step 2: Execute multi-agent analysis (parallel)
            agent_results = await self._agent_coordinator.analyze(context)

            # Step 3: Extract consensus data
            consensus = agent_results["consensus"]
            agent_signals = agent_results["agent_signals"]
            consensus_level = agent_results["consensus_level"]

            logger.debug(
                f"[{code}] 分析Agent完成: {len(consensus.participating_agents)}个Agent参与, 共识度{consensus_level:.2f}"
            )

            # Step 4: Build portfolio context with current price
            # Use the unified price from context (set by analysis/analyzer.py via get_current_price)
            # which already prioritizes daily close over potentially stale realtime quotes
            current_price = float(context.get("current_price", 0))
            if current_price <= 0:
                if "today" in context and context["today"]:
                    current_price = float(context["today"].get("close", 0))
                elif "realtime" in context and context["realtime"]:
                    current_price = float(context["realtime"].get("price", 0))

            available_cash = get_config().portfolio.available_cash if get_config().portfolio else 0.0

            portfolio_context = await build_portfolio_context(
                self._portfolio_service,
                code,
                current_price=current_price,
                available_cash=available_cash,
            )

            # Step 5: DecisionMaker makes final LLM-driven decision
            # Convert agent signals to analysis reports format
            analysis_reports = self._format_analysis_reports(agent_signals)

            decision_context = {
                "code": code,
                "stock_name": name,
                "analysis_reports": analysis_reports,
                "portfolio": portfolio_context,
                "market_data": context.get("today", {}),
                "risk_limits": {
                    "max_position": risk_manager_signal.metadata.get("max_position_size", 0.25),
                    "volatility_tier": risk_manager_signal.metadata.get("volatility_tier", "medium"),
                },
            }

            display.start_agent("DecisionMakerAgent")
            final_signal = await self._decision_maker.analyze(decision_context)
            action = final_signal.metadata.get("decision", "hold").upper()
            display.complete_agent("DecisionMakerAgent", normalize_signal(action), final_signal.confidence)

            logger.debug(
                f"[{code}] LLM决策完成: {final_signal.metadata.get('decision', 'unknown')} "
                f"(置信度{final_signal.confidence}%, 仓位{final_signal.metadata.get('position_ratio', 0) * 100:.0f}%)"
            )

            # Add portfolio context to context dict for result building
            context["portfolio"] = portfolio_context

            # Step 6: Build AnalysisResult from decision
            result = self._build_analysis_result_from_decision(
                code=code,
                name=name,
                final_signal=final_signal,
                agent_signals=agent_signals,
                consensus_level=consensus_level,
                context=context,
            )

            # Step 7: Populate debug fields (data_sources and raw_response)
            result.data_sources = self._collect_data_sources(agent_signals)
            result.raw_response = {
                "final_decision": {
                    "decision": final_signal.metadata.get("decision", "hold"),
                    "signal": final_signal.signal.to_string(),
                    "confidence": final_signal.confidence,
                    "reasoning": final_signal.reasoning,
                    "key_considerations": final_signal.metadata.get("key_considerations", []),
                    "risks_identified": final_signal.metadata.get("risks_identified", []),
                },
                "risk_management": {
                    "max_position_size": risk_manager_signal.metadata.get("max_position_size"),
                    "risk_score": risk_manager_signal.metadata.get("risk_score"),
                    "volatility_tier": risk_manager_signal.metadata.get("volatility_tier"),
                },
                "agent_signals": agent_signals,
                "analysis_reports": analysis_reports,
                "consensus": {
                    "consensus_level": consensus_level,
                    "participating_agents": consensus.participating_agents,
                    "risk_flags": consensus.risk_flags,
                },
            }

            return result

        except Exception as e:
            logger.error(f"多Agent分析 {name}({code}) 失败: {e}")
            # Return error result
            return AnalysisResult(
                code=code,
                name=name,
                sentiment_score=50,
                trend_prediction="震荡",
                operation_advice="持有",
                decision_type="hold",
                confidence_level="低",
                analysis_summary=f"多Agent分析失败: {str(e)[:100]}",
                risk_warning="分析失败，请稍后重试",
                success=False,
                error_message=str(e),
            )

    def _build_analysis_result_from_decision(
        self,
        code: str,
        name: str,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        context: dict[str, Any],
    ) -> AnalysisResult:
        """Build AnalysisResult from DecisionMaker output.

        Args:
            code: Stock code
            name: Stock name
            final_signal: Final signal from DecisionMakerAgent
            agent_signals: Signals from all analysis agents
            consensus_level: Consensus level among agents
            context: Analysis context

        Returns:
            AnalysisResult object
        """

        # Extract decision metadata (new format from DecisionMakerAgent)
        metadata = final_signal.metadata
        decision = metadata.get("decision", "hold").upper()
        position_ratio = metadata.get("position_ratio", 0.0)
        trade_quantity = metadata.get("trade_quantity", 0)
        position_action = metadata.get("position_action", "no_action")
        key_considerations = metadata.get("key_considerations", [])
        risks_identified = metadata.get("risks_identified", [])

        # Map decision to Chinese operation advice
        action_map = {
            "BUY": "买入",
            "HOLD": "持有",
            "SELL": "卖出",
        }
        operation_advice = action_map.get(decision, "观望")

        # Map decision to decision_type
        decision_type = decision.lower()

        # Determine trend prediction based on decision
        if decision == "BUY":
            trend_prediction = "看多" if final_signal.confidence >= 70 else "谨慎看多"
        elif decision == "SELL":
            trend_prediction = "看空" if final_signal.confidence >= 70 else "谨慎看空"
        else:
            trend_prediction = "震荡"

        # Build decision-focused dashboard
        dashboard = self._build_decision_dashboard(final_signal, agent_signals, consensus_level, context)

        # Build analysis summary with key considerations
        analysis_summary = self._build_decision_summary(
            final_signal, agent_signals, consensus_level, key_considerations
        )

        # Build risk warning from identified risks
        risk_warning = "风险提示: " + "; ".join(risks_identified[:3]) if risks_identified else "未识别到重大风险"

        # Detect consensus deviation: when decision direction differs from agent consensus
        consensus_direction = self._get_consensus_direction(agent_signals)
        if consensus_direction and decision.lower() != consensus_direction:
            consensus_label = {"buy": "看多", "sell": "看空", "hold": "持有/观望"}.get(
                consensus_direction, consensus_direction
            )
            risk_warning = (
                f"⚠️ 决策偏离共识(共识方向: {consensus_label} {consensus_level:.0%}, 决策方向: {decision}) | "
                + risk_warning
            )

        # Extract portfolio information from context
        portfolio_info = context.get("portfolio", {})
        has_position = portfolio_info.get("has_position", False)
        position_quantity = portfolio_info.get("position_quantity", 0)
        position_cost_price = portfolio_info.get("position_cost_price", 0.0)

        return AnalysisResult(
            code=code,
            name=name,
            sentiment_score=final_signal.confidence,
            trend_prediction=trend_prediction,
            operation_advice=operation_advice,
            decision_type=decision_type,
            confidence_level=self._score_to_confidence(final_signal.confidence),
            final_action=decision,
            position_ratio=position_ratio,
            action_quantity=trade_quantity,
            position_action=position_action,
            decision_reasoning=final_signal.reasoning,
            has_position=has_position,
            position_quantity=position_quantity,
            position_cost_price=position_cost_price,
            dashboard=dashboard,
            analysis_summary=analysis_summary,
            risk_warning=risk_warning,
            market_snapshot=self._build_market_snapshot(context),
            success=True,
        )

    def _format_analysis_reports(self, agent_signals: dict[str, Any]) -> list[dict[str, Any]]:
        """Format agent signals as analysis reports for DecisionMaker.

        This converts the agent signals from the voting format to
        a format suitable for LLM context input.

        Args:
            agent_signals: Dict of agent name -> signal data

        Returns:
            List of analysis report dicts with analyst, conclusion, confidence, reasoning
        """
        reports = []
        for agent_name, signal_data in agent_signals.items():
            # Extract key metrics from metadata if available
            metadata = signal_data.get("metadata", {})
            key_metrics = {}

            # Extract relevant metrics based on agent type
            if "TechnicalAgent" in agent_name:
                key_metrics = {
                    "trend": metadata.get("trend_assessment"),
                    "trend_strength": metadata.get("trend_strength"),
                    "rsi": metadata.get("rsi"),
                    "macd": metadata.get("macd_signal"),
                }
            elif "ChipAgent" in agent_name:
                key_metrics = {
                    "concentration": metadata.get("concentration"),
                    "profit_ratio": metadata.get("profit_ratio"),
                    "phase": metadata.get("phase"),
                }
            elif "ValuationAgent" in agent_name:
                key_metrics = {
                    "margin_of_safety": metadata.get("margin_of_safety"),
                    "fair_value": metadata.get("fair_value"),
                    "pe_ratio": metadata.get("pe_ratio"),
                }

            # Clean up None values
            key_metrics = {k: v for k, v in key_metrics.items() if v is not None}

            reports.append(
                {
                    "analyst": agent_name,
                    "conclusion": signal_data.get("signal", "hold"),
                    "confidence": signal_data.get("confidence", 0),
                    "reasoning": signal_data.get("reasoning", ""),
                    "key_metrics": key_metrics,
                }
            )
        return reports

    def _build_decision_dashboard(
        self,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build decision-focused dashboard for visualization."""
        metadata = final_signal.metadata
        decision = metadata.get("decision", "hold").upper()
        position_ratio = metadata.get("position_ratio", 0.0)
        key_considerations = metadata.get("key_considerations", [])
        risks_identified = metadata.get("risks_identified", [])

        return {
            "final_decision": {
                "decision": decision,
                "confidence": final_signal.confidence,
                "position_ratio": f"{position_ratio * 100:.0f}%",
                "reasoning": final_signal.reasoning,
            },
            "key_considerations": key_considerations,
            "risks_identified": risks_identified,
            "agent_reports": {
                "signals": {k: v["signal"] for k, v in agent_signals.items()},
                "confidences": {k: v["confidence"] for k, v in agent_signals.items()},
                "reasonings": {k: v["reasoning"] for k, v in agent_signals.items()},
                "consensus_level": f"{consensus_level:.0%}",
            },
            "market_context": {
                "current_position": metadata.get("current_position", "none"),
                "price": context.get("today", {}).get("close"),
            },
        }

    def _build_decision_summary(
        self,
        final_signal: Any,
        agent_signals: dict[str, Any],
        consensus_level: float,
        key_considerations: list[str],
    ) -> str:
        """Build decision-focused summary text."""
        metadata = final_signal.metadata
        decision = metadata.get("decision", "hold").upper()

        parts = [
            f"LLM决策: {decision}",
            f"置信度: {final_signal.confidence}%",
            f"共识度: {consensus_level:.0%}",
        ]

        # Add key considerations
        if key_considerations:
            parts.append(f"关键考量: {', '.join(key_considerations[:3])}")

        # Add agent summary
        buy_count = sum(1 for s in agent_signals.values() if s.get("signal", "").lower() == "buy")
        sell_count = sum(1 for s in agent_signals.values() if s.get("signal", "").lower() == "sell")
        parts.append(f"分析师: {buy_count}买/{sell_count}卖")

        return " | ".join(parts)

    def _score_to_confidence(self, score: int) -> str:
        """Convert numeric score to confidence level text."""
        if score >= 80:
            return "高"
        elif score >= 60:
            return "中"
        return "低"

    @staticmethod
    def _get_consensus_direction(agent_signals: dict[str, Any]) -> str | None:
        """Get the consensus direction (buy/sell/hold) from agent signals.

        Returns None when there is no clear consensus (tie for top direction)
        or when fewer than 2 agents participated (consensus is meaningless
        with a single agent — any override would always trigger a deviation warning).
        """
        if len(agent_signals) < 2:
            return None
        counts: dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        for signal_data in agent_signals.values():
            sig = signal_data.get("signal", "hold").lower()
            if sig in counts:
                counts[sig] += 1
        total = sum(counts.values())
        if total == 0:
            return None
        max_count = max(counts.values())
        top_directions = [d for d, c in counts.items() if c == max_count]
        if len(top_directions) > 1:
            return None
        if max_count <= total / 2:
            return None
        return top_directions[0]

    def _build_market_snapshot(self, context: dict[str, Any]) -> dict[str, Any]:
        """Build market snapshot from analysis context."""
        today = context.get("today", {}) or {}
        realtime = context.get("realtime", {}) or {}
        yesterday = context.get("yesterday", {}) or {}

        prev_close = yesterday.get("close")
        close = today.get("close")
        high = today.get("high")
        low = today.get("low")

        # Calculate amplitude
        amplitude = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None

        snapshot = {
            "date": context.get("date", "未知"),
            "close": self._format_price(close),
            "open": self._format_price(today.get("open")),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get("pct_chg")),
            "amplitude": self._format_percent(amplitude),
            "volume": today.get("volume"),
            "amount": today.get("amount"),
        }

        if realtime:
            snapshot.update(
                {
                    "price": self._format_price(realtime.get("price")),
                    "change_amount": realtime.get("change_amount"),
                    "volume_ratio": realtime.get("volume_ratio"),
                    "turnover_rate": self._format_percent(realtime.get("turnover_rate")),
                    "source": realtime.get("source"),
                }
            )

        return snapshot

    def _collect_data_sources(self, agent_signals: dict[str, Any]) -> str:
        """Collect data source information from agent signals."""
        agent_names = list(agent_signals.keys())
        return ", ".join(agent_names) if agent_names else "multi-agent"

    @staticmethod
    def _format_price(value: float | None) -> str:
        """Format price value for display."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    @staticmethod
    def _format_percent(value: float | None) -> str:
        """Format percentage value for display."""
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    async def batch_analyze(
        self,
        contexts: list[dict[str, Any]],
        delay_between: float = 2.0,
    ) -> list[AnalysisResult]:
        """
        Batch analyze multiple stocks (async).

        Args:
            contexts: Context data list
            delay_between: Delay between analyses in seconds

        Returns:
            List of AnalysisResult
        """
        import asyncio

        results = []

        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"等待 {delay_between} 秒后继续...")
                await asyncio.sleep(delay_between)

            result = await self.analyze(context)
            results.append(result)

        return results
