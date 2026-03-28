"""
Decision Maker Agent Module

The final decision maker that uses LLM for autonomous trading decisions.

Unlike the previous approach where agents voted and rules determined outcomes,
this agent receives analysis reports as context and lets the LLM make
autonomous decisions based on its own judgment.

Key principles:
- Analysis reports are INFORMATION, not VOTES
- LLM decides signal weights dynamically based on market context
- No hardcoded thresholds (no "weighted_score >= 30 = BUY")
- Respects risk limits as hard guards, not decision inputs
"""

import logging
from typing import Any

from ashare_analyzer.ai.tools import DECISION_TOOL
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent

logger = logging.getLogger(__name__)

# System prompt for LLM-driven decision making - NO hardcoded rules
DECISION_MAKER_SYSTEM_PROMPT = """You are the final decision maker for A-share trading.

Your role:
- You have FULL AUTONOMY to make trading decisions
- You receive analysis reports from multiple analysts as INFORMATION
- You are NOT bound by any voting or consensus mechanism
- You decide which signals to trust and how much weight to give them

=== Your Decision Framework ===

You will receive:
1. Stock context (code, name, current price, market data)
2. Analysis reports from specialized analysts
3. Current portfolio state (positions, available capital)
4. Risk limits (maximum position size as a hard guard)

Your output MUST include:
- decision: "buy", "sell", or "hold"
- confidence: 0-100 (your conviction level)
- position_ratio: 0.0-1.0 (target portfolio allocation, NOT exceeding risk limits)
- reasoning: Explain YOUR thought process, not just summarize analysts
- key_considerations: What factors most influenced your decision
- risks_identified: What concerns or risks you see

=== Important Principles ===

1. ANALYSTS PROVIDE CONTEXT, NOT VOTES
   - A "HOLD" from one analyst doesn't cancel a "BUY" from another
   - You decide which analyst's view is most relevant for current market conditions
   - Consider each analyst's specialty and track record

2. MARKET CONTEXT MATTERS
   - In a strong bull market, higher valuations may be acceptable
   - In a bear market, even "cheap" stocks may decline further
   - Technical patterns that worked historically may not work now

3. POSITION SIZING IS YOUR CALL
   - The risk limit is a MAXIMUM, not a recommendation
   - You can choose a smaller position if you see higher risk
   - Consider existing positions and correlation

4. NO ABSOLUTE RULES
   - There's no "must buy" or "must sell" threshold
   - Every decision is a probability assessment
   - Explain your reasoning clearly

=== Output Format ===
Use the make_trading_decision function to return your decision.

Remember: You are the DECISION MAKER, not a vote aggregator.
Trust your judgment based on the comprehensive information provided."""


class DecisionMakerAgent(BaseAgent):
    """
    LLM-driven final decision maker for trading.

    This agent receives analysis reports as context and makes autonomous
    trading decisions without hardcoded rules or voting mechanisms.
    """

    def __init__(self):
        """Initialize the Decision Maker Agent."""
        super().__init__("DecisionMakerAgent")
        self._ensure_llm_client()

    def is_available(self) -> bool:
        """Always available - will return HOLD if LLM unavailable."""
        return True

    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Make final trading decision based on analysis reports (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - analysis_reports: List of analyst reports
                - portfolio: Current portfolio state
                - market_data: Current market data
                - risk_limits: Maximum position size and other limits

        Returns:
            AgentSignal with final trading decision
        """
        stock_code = context.get("code", "")
        stock_name = context.get("stock_name", "")

        self._logger.debug(f"[{stock_code}] DecisionMakerAgent starting decision process")

        try:
            analysis_reports = context.get("analysis_reports", [])
            portfolio = context.get("portfolio", {})
            risk_limits = context.get("risk_limits", {})
            market_data = context.get("market_data", {})

            if not analysis_reports:
                self._logger.warning(f"[{stock_code}] No analysis reports provided")
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="No analysis reports available for decision",
                    metadata={"error": "missing_analysis_reports"},
                )

            max_position = risk_limits.get("max_position", 0.25)

            # Use LLM for decision
            if self._llm_client and self._llm_client.is_available():
                decision = await self._make_llm_decision(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    analysis_reports=analysis_reports,
                    portfolio=portfolio,
                    market_data=market_data,
                    max_position=max_position,
                )
            else:
                # No fallback rules - return HOLD when LLM unavailable
                self._logger.warning(f"[{stock_code}] LLM unavailable, returning HOLD")
                return AgentSignal(
                    agent_name=self.name,
                    signal=SignalType.HOLD,
                    confidence=0,
                    reasoning="LLM 服务不可用，无法做出决策",
                    metadata={"error": "llm_unavailable"},
                )

            # Apply hard risk limit
            decision["position_ratio"] = min(decision.get("position_ratio", 0), max_position)

            self._logger.debug(
                f"[{stock_code}] Decision: {decision['decision']} "
                f"(confidence {decision['confidence']}%, position {decision['position_ratio'] * 100:.0f}%)"
            )

            signal = self._decision_to_signal(decision["decision"])

            return AgentSignal(
                agent_name=self.name,
                signal=signal,
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
                metadata={
                    "decision": decision["decision"],
                    "position_ratio": decision["position_ratio"],
                    "trade_quantity": decision.get("trade_quantity", 0),
                    "position_action": decision.get("position_action", "no_action"),
                    "key_considerations": decision.get("key_considerations", []),
                    "risks_identified": decision.get("risks_identified", []),
                    "max_position_limit": max_position,
                    "analyst_count": len(analysis_reports),
                },
            )

        except Exception as e:
            self._logger.error(f"[{stock_code}] DecisionMaker failed: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"Decision process failed: {str(e)}",
                metadata={"error": str(e)},
            )

    async def _make_llm_decision(
        self,
        stock_code: str,
        stock_name: str,
        analysis_reports: list[dict[str, Any]],
        portfolio: dict[str, Any],
        market_data: dict[str, Any],
        max_position: float,
    ) -> dict[str, Any]:
        """Use LLM to make autonomous decision (async)."""
        prompt = self._build_decision_prompt(
            stock_code=stock_code,
            stock_name=stock_name,
            analysis_reports=analysis_reports,
            portfolio=portfolio,
            market_data=market_data,
            max_position=max_position,
        )

        self._logger.debug(f"[{stock_code}] Calling LLM for decision...")

        if not self._llm_client:
            return self._make_hold_decision("LLM client not initialized")

        try:
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=DECISION_TOOL,
                generation_config={"temperature": 0.3, "max_output_tokens": 1024},
                system_prompt=DECISION_MAKER_SYSTEM_PROMPT,
                agent_name="DecisionMakerAgent",
            )

            if result and isinstance(result, dict) and "decision" in result:
                if isinstance(result["decision"], str):
                    result["decision"] = result["decision"].lower()
                if result["decision"] not in ["buy", "sell", "hold"]:
                    result["decision"] = "hold"

                self._logger.debug(f"[{stock_code}] LLM decision: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM returned invalid format")
                return self._make_hold_decision("LLM 返回格式无效")

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM call failed: {e}")
            return self._make_hold_decision(f"LLM 调用失败: {str(e)}")

    def _build_decision_prompt(
        self,
        stock_code: str,
        stock_name: str,
        analysis_reports: list[dict[str, Any]],
        portfolio: dict[str, Any],
        market_data: dict[str, Any],
        max_position: float,
    ) -> str:
        """Build the decision prompt with all context."""
        # Format analysis reports
        reports_text = self._format_analysis_reports(analysis_reports)

        # Format portfolio state
        portfolio_text = self._format_portfolio(portfolio)

        # Format market data
        market_text = self._format_market_data(market_data)

        return f"""请对以下股票做出交易决策。

=== 股票信息 ===
代码: {stock_code}
名称: {stock_name}
{market_text}

=== 分析师报告 ===
以下是各专业分析师的分析结果。请注意，这些是供参考的信息，不是投票。
你有权根据自己的判断决定每个报告的权重和可信度。

{reports_text}

=== 当前持仓状态 ===
{portfolio_text}

=== 风险限制 ===
最大仓位限制: {max_position * 100:.0f}% (这是硬性上限，不是建议)

请使用 make_trading_decision 函数输出你的决策。
记住：
- 你是最终决策者，不是投票汇总者
- 分析报告是参考信息，你需要自己判断其价值
- 解释你的思考过程，不要只是重复分析师的观点"""

    def _format_analysis_reports(self, reports: list[dict[str, Any]]) -> str:
        """Format analysis reports as readable text."""
        lines = []
        for report in reports:
            analyst = report.get("analyst", "Unknown")
            conclusion = report.get("conclusion", "hold").upper()
            confidence = report.get("confidence", 0)
            reasoning = report.get("reasoning", "No reasoning provided")
            key_metrics = report.get("key_metrics", {})

            lines.append(f"\n--- {analyst} ---")
            lines.append(f"结论: {conclusion} (置信度: {confidence}%)")
            lines.append(f"理由: {reasoning}")
            if key_metrics:
                metrics_str = ", ".join(f"{k}: {v}" for k, v in key_metrics.items() if v is not None)
                if metrics_str:
                    lines.append(f"关键指标: {metrics_str}")

        return "\n".join(lines)

    def _format_portfolio(self, portfolio: dict[str, Any]) -> str:
        """Format portfolio state as readable text."""
        if not portfolio:
            return "无持仓信息"

        has_position = portfolio.get("has_position", False)
        if not has_position:
            return "当前无持仓"

        quantity = portfolio.get("position_quantity", 0)
        cost_price = portfolio.get("position_cost_price", 0)
        current_price = portfolio.get("current_price", 0)
        profit_pct = portfolio.get("current_profit_loss_pct")
        total_value = portfolio.get("total_value", 0)
        position_ratio = portfolio.get("current_position_ratio", 0)

        profit_str = f"{profit_pct:.2f}%" if profit_pct is not None else "N/A"
        return f"""持有股数: {quantity} 股
成本价: ¥{cost_price:.2f}
当前价: ¥{current_price:.2f}
浮盈亏: {profit_str}
持仓市值: ¥{(current_price * quantity):.0f}
占总资产: {position_ratio * 100:.1f}%
总资产: ¥{total_value:.0f}"""

    def _format_market_data(self, market_data: dict[str, Any]) -> str:
        """Format market data as readable text."""
        if not market_data:
            return "无市场数据"

        close = market_data.get("close")
        pct_chg = market_data.get("pct_chg")
        volume = market_data.get("volume")
        amount = market_data.get("amount")

        lines = []
        if close:
            lines.append(f"收盘价: ¥{close:.2f}")
        if pct_chg is not None:
            lines.append(f"涨跌幅: {pct_chg:.2f}%")
        if volume:
            lines.append(f"成交量: {volume:,.0f}")
        if amount:
            lines.append(f"成交额: ¥{amount:,.0f}")

        return "\n".join(lines) if lines else "无市场数据"

    def _decision_to_signal(self, decision: str) -> SignalType:
        return SignalType.from_string(decision)

    def _make_hold_decision(self, reason: str) -> dict[str, Any]:
        """Create a hold decision with reasoning."""
        return {
            "decision": "hold",
            "confidence": 0,
            "position_ratio": 0,
            "trade_quantity": 0,
            "position_action": "no_action",
            "reasoning": reason,
            "key_considerations": [],
            "risks_identified": [reason],
        }
