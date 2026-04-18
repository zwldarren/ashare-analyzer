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
DECISION_MAKER_SYSTEM_PROMPT = """你是A股交易的最终决策者。

你的角色：
- 你拥有完全的交易决策自主权
- 你收到来自多位分析师的分析报告作为信息参考
- 你不受任何投票或共识机制的约束
- 你决定信任哪些信号以及给予多少权重

=== 你的决策框架 ===

你将收到：
1. 股票上下文（代码、名称、当前价格、市场数据）
2. 专业分析师的分析报告
3. 当前持仓状态（持仓、可用资金）
4. 风险限制（最大仓位作为硬性约束）

你的输出必须包含：
- decision: "buy"、"sell" 或 "hold"
- confidence: 0-100（你的确信程度）
- position_ratio: 0.0-1.0（目标仓位配比，不得超过风险限制）
- reasoning: 解释你的思考过程，而非简单汇总分析师观点
- key_considerations: 最影响你决策的因素
- risks_identified: 你发现的风险或隐忧

=== 重要原则 ===

1. 分析师提供的是上下文，而非投票
   - 一个分析师的"HOLD"不会抵消另一个的"BUY"
   - 你决定在当前市场条件下哪个分析师的观点最相关
   - 考虑每位分析师的专业领域和过往表现

2. 市场环境至关重要
   - 在强势牛市中，较高估值可能是可接受的
   - 在熊市中，即使是"便宜"的股票也可能继续下跌
   - 历史上有效的技术模式现在未必仍然有效

3. 仓位大小由你决定
   - 风险限制是上限，而非建议
   - 如果你认为风险较高，可以选择更小的仓位
   - 考虑现有持仓和相关性

4. 没有绝对规则
   - 不存在"必须买入"或"必须卖出"的阈值
   - 每一个决策都是概率评估
   - 清晰地解释你的推理

=== 输出格式 ===
请使用 make_trading_decision 函数返回你的决策。

记住：你是决策者，而非投票汇总者。
基于所提供的综合信息，相信你自己的判断。"""


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

            # Position risk enforcement: when existing position exceeds max_position,
            # force a reduction independently of the LLM's trading signal
            current_position_ratio = portfolio.get("current_position_ratio", 0)
            if current_position_ratio > max_position and portfolio.get("has_position", False):
                # If LLM didn't already decide to sell/reduce, override to reduce position
                if decision["decision"] in ("hold", "buy"):
                    original_decision = decision["decision"]
                    llm_failed = decision.get("llm_failed", False)
                    self._logger.info(
                        f"[{stock_code}] Position risk override: "
                        f"{current_position_ratio:.0%} > {max_position:.0%} limit, "
                        f"forcing reduction (original: {original_decision})"
                    )
                    decision["decision"] = "sell"
                    position_quantity = portfolio.get("position_quantity", 0)
                    excess_ratio = current_position_ratio - max_position
                    shares_to_sell = max(1, int(position_quantity * (excess_ratio / current_position_ratio)))
                    decision["trade_quantity"] = min(shares_to_sell, position_quantity)
                    decision["position_action"] = "reduce_position"
                    decision["position_ratio"] = max_position

                    if llm_failed:
                        decision["confidence"] = 70
                        decision["reasoning"] = (
                            f"风控强制减仓: 当前持仓占比{current_position_ratio:.0%}"
                            f"超过{max_position:.0%}限制(LLM决策不可用，基于风控规则)。"
                        )
                    else:
                        decision["reasoning"] = (
                            f"仓位风险强制减仓: 当前持仓占比{current_position_ratio:.0%}"
                            f"超过{max_position:.0%}限制。原决策:{original_decision}。"
                            f"{decision.get('reasoning', '')}"
                        )

                    if "key_considerations" not in decision:
                        decision["key_considerations"] = []
                    decision["key_considerations"].insert(
                        0, f"持仓占比{current_position_ratio:.0%}超过{max_position:.0%}风险限制(强制)"
                    )
                    if llm_failed:
                        decision["key_considerations"].insert(1, "LLM决策不可用，基于风控规则减仓")
                    decision.pop("llm_failed", None)
                # Even if LLM already said sell, annotate that position risk was a factor
                elif decision["decision"] == "sell":
                    llm_failed = decision.get("llm_failed", False)
                    if llm_failed:
                        decision["confidence"] = 70
                    if "key_considerations" not in decision:
                        decision["key_considerations"] = []
                    if not any("风险限制" in c for c in decision.get("key_considerations", [])):
                        decision["key_considerations"].append(
                            f"持仓占比{current_position_ratio:.0%}超过{max_position:.0%}风险限制"
                        )
                    if llm_failed:
                        decision["key_considerations"].append("LLM决策不可用，基于风控规则减仓")
                    decision.pop("llm_failed", None)

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
                generation_config={"temperature": 0.3, "max_output_tokens": 16384},
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
            reasoning = report.get("reasoning", "No reasoning provided").replace("\n", "; ")
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
        available_cash = portfolio.get("available_cash", 0)

        if not has_position and available_cash <= 0:
            return "当前无持仓"

        quantity = portfolio.get("position_quantity", 0)
        cost_price = portfolio.get("position_cost_price", 0)
        current_price = portfolio.get("current_price", 0)
        profit_pct = portfolio.get("current_profit_loss_pct")
        total_value = portfolio.get("total_value", 0)
        position_ratio = portfolio.get("current_position_ratio", 0)

        profit_str = f"{profit_pct:.2f}%" if profit_pct is not None else "N/A"
        lines = []
        if has_position:
            lines.extend(
                [
                    f"持有股数: {quantity} 股",
                    f"成本价: ¥{cost_price:.2f}",
                    f"当前价: ¥{current_price:.2f}",
                    f"浮盈亏: {profit_str}",
                    f"持仓市值: ¥{(current_price * quantity):.0f}",
                    f"占总资产: {position_ratio * 100:.1f}%",
                ]
            )
        if available_cash > 0:
            lines.append(f"可用现金: ¥{available_cash:.0f}")
        lines.append(f"总资产: ¥{total_value:.0f}")
        return "\n".join(lines)

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
            "llm_failed": True,
        }
