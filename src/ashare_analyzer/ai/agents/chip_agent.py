"""
Chip Distribution Agent Module

Chip (筹码) distribution analysis agent using LLM for A-share market.

This agent analyzes筹码集中度, profit ratio, and cost distribution using LLM to:
- Assess main force control degree
- Identify accumulation or distribution phases
- Generate trading signals with reasoning
- Evaluate profit-taking risks

Combines quantitative metrics with AI-driven interpretation.
"""

from typing import Any

from ashare_analyzer.ai.tools import ANALYZE_SIGNAL_TOOL
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent

# System prompt for chip analysis
CHIP_SYSTEM_PROMPT = """你是一名专业的A股市场筹码分布分析师。

你的任务：分析筹码结构数据，评估主力控盘程度，生成交易信号。

关键指标解读：
1. 获利比例：越高=越多持有人盈利=潜在抛压越大
2. 集中度：**越低=越集中=主力控盘越强**
   - 例：集中度10% = 高度集中，强控盘
   - 例：集中度30% = 中度集中
   - 例：集中度50%以上 = 分散，弱控盘
3. 平均成本：与当前价格对比，评估盈亏分布
4. 股价成本比：当前价格距离平均持仓成本的偏离程度

分析指引：
- **低集中度（< 15%）**+ 低获利比例：主力吸筹阶段，潜在上涨空间大
- **低集中度（< 15%）**+ 高获利比例：警惕出货，获利了结风险大
- **高集中度（> 30%）**：散户主导，控盘弱，回避
- 股价远高于成本 + 高获利比例：出货风险高
- 股价接近成本 + **低集中度**：良好入场机会

重要说明：
- 集中度10-20% = 优秀（高度集中，强控盘）
- 集中度20-30% = 良好（中度集中）
- 集中度30%以上 = 较差（分散，散户主导）

信号生成规则：
- BUY：高集中度，低至中等获利比例，股价接近成本
- SELL：高获利比例，出货迹象，股价远高于成本
- SHORT：极度高估，高获利比例，明显出货迹象
- HOLD：信号混合或处于过渡期，或高集中度、散户主导

请使用 analyze_signal 函数返回你的分析。"""


class ChipAgent(BaseAgent):
    """
    Chip Distribution Analysis Agent using LLM.

    This agent analyzes筹码结构 using LLM for sophisticated interpretation
    of accumulation/distribution phases and main force control.

    Example:
        agent = ChipAgent()
        signal = agent.analyze({
            "code": "600519",
            "chip": {
                "profit_ratio": 0.75,
                "avg_cost": 1780.0,
                "concentration_90": 12.5,
                "concentration_70": 8.3
            },
            "today": {"close": 1820.0}
        })
    """

    def __init__(self):
        """Initialize the Chip Agent."""
        super().__init__("ChipAgent")
        self._ensure_llm_client()

    def is_available(self) -> bool:
        """Chip agent is available if chip data exists, with fallback."""
        return True

    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute chip distribution analysis using LLM (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - chip: Chip distribution data
                - today: Current day's data with close price

        Returns:
            AgentSignal with LLM-generated trading signal
        """
        stock_code = context.get("code", "")
        chip_data = context.get("chip", {})
        today = context.get("today", {})

        self._logger.debug(f"[{stock_code}] ChipAgent开始筹码分析")

        # Check if chip data is available
        if not chip_data:
            self._logger.warning(f"[{stock_code}] 无筹码数据，跳过筹码分析")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="无筹码分布数据",
                metadata={"error": "missing_chip_data"},
            )

        try:
            # Extract and calculate chip metrics
            chip_metrics = self._calculate_chip_metrics(chip_data, today)

            # Use LLM if available for sophisticated analysis
            if self._llm_client and self._llm_client.is_available():
                llm_analysis = await self._analyze_with_llm(stock_code, chip_metrics)
                if llm_analysis:
                    return self._build_signal_from_llm(llm_analysis, chip_metrics)

            # Fallback to rule-based analysis
            return self._rule_based_analysis(stock_code, chip_metrics)

        except Exception as e:
            self._logger.error(f"[{stock_code}] ChipAgent分析失败: {e}")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning=f"筹码分析失败: {str(e)}",
                metadata={"error": str(e)},
            )

    def _calculate_chip_metrics(self, chip_data: dict[str, Any], today: dict[str, Any]) -> dict[str, Any]:
        """Calculate all chip metrics from raw data."""
        profit_ratio = chip_data.get("profit_ratio", 0)
        avg_cost = chip_data.get("avg_cost", 0)
        concentration_90 = chip_data.get("concentration_90", 100)
        concentration_70 = chip_data.get("concentration_70", 100)
        close = today.get("close", 0)

        # Calculate price relative to average cost
        price_to_cost_ratio = (close - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0

        # Determine control level (concentration_90 is decimal: 0.10 = 10%)
        if concentration_90 < 0.10:
            control_level = "高度控盘" if profit_ratio < 0.8 else "高度控盘-出货风险"
        elif concentration_90 < 0.20:
            control_level = "中度控盘"
        elif concentration_90 < 0.30:
            control_level = "轻度控盘"
        else:
            control_level = "散户主导"

        # Determine phase
        if concentration_90 < 0.15 and profit_ratio < 0.3:
            phase = "accumulation"
        elif concentration_90 < 0.15 and profit_ratio > 0.8:
            phase = "distribution"
        elif concentration_90 < 0.25:
            phase = "consolidation"
        else:
            phase = "unknown"

        return {
            "profit_ratio": profit_ratio,
            "avg_cost": avg_cost,
            "concentration_90": concentration_90,
            "concentration_70": concentration_70,
            "close": close,
            "price_to_cost_ratio": price_to_cost_ratio,
            "control_level": control_level,
            "phase": phase,
        }

    async def _analyze_with_llm(self, stock_code: str, chip_metrics: dict[str, Any]) -> dict[str, Any] | None:
        """Use LLM for chip analysis with Function Call (async)."""
        if not self._llm_client:
            return None

        try:
            prompt = self._build_chip_prompt(stock_code, chip_metrics)

            self._logger.debug(f"[{stock_code}] ChipAgent调用LLM进行筹码分析...")
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=ANALYZE_SIGNAL_TOOL,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
                system_prompt=CHIP_SYSTEM_PROMPT,
                agent_name="ChipAgent",
            )

            if result and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM筹码分析成功")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM筹码分析返回格式无效")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM筹码分析失败: {e}")
            return None

    def _build_chip_prompt(self, stock_code: str, metrics: dict[str, Any]) -> str:
        """Build chip analysis prompt for LLM."""
        return f"""请作为专业的筹码分布分析师，分析以下股票的筹码结构并生成交易信号。

股票代码: {stock_code}

=== 筹码分布数据 ===
- 获利比例: {metrics["profit_ratio"] * 100:.1f}%
- 平均成本: {metrics["avg_cost"]:.2f}
- 90%筹码集中度: {metrics["concentration_90"] * 100:.1f}%
- 70%筹码集中度: {metrics["concentration_70"] * 100:.1f}%
- 当前股价: {metrics["close"]:.2f}

=== 计算指标 ===
- 股价与成本偏离: {metrics["price_to_cost_ratio"]:.1f}%
- 控盘程度: {metrics["control_level"]}
- 当前阶段: {metrics["phase"]}

=== 分析要点 ===
1. 获利比例: 越高说明越多人盈利，抛压风险越大
2. 筹码集中度: 越低说明筹码越集中，主力控盘度越高
3. 股价与成本偏离: 正值越大说明股价远高于成本，回调风险越大

请分析以上筹码数据，评估主力控盘程度和出货风险，使用 analyze_signal 函数返回交易信号。"""

    def _build_signal_from_llm(self, llm_analysis: dict[str, Any], chip_metrics: dict[str, Any]) -> AgentSignal:
        """Build AgentSignal from LLM analysis."""
        signal_str = llm_analysis.get("signal", "hold")
        confidence = llm_analysis.get("confidence", 50)
        reasoning = llm_analysis.get("reasoning", "无详细分析")

        # Convert string signal to SignalType enum
        signal = SignalType.from_string(signal_str)

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "control_assessment": llm_analysis.get("control_assessment", "未知"),
                "phase": llm_analysis.get("phase", "unknown"),
                "risk_level": llm_analysis.get("risk_level", "medium"),
                "key_factors": llm_analysis.get("key_factors", []),
                "recommendation": llm_analysis.get("recommendation", ""),
                # Raw data for reference
                "profit_ratio": round(chip_metrics["profit_ratio"], 4),
                "avg_cost": round(chip_metrics["avg_cost"], 2),
                "concentration_90": round(chip_metrics["concentration_90"], 2),
                "price_to_cost_ratio": round(chip_metrics["price_to_cost_ratio"], 2),
                "analysis_method": "llm",
            },
        )

    def _rule_based_analysis(self, stock_code: str, metrics: dict[str, Any]) -> AgentSignal:
        """Fallback rule-based chip analysis."""
        profit_ratio = metrics["profit_ratio"]
        concentration_90 = metrics["concentration_90"]
        price_to_cost_ratio = metrics["price_to_cost_ratio"]
        control_level = metrics["control_level"]
        phase = metrics["phase"]

        # High concentration scenarios (concentration_90 is decimal: 0.15 = 15%)
        if concentration_90 < 0.15:
            if "高度控盘" in control_level and profit_ratio < 0.3:
                signal_type, confidence = SignalType.BUY, 75
                reasoning = (
                    f"筹码高度集中({concentration_90 * 100:.1f}%)，主力吸筹阶段，"
                    f"获利盘仅{profit_ratio * 100:.1f}%，潜在上涨空间大"
                )
            elif profit_ratio < 0.7:
                signal_type, confidence = SignalType.BUY, 70
                reasoning = f"筹码高度集中，获利盘{profit_ratio * 100:.1f}%适中，主力控盘度高，趋势有望延续"
            elif profit_ratio < 0.9:
                if price_to_cost_ratio > 15:
                    signal_type, confidence = SignalType.HOLD, 60
                    reasoning = (
                        f"筹码集中但获利盘较高({profit_ratio * 100:.1f}%)，"
                        f"现价高于成本{price_to_cost_ratio:.1f}%，警惕出货"
                    )
                else:
                    signal_type, confidence = SignalType.BUY, 60
                    reasoning = f"筹码集中，获利盘{profit_ratio * 100:.1f}%，关注回调机会"
            else:
                # A股不支持做空，获利盘过高时建议卖出
                if price_to_cost_ratio > 20:
                    signal_type, confidence = SignalType.SELL, 75
                    reasoning = (
                        f"获利盘过高({profit_ratio * 100:.1f}%)且股价远高于成本"
                        f"{price_to_cost_ratio:.1f}%，主力出货概率大，建议卖出"
                    )
                else:
                    signal_type, confidence = SignalType.SELL, 70
                    reasoning = f"获利盘过高({profit_ratio * 100:.1f}%)，筹码集中度高，主力可能正在出货"
        elif concentration_90 < 0.25:
            if profit_ratio < 0.7:
                signal_type, confidence = SignalType.HOLD, 50
                reasoning = f"筹码相对集中({concentration_90 * 100:.1f}%)，获利盘{profit_ratio * 100:.1f}%，趋势待确认"
            else:
                signal_type, confidence = SignalType.HOLD, 45
                reasoning = f"筹码相对集中但获利盘较高({profit_ratio * 100:.1f}%)，谨慎持有"
        else:
            signal_type, confidence = SignalType.HOLD, 40
            reasoning = f"筹码分散({concentration_90 * 100:.1f}%)，散户主导，缺乏主力"

        self._logger.debug(f"[{stock_code}] ChipAgent规则分析完成: {signal_type.to_string()} (置信度{confidence}%)")

        return AgentSignal(
            agent_name=self.name,
            signal=signal_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "profit_ratio": round(profit_ratio, 4),
                "avg_cost": round(metrics["avg_cost"], 2),
                "concentration_90": round(concentration_90, 2),
                "price_to_cost_ratio": round(price_to_cost_ratio, 2),
                "control_level": control_level,
                "phase": phase,
                "analysis_method": "rule_based",
            },
        )
