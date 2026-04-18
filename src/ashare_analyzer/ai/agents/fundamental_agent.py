"""
Fundamental Analysis Agent Module

Specialized agent for analyzing company fundamentals and financial metrics.

This agent focuses exclusively on:
- Financial statement analysis (ROE, ROA, margins)
- Balance sheet health (debt ratios, current ratio)
- Earnings consistency and growth
- Profitability metrics

Inspired by ai-hedge-fund's Fundamentals Analyst.
"""

import logging
from typing import Any

from ashare_analyzer.ai.tools import ANALYZE_SIGNAL_TOOL
from ashare_analyzer.exceptions import handle_errors
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent
from .utils import FinancialScorer, score_to_signal

logger = logging.getLogger(__name__)

# System prompt for fundamental analysis
FUNDAMENTAL_SYSTEM_PROMPT = """你是一名专业的A股市场基本面分析师。

你的任务：分析财务指标，评估公司健康状况，生成交易信号。

=== 你将收到的关键指标 ===
- 盈利能力：ROE、ROA、net_margin、gross_margin
- 财务健康：debt_to_equity、current_ratio、interest_coverage
- 成长性：revenue_growth、earnings_growth
- 稳定性：eps_consistency、revenue_consistency

=== 解读指引 ===
- ROE > 15% 为优秀，但需考虑行业特性
- debt_to_equity < 0.5 通常安全，但公用事业/金融行业可能更高
- 关注盈利质量，而非仅看增速
- 关注时间维度上的一致性模式
- 有行业均值时进行横向对比

=== 信号指引 ===
BUY条件：
- 盈利能力强（ROE > 15%，净利率 > 10%）
- 财务健康（debt_to_equity < 0.5，current_ratio > 1.5）
- 正增长趋势
- 盈利稳定性好

SELL条件：
- 盈利能力弱（ROE < 5%，净利率 < 5%）
- 杠杆过高（debt_to_equity > 1.0）
- 营收或利润下滑
- 出现财务困境迹象

HOLD条件：
- 信号混合
- 指标尚可但不突出
- 未来前景不确定

=== 置信度等级 ===
- 90-100%：各维度基本面强劲
- 70-89%：基本面良好，存在少量担忧
- 50-69%：信号混合或数据有限
- 30-49%：基本面较弱或存在重大隐忧
- 10-29%：多维度基本面表现不佳

请使用 analyze_signal 函数返回你的分析。"""


class FundamentalAgent(BaseAgent):
    """
    Fundamental Analysis Agent for financial metrics assessment.

    This agent analyzes:
    - Profitability metrics (ROE, ROA, net margin)
    - Financial health (debt-to-equity, current ratio)
    - Earnings stability and growth
    - Balance sheet strength

    Uses LLM for analysis when available, with heuristic fallback.

    Example:
        agent = FundamentalAgent()
        signal = await agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "financial_data": {...}
        })
    """

    def __init__(self):
        """Initialize the Fundamental Agent."""
        super().__init__("FundamentalAgent")
        self._ensure_llm_client()

    def _build_fundamental_prompt(self, stock_code: str, financial_data: dict[str, Any]) -> str:
        """Build the fundamental analysis prompt for LLM."""
        # Build metrics sections
        profitability = []
        if financial_data.get("roe"):
            profitability.append(f"- ROE: {financial_data['roe']:.2f}%")
        if financial_data.get("roa"):
            profitability.append(f"- ROA: {financial_data['roa']:.2f}%")
        if financial_data.get("net_margin"):
            profitability.append(f"- Net Margin: {financial_data['net_margin']:.2f}%")
        if financial_data.get("gross_margin"):
            profitability.append(f"- Gross Margin: {financial_data['gross_margin']:.2f}%")

        health = []
        if financial_data.get("debt_to_equity") is not None:
            health.append(f"- Debt-to-Equity: {financial_data['debt_to_equity']:.2f}")
        if financial_data.get("asset_liability_ratio") is not None:
            health.append(f"- Asset-Liability Ratio: {financial_data['asset_liability_ratio']:.1f}%")
        if financial_data.get("current_ratio"):
            health.append(f"- Current Ratio: {financial_data['current_ratio']:.2f}")
        if financial_data.get("interest_coverage"):
            health.append(f"- Interest Coverage: {financial_data['interest_coverage']:.2f}")

        growth = []
        if financial_data.get("revenue_growth") is not None:
            growth.append(f"- Revenue Growth: {financial_data['revenue_growth']:.2f}%")
        if financial_data.get("earnings_growth") is not None:
            growth.append(f"- Earnings Growth: {financial_data['earnings_growth']:.2f}%")

        consistency = []
        if financial_data.get("eps_consistency") is not None:
            consistency.append(f"- EPS Consistency: {financial_data['eps_consistency']:.1f}%")
        if financial_data.get("revenue_consistency") is not None:
            consistency.append(f"- Revenue Consistency: {financial_data['revenue_consistency']:.1f}%")

        # Build the prompt
        sections = [f"=== {stock_code} Fundamental Analysis ===", "", "=== Profitability Metrics ==="]
        if profitability:
            sections.extend(profitability)
        else:
            sections.append("No profitability data available")

        sections.extend(["", "=== Financial Health ==="])
        if health:
            sections.extend(health)
        else:
            sections.append("No health metrics available")

        sections.extend(["", "=== Growth Metrics ==="])
        if growth:
            sections.extend(growth)
        else:
            sections.append("No growth data available")

        sections.extend(["", "=== Consistency Metrics ==="])
        if consistency:
            sections.extend(consistency)
        else:
            sections.append("No consistency data available")

        sections.extend(["", "请根据以上财务指标进行分析，使用 analyze_signal 函数返回交易信号。"])

        return "\n".join(sections)

    async def _analyze_with_llm(self, stock_code: str, financial_data: dict[str, Any]) -> dict[str, Any] | None:
        """Use LLM for fundamental analysis with Function Call."""
        if not self._llm_client:
            return None

        try:
            prompt = self._build_fundamental_prompt(stock_code, financial_data)

            self._logger.debug(f"[{stock_code}] FundamentalAgent calling LLM...")
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=ANALYZE_SIGNAL_TOOL,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
                agent_name="FundamentalAgent",
            )

            if result and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM fundamental analysis successful: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM fundamental analysis returned invalid format")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM fundamental analysis failed: {e}")
            return None

    def _build_signal_from_llm(self, llm_analysis: dict[str, Any], financial_data: dict[str, Any]) -> AgentSignal:
        """Build AgentSignal from LLM analysis result."""
        signal_str = llm_analysis.get("signal", "hold")
        signal = SignalType.from_string(signal_str)
        confidence = llm_analysis.get("confidence", 50)
        reasoning = llm_analysis.get("reasoning", "无详细分析")
        key_metrics = llm_analysis.get("key_metrics", {})

        # Build metadata with key financials
        metadata = {
            "analysis_method": "llm",
            "roe": financial_data.get("roe"),
            "net_margin": financial_data.get("net_margin"),
            "debt_to_equity": financial_data.get("debt_to_equity"),
            "current_ratio": financial_data.get("current_ratio"),
            "revenue_growth": financial_data.get("revenue_growth"),
        }
        metadata.update(key_metrics)

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata,
        )

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    @handle_errors("基本面分析失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute fundamental analysis using LLM (async).

        Args:
            context: Analysis context containing:
                - code: Stock code
                - stock_name: Stock name
                - financial_data: Financial metrics dict with keys:
                    - roe: Return on equity
                    - roa: Return on assets
                    - net_margin: Net profit margin
                    - gross_margin: Gross profit margin
                    - debt_to_equity: Debt to equity ratio
                    - current_ratio: Current ratio
                    - revenue_growth: Revenue growth rate
                    - earnings_growth: Earnings growth rate
                    - eps_consistency: EPS consistency score

        Returns:
            AgentSignal with fundamental-based trading signal
        """
        stock_code = context.get("code", "")
        financial_data = context.get("financial_data", {})

        self._logger.debug(f"[{stock_code}] FundamentalAgent starting analysis")

        if not financial_data:
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="无财务数据可用",
                metadata={"error": "no_financial_data"},
            )

        # Try LLM analysis first
        if self._llm_client and self._llm_client.is_available():
            llm_result = await self._analyze_with_llm(stock_code, financial_data)
            if llm_result:
                return self._build_signal_from_llm(llm_result, financial_data)

        # Fallback to heuristic analysis
        self._logger.warning(f"[{stock_code}] LLM unavailable, using heuristic fallback")
        return await self._heuristic_analysis(context)

    async def _heuristic_analysis(self, context: dict[str, Any]) -> AgentSignal:
        """Fallback heuristic analysis when LLM is unavailable."""
        stock_code = context.get("code", "")
        financial_data = context.get("financial_data", {})

        # 检查是否有基本财务数据（ROE等）还是只有PE/PB
        has_basic_metrics = any(
            financial_data.get(k) is not None and financial_data.get(k) != 0
            for k in ["roe", "roa", "net_margin", "gross_margin", "debt_to_equity"]
        )

        # 如果只有PE/PB等基础数据，进行简化分析
        if not has_basic_metrics:
            return self._analyze_with_basic_data(stock_code, financial_data)

        # Calculate scores for each dimension
        profitability_score = self._analyze_profitability(financial_data)
        health_score = self._analyze_financial_health(financial_data)
        growth_score = self._analyze_growth(financial_data)
        consistency_score = self._analyze_consistency(financial_data)

        # Calculate weighted total score (max 100)
        total_score = profitability_score * 0.35 + health_score * 0.30 + growth_score * 0.20 + consistency_score * 0.15

        # Generate signal based on score thresholds
        signal, confidence = self._score_to_signal(total_score)

        # Build reasoning
        reasoning_parts = []
        if profitability_score >= 8:
            reasoning_parts.append(f"盈利能力优秀({profitability_score}/10)")
        elif profitability_score >= 5:
            reasoning_parts.append(f"盈利能力良好({profitability_score}/10)")
        else:
            reasoning_parts.append(f"盈利能力较弱({profitability_score}/10)")

        if health_score >= 8:
            reasoning_parts.append(f"财务健康优秀({health_score}/10)")
        elif health_score >= 5:
            reasoning_parts.append(f"财务健康良好({health_score}/10)")
        else:
            reasoning_parts.append(f"财务健康需关注({health_score}/10)")

        reasoning = " / ".join(reasoning_parts)

        self._logger.debug(
            f"[{stock_code}] FundamentalAgent分析完成: {signal} (总分{total_score:.1f}/100, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "analysis_method": "heuristic",
                "total_score": round(total_score, 1),
                "profitability_score": profitability_score,
                "health_score": health_score,
                "growth_score": growth_score,
                "consistency_score": consistency_score,
                "roe": financial_data.get("roe"),
                "net_margin": financial_data.get("net_margin"),
                "debt_to_equity": financial_data.get("debt_to_equity"),
                "current_ratio": financial_data.get("current_ratio"),
                "revenue_growth": financial_data.get("revenue_growth"),
            },
        )

    def _analyze_profitability(self, data: dict[str, Any]) -> int:
        """
        Analyze profitability metrics (0-10 scale).

        Scoring:
        - ROE > 15%: +3, > 10%: +2, > 5%: +1
        - Net Margin > 20%: +3, > 15%: +2, > 10%: +1
        - Gross Margin > 40%: +2, > 30%: +1
        - ROA > 8%: +2, > 5%: +1
        """
        score = 0

        # ROE scoring
        roe = data.get("roe", 0)
        if roe > 15:
            score += 3
        elif roe > 10:
            score += 2
        elif roe > 5:
            score += 1

        # Net margin scoring
        net_margin = data.get("net_margin", 0)
        if net_margin > 20:
            score += 3
        elif net_margin > 15:
            score += 2
        elif net_margin > 10:
            score += 1

        # Gross margin scoring
        gross_margin = data.get("gross_margin", 0)
        if gross_margin > 40:
            score += 2
        elif gross_margin > 30:
            score += 1

        # ROA scoring
        roa = data.get("roa", 0)
        if roa > 8:
            score += 2
        elif roa > 5:
            score += 1

        return min(score, 10)

    def _analyze_financial_health(self, data: dict[str, Any]) -> int:
        """Analyze financial health metrics (0-10 scale)."""
        return FinancialScorer.analyze_financial_health(data)

    def _analyze_growth(self, data: dict[str, Any]) -> int:
        """
        Analyze growth metrics (0-10 scale).

        Scoring:
        - Revenue Growth > 20%: +3, > 10%: +2, > 5%: +1
        - Earnings Growth > 20%: +3, > 10%: +2, > 5%: +1
        - Book Value Growth > 15%: +2, > 10%: +1
        """
        score = 0

        # Revenue growth scoring
        revenue_growth = data.get("revenue_growth", 0)
        if revenue_growth > 20:
            score += 3
        elif revenue_growth > 10:
            score += 2
        elif revenue_growth > 5:
            score += 1

        # Earnings growth scoring
        earnings_growth = data.get("earnings_growth", 0)
        if earnings_growth > 20:
            score += 3
        elif earnings_growth > 10:
            score += 2
        elif earnings_growth > 5:
            score += 1

        # Book value growth scoring
        bv_growth = data.get("book_value_growth", 0)
        if bv_growth > 15:
            score += 2
        elif bv_growth > 10:
            score += 1

        return min(score, 10)

    def _analyze_consistency(self, data: dict[str, Any]) -> int:
        """
        Analyze earnings consistency (0-10 scale).

        Scoring:
        - EPS consistency > 80%: +4, > 60%: +2
        - Revenue consistency > 80%: +3, > 60%: +1
        - Dividend track record: +3
        """
        score = 0

        # EPS consistency scoring
        eps_consistency = data.get("eps_consistency", 0)
        if eps_consistency > 80:
            score += 4
        elif eps_consistency > 60:
            score += 2

        # Revenue consistency scoring
        revenue_consistency = data.get("revenue_consistency", 0)
        if revenue_consistency > 80:
            score += 3
        elif revenue_consistency > 60:
            score += 1

        # Dividend track record
        dividend_record = data.get("dividend_record", False)
        if dividend_record:
            score += 3

        return min(score, 10)

    def _score_to_signal(self, score: float) -> tuple[SignalType, int]:
        return score_to_signal(score, buy_threshold=70.0, sell_threshold=30.0)

    def _analyze_with_basic_data(self, stock_code: str, financial_data: dict[str, Any]) -> AgentSignal:
        """
        当只有基础数据（PE/PB等）时进行简化分析。

        基于PE/PB、波动率等指标给出基本面评估。
        """
        pe_ratio = financial_data.get("pe_ratio", 0)
        pb_ratio = financial_data.get("pb_ratio", 0)
        volatility = financial_data.get("volatility", 0)
        price_momentum = financial_data.get("price_momentum_20d", 0)

        # 简化评分（0-100）
        score = 50  # 中性起点

        # PE评分
        if pe_ratio > 0:
            if pe_ratio < 15:
                score += 15
            elif pe_ratio < 25:
                score += 5
            elif pe_ratio > 50:
                score -= 15
            elif pe_ratio > 30:
                score -= 5

        # PB评分
        if pb_ratio > 0:
            if pb_ratio < 1.5:
                score += 10
            elif pb_ratio > 5:
                score -= 10

        # 波动率评分（低波动加分）
        if volatility > 0:
            if volatility < 20:
                score += 5
            elif volatility > 50:
                score -= 5

        # 价格动量评分
        if price_momentum > 10:
            score += 5
        elif price_momentum < -10:
            score -= 5

        # 限制分数范围
        score = max(0, min(100, score))

        # 生成信号
        signal, confidence = self._score_to_signal(score)

        # 构建理由
        reasoning_parts = []
        if pe_ratio > 0:
            if pe_ratio < 15:
                reasoning_parts.append(f"PE{pe_ratio:.1f}偏低")
            elif pe_ratio > 30:
                reasoning_parts.append(f"PE{pe_ratio:.1f}偏高")
            else:
                reasoning_parts.append(f"PE{pe_ratio:.1f}合理")

        if pb_ratio > 0:
            if pb_ratio < 1.5:
                reasoning_parts.append(f"PB{pb_ratio:.1f}偏低")
            elif pb_ratio > 3:
                reasoning_parts.append(f"PB{pb_ratio:.1f}偏高")

        if volatility > 0:
            reasoning_parts.append(f"波动率{volatility:.1f}%")

        reasoning = " / ".join(reasoning_parts) if reasoning_parts else "基于有限数据的基础分析"

        self._logger.debug(
            f"[{stock_code}] FundamentalAgent简化分析完成: {signal} (基础评分{score:.1f}/100, 置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "analysis_type": "basic",
                "score": round(score, 1),
                "pe_ratio": pe_ratio,
                "pb_ratio": pb_ratio,
                "volatility": volatility,
                "price_momentum_20d": price_momentum,
            },
        )
