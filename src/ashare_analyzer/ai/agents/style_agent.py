"""
Style Agent Module

Merged investment style agent combining Value, Growth, and Momentum analysis.

This agent provides a unified investment style analysis with:
- Value investing (Buffett/Graham principles): 30%
- Growth investing (disruptive innovation): 35%
- Momentum investing (trend following): 35%
"""

import math
from typing import Any

from ashare_analyzer.ai.tools import ANALYZE_SIGNAL_TOOL
from ashare_analyzer.constants import (
    A_SHARE_GRAHAM_MULTIPLIER,
    A_SHARE_MOS_MODERATE,
    A_SHARE_MOS_STRONG,
    A_SHARE_PB_EXCELLENT,
    A_SHARE_PB_GOOD,
    A_SHARE_PE_ACCEPTABLE,
    A_SHARE_PE_EXCELLENT,
    A_SHARE_PE_GOOD,
)
from ashare_analyzer.exceptions import handle_errors
from ashare_analyzer.models import AgentSignal, SignalType

from .base import BaseAgent
from .utils import FinancialScorer, score_to_signal

# System prompt for style analysis
STYLE_SYSTEM_PROMPT = """你是一位专业的投资风格分析师。

你的任务：分析股票的投资风格特征并生成交易信号。

=== 风格维度 ===
你将收到三个维度的数据：

1. 价值属性：
   - PE ratio、PB ratio（越低越有价值属性）
   - margin of safety（估值安全边际）
   - 股息率（越高越好）
   - 财务稳定性指标

2. 成长属性：
   - 营收增长率
   - EPS 增长率
   - 市场机会规模
   - 创新指标

3. 动量属性：
   - 价格趋势（均线排列）
   - 成交量确认
   - 相对市场强弱
   - 技术信号

=== 分析方法 ===
- 股票可以同时展现多种风格特征
- 根据当前数据动态调整风格权重，而非固定比例
- 考虑该股票哪种风格占主导
- 评估每种风格信号的质量
- 市场环境可能偏好特定风格

=== 信号指引 ===
BUY 条件：
- 价值属性强劲，margin of safety >20%
- 成长属性强劲，增长指标可持续
- 动量属性强劲，有成交量确认
- 一种风格明显主导且有清晰信号

SELL 条件：
- 估值过高，margin of safety 为负
- 成长性恶化
- 动量趋势破位
- 风格信号相互矛盾

HOLD 条件：
- 风格信号混合
- 各维度指标适中
- 主导风格不明确

=== 置信度等级 ===
- 90-100%: 一种风格强烈主导，指标明确
- 70-89%: 风格倾向明确，有数据支撑
- 50-69%: 风格信号中等，存在一定不确定性
- 30-49%: 风格信号矛盾
- 10-29%: 无明确风格模式

请使用 analyze_signal 函数返回你的分析结果。"""


class StyleAgent(BaseAgent):
    """
    Unified Investment Style Agent.

    Combines three investment styles into one comprehensive analysis:
    1. Value (30%): Margin of safety, strong fundamentals, moat indicators
    2. Growth (35%): Revenue/EPS growth, innovation, market opportunity
    3. Momentum (35%): Price trends, volume confirmation, relative strength

    This consolidation reduces agent count from 9 to 7 while maintaining
    full analytical coverage of investment styles.

    Attributes:
        None - uses context data only

    Example:
        agent = StyleAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "financial_data": {...},
            "valuation_data": {...},
            "growth_data": {...},
            "price_data": {...},
            "technical_data": {...},
            "market_data": {...},
        })
    """

    # Weight distribution across three styles
    STYLE_WEIGHTS = {
        "value": 0.30,
        "growth": 0.35,
        "momentum": 0.35,
    }

    # Value sub-weights
    VALUE_WEIGHTS = {
        "valuation": 0.30,
        "fundamentals": 0.25,
        "moat": 0.20,
        "management": 0.15,
        "safety": 0.10,
    }

    # Growth sub-weights
    GROWTH_WEIGHTS = {
        "growth_metrics": 0.40,
        "innovation": 0.25,
        "market": 0.20,
        "valuation": 0.15,
    }

    # Momentum sub-weights
    MOMENTUM_WEIGHTS = {
        "price_momentum": 0.35,
        "trend": 0.25,
        "volume": 0.20,
        "relative_strength": 0.20,
    }

    def __init__(self):
        """Initialize the Style Agent."""
        super().__init__("StyleAgent")
        self._ensure_llm_client()

    def _build_style_prompt(self, stock_code: str, context: dict[str, Any]) -> str:
        """Build the style analysis prompt for LLM."""
        financial_data = context.get("financial_data", {})
        valuation_data = context.get("valuation_data", {})
        technical_data = context.get("technical_data", {})
        today = context.get("today", {})

        sections = [f"=== {stock_code} Style Analysis ===", ""]

        # Value metrics
        sections.append("=== Value Attributes ===")
        if valuation_data.get("pe_ratio"):
            sections.append(f"- PE Ratio: {valuation_data['pe_ratio']:.2f}")
        if valuation_data.get("pb_ratio"):
            sections.append(f"- PB Ratio: {valuation_data['pb_ratio']:.2f}")
        if valuation_data.get("margin_of_safety") is not None:
            sections.append(f"- Margin of Safety: {valuation_data['margin_of_safety']:.1f}%")
        if valuation_data.get("dividend_yield"):
            sections.append(f"- Dividend Yield: {valuation_data['dividend_yield']:.2f}%")
        if financial_data.get("debt_to_equity") is not None:
            sections.append(f"- Debt-to-Equity: {financial_data['debt_to_equity']:.2f}")
        if financial_data.get("asset_liability_ratio") is not None:
            sections.append(f"- Asset-Liability Ratio: {financial_data['asset_liability_ratio']:.1f}%")
        if not any(
            [
                valuation_data.get("pe_ratio"),
                valuation_data.get("pb_ratio"),
                valuation_data.get("dividend_yield"),
            ]
        ):
            sections.append("Limited value data available")

        # Growth metrics
        sections.extend(["", "=== Growth Attributes ==="])
        if financial_data.get("revenue_growth") is not None:
            sections.append(f"- Revenue Growth: {financial_data['revenue_growth']:.2f}%")
        if financial_data.get("earnings_growth") is not None:
            sections.append(f"- EPS Growth: {financial_data['earnings_growth']:.2f}%")
        if financial_data.get("roe"):
            sections.append(f"- ROE: {financial_data['roe']:.2f}%")
        if not any([financial_data.get("revenue_growth"), financial_data.get("earnings_growth")]):
            sections.append("Limited growth data available")

        # Momentum metrics
        sections.extend(["", "=== Momentum Attributes ==="])
        if today.get("close") and today.get("ma5"):
            sections.append(f"- Price vs MA5: {((today['close'] - today['ma5']) / today['ma5'] * 100):.2f}%")
        if today.get("pct_chg") is not None:
            sections.append(f"- Price Change: {today['pct_chg']:.2f}%")
        if technical_data.get("rsi_14"):
            sections.append(f"- RSI(14): {technical_data['rsi_14']:.1f}")
        if technical_data.get("macd_hist"):
            sections.append(f"- MACD Histogram: {technical_data['macd_hist']:.4f}")
        if today.get("volume_ratio"):
            sections.append(f"- Volume Ratio: {today['volume_ratio']:.2f}")
        if not any([today.get("close"), today.get("pct_chg"), technical_data.get("rsi_14")]):
            sections.append("Limited momentum data available")

        sections.extend(
            [
                "",
                "请根据以上风格维度分析，判断该股票的投资风格特征，使用 analyze_signal 函数返回交易信号。",
            ]
        )

        return "\n".join(sections)

    async def _analyze_with_llm(self, stock_code: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """Use LLM for style analysis with Function Call."""
        if not self._llm_client:
            return None

        try:
            prompt = self._build_style_prompt(stock_code, context)

            self._logger.debug(f"[{stock_code}] StyleAgent calling LLM...")
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=ANALYZE_SIGNAL_TOOL,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                system_prompt=STYLE_SYSTEM_PROMPT,
                agent_name="StyleAgent",
            )

            if result and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM style analysis successful: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM style analysis returned invalid format")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM style analysis failed: {e}")
            return None

    def _build_signal_from_llm(self, llm_analysis: dict[str, Any], context: dict[str, Any]) -> AgentSignal:
        """Build AgentSignal from LLM analysis result."""
        signal_str = llm_analysis.get("signal", "hold")
        signal = SignalType.from_string(signal_str)
        confidence = llm_analysis.get("confidence", 50)
        reasoning = llm_analysis.get("reasoning", "无详细分析")
        key_metrics = llm_analysis.get("key_metrics", {})

        # Calculate style scores for metadata consistency
        current_price = context.get("current_price", 0)
        financial_data = context.get("financial_data", {})
        valuation_data = context.get("valuation_data", {})
        growth_data = context.get("growth_data", {})
        price_data = context.get("price_data", {})
        technical_data = context.get("technical_data", {})
        market_data = context.get("market_data", {})

        value_score = self._heuristic_value_score(current_price, financial_data, valuation_data)
        growth_score = self._heuristic_growth_score(current_price, financial_data, growth_data)
        momentum_score = self._heuristic_momentum_score(price_data, technical_data, market_data)
        total_score = (
            value_score * self.STYLE_WEIGHTS["value"]
            + growth_score * self.STYLE_WEIGHTS["growth"]
            + momentum_score * self.STYLE_WEIGHTS["momentum"]
        )

        metadata = {
            "analysis_method": "llm",
            "value_score": value_score,
            "growth_score": growth_score,
            "momentum_score": momentum_score,
            "total_score": total_score,
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

    @handle_errors("风格分析失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute style analysis using LLM (async).

        Args:
            context: Analysis context containing financial_data, valuation_data, technical_data

        Returns:
            AgentSignal with style-based trading signal
        """
        stock_code = context.get("code", "")

        self._logger.debug(f"[{stock_code}] StyleAgent starting analysis")

        # Try LLM analysis first
        if self._llm_client and self._llm_client.is_available():
            llm_result = await self._analyze_with_llm(stock_code, context)
            if llm_result:
                return self._build_signal_from_llm(llm_result, context)

        # Fallback to heuristic analysis
        self._logger.warning(f"[{stock_code}] LLM unavailable, using heuristic fallback")
        return await self._heuristic_analysis(context)

    async def _heuristic_analysis(self, context: dict[str, Any]) -> AgentSignal:
        """Fallback heuristic analysis when LLM is unavailable."""
        stock_code = context.get("code", "")
        current_price = context.get("current_price", 0)
        financial_data = context.get("financial_data", {})
        valuation_data = context.get("valuation_data", {})
        growth_data = context.get("growth_data", {})
        price_data = context.get("price_data", {})
        technical_data = context.get("technical_data", {})
        market_data = context.get("market_data", {})

        # Calculate style scores using heuristic methods
        value_score = self._heuristic_value_score(current_price, financial_data, valuation_data)
        growth_score = self._heuristic_growth_score(current_price, financial_data, growth_data)
        momentum_score = self._heuristic_momentum_score(price_data, technical_data, market_data)

        # Combine scores with fixed weights
        total_score = (
            value_score * self.STYLE_WEIGHTS["value"]
            + growth_score * self.STYLE_WEIGHTS["growth"]
            + momentum_score * self.STYLE_WEIGHTS["momentum"]
        )

        # Generate signal
        signal, confidence = score_to_signal(total_score)

        # Build reasoning
        reasoning_parts = [
            f"价值属性{'强' if value_score >= 70 else '中' if value_score >= 40 else '弱'}({value_score:.0f})",
            f"成长属性{'强' if growth_score >= 70 else '中' if growth_score >= 40 else '弱'}({growth_score:.0f})",
            f"动量属性{'强' if momentum_score >= 70 else '中' if momentum_score >= 40 else '弱'}({momentum_score:.0f})",
        ]
        reasoning = " / ".join(reasoning_parts)

        self._logger.debug(
            f"[{stock_code}] StyleAgent heuristic analysis: {signal} "
            f"(价值{value_score:.0f}/成长{growth_score:.0f}/动量{momentum_score:.0f}, "
            f"置信度{confidence}%)"
        )

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "analysis_method": "heuristic_fallback",
                "value_score": value_score,
                "growth_score": growth_score,
                "momentum_score": momentum_score,
                "total_score": total_score,
            },
        )

    def _heuristic_value_score(self, current_price: float, financial_data: dict, valuation_data: dict) -> float:
        """Analyze value investing characteristics (0-100 scale)."""
        if not financial_data:
            return 50.0  # Neutral if no data

        # Calculate sub-scores (0-10 each)
        valuation = self._score_valuation(current_price, valuation_data)
        fundamentals = self._score_fundamentals(financial_data)
        moat = self._score_moat(financial_data)
        management = self._score_management(financial_data)
        safety = FinancialScorer.analyze_safety(financial_data)

        # Weighted average, scaled to 0-100
        score = (
            valuation * self.VALUE_WEIGHTS["valuation"]
            + fundamentals * self.VALUE_WEIGHTS["fundamentals"]
            + moat * self.VALUE_WEIGHTS["moat"]
            + management * self.VALUE_WEIGHTS["management"]
            + safety * self.VALUE_WEIGHTS["safety"]
        ) * 10

        return min(score, 100)

    def _heuristic_growth_score(self, current_price: float, financial_data: dict, growth_data: dict) -> float:
        """Analyze growth investing characteristics (0-100 scale)."""
        if not growth_data:
            return 50.0  # Neutral if no data

        # Check for detailed growth metrics
        has_detailed = any(
            growth_data.get(k) is not None and growth_data.get(k) != 0 for k in ["revenue_cagr", "eps_cagr", "fcf_cagr"]
        )

        if not has_detailed:
            # Use simplified analysis based on price momentum
            return self._analyze_basic_growth(growth_data)

        # Calculate sub-scores (0-10 each)
        growth_metrics = self._score_growth_metrics(growth_data)
        innovation = self._score_innovation(growth_data)
        market = self._score_market_opportunity(growth_data)
        valuation = self._score_growth_valuation(current_price, financial_data, growth_data)

        # Weighted average, scaled to 0-100
        score = (
            growth_metrics * self.GROWTH_WEIGHTS["growth_metrics"]
            + innovation * self.GROWTH_WEIGHTS["innovation"]
            + market * self.GROWTH_WEIGHTS["market"]
            + valuation * self.GROWTH_WEIGHTS["valuation"]
        ) * 10

        return min(score, 100)

    def _heuristic_momentum_score(self, price_data: dict, technical_data: dict, market_data: dict) -> float:
        """Analyze momentum investing characteristics (0-100 scale)."""
        if not price_data:
            return 50.0  # Neutral if no data

        # Calculate sub-scores (0-10 each)
        price_momentum = self._score_price_momentum(price_data)
        trend = self._score_trend_strength(technical_data)
        volume = self._score_volume(technical_data)
        rs = self._score_relative_strength(market_data)

        # Weighted average, scaled to 0-100
        score = (
            price_momentum * self.MOMENTUM_WEIGHTS["price_momentum"]
            + trend * self.MOMENTUM_WEIGHTS["trend"]
            + volume * self.MOMENTUM_WEIGHTS["volume"]
            + rs * self.MOMENTUM_WEIGHTS["relative_strength"]
        ) * 10

        return min(score, 100)

    # Value Analysis Methods
    def _score_valuation(self, current_price: float, data: dict) -> int:
        """Score valuation metrics (0-10).

        A-share adjusted thresholds:
        - P/E: Average PE ~20-30 (vs US 15-20)
        - P/B: Average PB ~2-3 (vs US 1.5-2)
        - Margin of safety: More lenient thresholds for A-share volatility
        """
        score = 0
        if current_price <= 0:
            return 5

        # P/E scoring - A-share adjusted (average PE ~20-30)
        eps = data.get("eps", 0)
        if eps > 0:
            pe = current_price / eps
            if pe < A_SHARE_PE_EXCELLENT:  # < 20: excellent value
                score += 3
            elif pe < A_SHARE_PE_GOOD:  # < 25: good value
                score += 2
            elif pe < A_SHARE_PE_ACCEPTABLE:  # < 35: acceptable
                score += 1

        # P/B scoring - A-share adjusted (average PB ~2-3)
        bvps = data.get("book_value_per_share", 0)
        if bvps > 0:
            pb = current_price / bvps
            if pb < A_SHARE_PB_EXCELLENT:  # < 2.5: excellent value
                score += 2
            elif pb < A_SHARE_PB_GOOD:  # < 3.5: good value
                score += 1

        # Margin of safety - A-share adjusted
        graham = self._calculate_graham_number(data, a_share_adjusted=True)
        if graham > 0 and current_price > 0:
            mos = (graham - current_price) / current_price * 100
            if mos > A_SHARE_MOS_STRONG:  # > 20%: strong margin
                score += 3
            elif mos > A_SHARE_MOS_MODERATE:  # > 10%: moderate margin
                score += 2

        return min(score, 10)

    def _score_fundamentals(self, data: dict) -> int:
        """Score fundamental metrics (0-10)."""
        score = 0
        roe = data.get("roe", 0)
        if roe > 15:
            score += 3
        elif roe > 12:
            score += 2
        elif roe > 8:
            score += 1

        net_margin = data.get("net_margin", 0)
        if net_margin > 20:
            score += 2
        elif net_margin > 15:
            score += 1

        op_margin = data.get("operating_margin", 0)
        if op_margin > 15:
            score += 2
        elif op_margin > 10:
            score += 1

        roa = data.get("roa", 0)
        if roa > 8:
            score += 1

        return min(score, 10)

    def _score_moat(self, data: dict) -> int:
        """Score competitive moat (0-10)."""
        score = 0
        roe_consistency = data.get("roe_consistency", 0)
        if roe_consistency > 80:
            score += 3
        elif roe_consistency > 60:
            score += 2

        gross_margin = data.get("gross_margin", 0)
        if gross_margin > 40:
            score += 2
        elif gross_margin > 30:
            score += 1

        asset_turnover = data.get("asset_turnover", 0)
        if asset_turnover > 1.0:
            score += 1

        stability = data.get("performance_stability", 0)
        if stability > 70:
            score += 1

        return min(score, 10)

    def _score_management(self, data: dict) -> int:
        """Score management quality (0-10)."""
        score = 0
        if data.get("share_buybacks", False):
            score += 3
        if data.get("dividend_record", False):
            score += 2

        payout_ratio = data.get("payout_ratio", 0)
        if 0 < payout_ratio < 50:
            score += 2

        roic = data.get("roic", 0)
        if roic > 12:
            score += 3
        elif roic > 8:
            score += 2

        return min(score, 10)

    # Growth Analysis Methods
    def _score_growth_metrics(self, data: dict) -> int:
        """Score growth metrics (0-10)."""
        score = 0
        revenue_cagr = data.get("revenue_cagr", 0)
        if revenue_cagr > 30:
            score += 4
        elif revenue_cagr > 20:
            score += 3
        elif revenue_cagr > 15:
            score += 2
        elif revenue_cagr > 10:
            score += 1

        eps_cagr = data.get("eps_cagr", 0)
        if eps_cagr > 30:
            score += 3
        elif eps_cagr > 20:
            score += 2
        elif eps_cagr > 15:
            score += 1

        fcf_cagr = data.get("fcf_cagr", 0)
        if fcf_cagr > 20:
            score += 2
        elif fcf_cagr > 10:
            score += 1

        return min(score, 10)

    def _score_innovation(self, data: dict) -> int:
        """Score innovation metrics (0-10)."""
        score = 0
        rd_intensity = data.get("rd_intensity", 0)
        if rd_intensity > 15:
            score += 4
        elif rd_intensity > 10:
            score += 3
        elif rd_intensity > 5:
            score += 2

        gm_expansion = data.get("gross_margin_expansion", 0)
        if gm_expansion > 5:
            score += 2
        elif gm_expansion > 2:
            score += 1

        if data.get("operating_leverage", False):
            score += 2

        return min(score, 10)

    def _score_market_opportunity(self, data: dict) -> int:
        """Score market opportunity (0-10)."""
        score = 0
        tam = data.get("total_addressable_market", 0)
        if tam > 100:
            score += 3
        elif tam > 50:
            score += 2
        elif tam > 10:
            score += 1

        market_share = data.get("market_share", 0)
        if 0 < market_share < 10:
            score += 2

        if data.get("scalability", False):
            score += 3

        return min(score, 10)

    def _score_growth_valuation(self, price: float, fin_data: dict, growth_data: dict) -> int:
        """Score growth-adjusted valuation (0-10)."""
        score = 0
        if price <= 0:
            return 5

        eps = fin_data.get("eps", 0)
        growth_rate = growth_data.get("eps_cagr", 0)
        if eps > 0 and growth_rate > 0:
            peg = (price / eps) / growth_rate
            if peg < 1.0:
                score += 4
            elif peg < 1.5:
                score += 3
            elif peg < 2.0:
                score += 2

        revenue_cagr = growth_data.get("revenue_cagr", 0)
        if revenue_cagr > 25:
            score += 1

        return min(score, 10)

    def _analyze_basic_growth(self, growth_data: dict) -> float:
        """Simplified growth analysis when detailed data unavailable."""
        score = 50
        revenue_cagr = growth_data.get("revenue_cagr", 0)
        price_momentum = growth_data.get("price_momentum_1m", 0)
        forward_pe = growth_data.get("forward_pe", 0)

        if revenue_cagr > 30:
            score += 20
        elif revenue_cagr > 20:
            score += 15
        elif revenue_cagr > 10:
            score += 10
        elif revenue_cagr < -20:
            score -= 20
        elif revenue_cagr < -10:
            score -= 10

        if price_momentum > 10:
            score += 10
        elif price_momentum < -10:
            score -= 10

        if 0 < forward_pe < 20:
            score += 5
        elif forward_pe > 50:
            score -= 10

        return max(0, min(100, score))

    # Momentum Analysis Methods
    def _score_price_momentum(self, price_data: dict) -> int:
        """Score price momentum (0-10)."""
        score = 0
        ret_1m = self._get_return(price_data, 20)
        ret_3m = self._get_return(price_data, 60)
        ret_6m = self._get_return(price_data, 120)

        if ret_1m > 15:
            score += 3
        elif ret_1m > 8:
            score += 2
        elif ret_1m > 0:
            score += 1

        if ret_3m > 30:
            score += 3
        elif ret_3m > 15:
            score += 2
        elif ret_3m > 5:
            score += 1

        if ret_6m > 50:
            score += 2
        elif ret_6m > 25:
            score += 1

        if ret_1m > 0 and ret_3m > 0 and ret_6m > 0:
            score += 1

        return min(score, 10)

    def _score_trend_strength(self, technical_data: dict) -> int:
        """Score trend strength (0-10)."""
        score = 0
        adx = technical_data.get("adx", 0)
        if adx > 30:
            score += 3
        elif adx > 25:
            score += 2
        elif adx > 20:
            score += 1

        ma20 = technical_data.get("ma20", 0)
        ma60 = technical_data.get("ma60", 0)
        ma120 = technical_data.get("ma120", 0)
        current = technical_data.get("current_price", 0)

        if all([ma20, ma60, ma120, current]):
            if current > ma20 > ma60 > ma120:
                score += 3
            elif current > ma20 > ma60:
                score += 2
            elif current > ma20:
                score += 1

        return min(score, 10)

    def _score_volume(self, technical_data: dict) -> int:
        """Score volume patterns (0-10)."""
        score = 0
        vol_momentum = technical_data.get("volume_momentum", 0)
        if vol_momentum > 150:
            score += 3
        elif vol_momentum > 120:
            score += 2
        elif vol_momentum > 100:
            score += 1

        if technical_data.get("obv_trend") == "up":
            score += 2

        up_ratio = technical_data.get("up_volume_ratio", 0.5)
        if up_ratio > 0.6:
            score += 2
        elif up_ratio > 0.55:
            score += 1

        if technical_data.get("volume_spike", False):
            score += 2

        return min(score, 10)

    def _score_relative_strength(self, market_data: dict) -> int:
        """Score relative strength (0-10)."""
        score = 0
        rs_ratio = market_data.get("relative_strength_ratio", 1.0)
        if rs_ratio > 1.2:
            score += 3
        elif rs_ratio > 1.1:
            score += 2
        elif rs_ratio > 1.0:
            score += 1

        if market_data.get("outperforming_in_down_market", False):
            score += 3

        if market_data.get("rs_trend") == "improving":
            score += 2

        rs_rank = market_data.get("relative_strength_rank", 50)
        if rs_rank > 80:
            score += 2
        elif rs_rank > 60:
            score += 1

        return min(score, 10)

    # Helper Methods
    def _calculate_graham_number(self, data: dict, a_share_adjusted: bool = False) -> float:
        """Calculate Graham Number.

        Args:
            data: Valuation data with eps and book_value_per_share
            a_share_adjusted: If True, use A-share adjusted multiplier (18.0 vs 22.5)

        Returns:
            Graham number representing fair value estimate
        """
        eps = data.get("eps", 0)
        bvps = data.get("book_value_per_share", 0)
        if eps <= 0 or bvps <= 0:
            return 0
        multiplier = A_SHARE_GRAHAM_MULTIPLIER if a_share_adjusted else 22.5
        return math.sqrt(multiplier * eps * bvps)

    def _get_return(self, price_data: dict, days: int) -> float:
        """Calculate return over specified days."""
        try:
            if isinstance(price_data, dict) and "close" in price_data:
                closes = price_data["close"]
            else:
                return 0

            if len(closes) < days + 1:
                return 0

            current = closes[-1]
            past = closes[-(days + 1)]

            if past <= 0:
                return 0

            return (current - past) / past * 100
        except Exception:
            self._logger.debug(f"Error calculating change in {days} days")
            return 0
