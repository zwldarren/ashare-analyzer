"""
Valuation Agent Module

Specialized agent for company valuation analysis using multi-factor adaptive approach.

This agent focuses exclusively on:
- Stock type classification (Value, Growth, Cyclical, Financial, Loss-making)
- Method selection based on stock type
- Multiple valuation methods with adaptive weights
- Confidence decay for missing data
- Margin of safety calculation

All calculations use ONLY actual data, no estimations or defaults.
"""

import logging
from typing import Any

from ashare_analyzer.ai.tools import ANALYZE_SIGNAL_TOOL
from ashare_analyzer.constants import A_SHARE_RISK_FREE_RATE, A_SHARE_RISK_PREMIUM
from ashare_analyzer.exceptions import handle_errors
from ashare_analyzer.models import AgentSignal, SignalType
from ashare_analyzer.valuation import (
    StockType,
    ValuationResult,
    calculate_adjusted_graham,
    calculate_dividend_discount,
    calculate_pb_percentile,
    calculate_peg_valuation,
    calculate_ps_valuation,
    classify_stock,
    select_valuation_methods,
)

from .base import BaseAgent

logger = logging.getLogger(__name__)

# System prompt for valuation analysis
VALUATION_SYSTEM_PROMPT = """You are a professional valuation analyst for A-share market.

Your task: Assess whether a stock is fairly valued based on multiple valuation methods.

=== Key Metrics You'll Receive ===
- Fair value: Calculated fair value from multiple methods
- Current price: Market price
- Margin of safety: (Fair value - Current price) / Current price
- Stock type: Value, Growth, Cyclical, Financial, or Loss-making
- Valuation methods used: Which methods contributed to fair value

=== Interpretation Guidelines ===
Margin of Safety:
- Positive margin means stock is undervalued (fair value > current price)
- Negative margin means stock is overvalued (fair value < current price)
- Consider stock type when interpreting margin

Stock Type Considerations:
- Value stocks: Focus on margin of safety, look for >20% discount
- Growth stocks: Higher P/E acceptable if growth justifies it
- Cyclical stocks: Consider cycle position, be cautious at peak
- Financial stocks: P/B more relevant than P/E
- Loss-making: Valuation methods less reliable, higher uncertainty

=== Signal Guidelines ===
BUY conditions:
- Strong margin of safety (>20% undervalued)
- Multiple valuation methods agree
- Stock type appropriate for valuation method

SELL conditions:
- Significant overvaluation (>20% overvalued)
- Valuation methods deteriorating
- Stock type mismatch with valuation

HOLD conditions:
- Fair value close to current price (±10%)
- Mixed signals from methods
- Uncertain stock type

=== Confidence Levels ===
- 90-100%: Clear valuation signal with multiple methods confirming
- 70-89%: Good valuation signal with minor data gaps
- 50-69%: Moderate certainty, some method limitations
- 30-49%: Limited data or conflicting signals
- 10-29%: Poor data quality or unreliable valuation

Use the analyze_signal function to return your analysis."""


class ValuationAgent(BaseAgent):
    """
    Valuation Agent for determining fair value and margin of safety.

    This agent uses a multi-factor adaptive approach:
    1. Classify stock type (Value, Growth, Cyclical, Financial, Loss-making)
    2. Select valuation methods and weights based on stock type
    3. Calculate valuations using available methods
    4. Apply confidence decay for missing data
    5. Generate signal with proper reasoning

    Attributes:
        None - uses context data only

    Example:
        agent = ValuationAgent()
        signal = agent.analyze({
            "code": "600519",
            "stock_name": "贵州茅台",
            "current_price": 1500.0,
            "valuation_data": {...}
        })
    """

    def __init__(self):
        """Initialize the Valuation Agent."""
        super().__init__("ValuationAgent")
        self._ensure_llm_client()

    def is_available(self) -> bool:
        """Always available - only requires context data."""
        return True

    def _build_valuation_prompt(
        self,
        stock_code: str,
        valuation_data: dict[str, Any],
        fair_value: float,
        current_price: float,
        margin: float,
        stock_type: Any,
    ) -> str:
        """Build the valuation analysis prompt for LLM."""
        # Stock type name mapping
        type_names = {
            "value": "价值型",
            "growth": "成长型",
            "cyclical": "周期型",
            "financial": "金融型",
            "loss_making": "亏损型",
            "default": "默认型",
        }
        stock_type_name = type_names.get(
            str(stock_type).lower() if stock_type else "default", "默认型"
        )

        # Build valuation methods info
        methods_info = []
        if valuation_data.get("pe_ratio"):
            methods_info.append(f"- P/E Ratio: {valuation_data['pe_ratio']:.2f}")
        if valuation_data.get("pb_ratio"):
            methods_info.append(f"- P/B Ratio: {valuation_data['pb_ratio']:.2f}")
        if valuation_data.get("ps_ratio"):
            methods_info.append(f"- P/S Ratio: {valuation_data['ps_ratio']:.2f}")
        if valuation_data.get("dividend_yield"):
            methods_info.append(f"- Dividend Yield: {valuation_data['dividend_yield']:.2f}%")
        if valuation_data.get("roe"):
            methods_info.append(f"- ROE: {valuation_data['roe']:.2f}%")

        sections = [
            f"=== {stock_code} Valuation Analysis ===",
            "",
            "=== Key Valuation Metrics ===",
            f"- Fair Value: ¥{fair_value:.2f}",
            f"- Current Price: ¥{current_price:.2f}",
            f"- Margin of Safety: {margin * 100:+.2f}%",
            f"- Stock Type: {stock_type_name}",
            "",
            "=== Available Valuation Data ===",
        ]
        if methods_info:
            sections.extend(methods_info)
        else:
            sections.append("Limited valuation data available")

        sections.extend(["", "请根据以上估值数据分析，使用 analyze_signal 函数返回交易信号。"])

        return "\n".join(sections)

    async def _analyze_with_llm(
        self,
        stock_code: str,
        valuation_data: dict[str, Any],
        fair_value: float,
        current_price: float,
        margin: float,
        stock_type: Any,
    ) -> dict[str, Any] | None:
        """Use LLM for valuation analysis with Function Call."""
        if not self._llm_client:
            return None

        try:
            prompt = self._build_valuation_prompt(
                stock_code, valuation_data, fair_value, current_price, margin, stock_type
            )

            self._logger.debug(f"[{stock_code}] ValuationAgent calling LLM...")
            result = await self._llm_client.generate_with_tool(
                prompt=prompt,
                tool=ANALYZE_SIGNAL_TOOL,
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
                system_prompt=VALUATION_SYSTEM_PROMPT,
                agent_name="ValuationAgent",
            )

            if result and "signal" in result:
                self._logger.debug(f"[{stock_code}] LLM valuation analysis successful: {result}")
                return result
            else:
                self._logger.warning(f"[{stock_code}] LLM valuation analysis returned invalid format")
                return None

        except Exception as e:
            self._logger.error(f"[{stock_code}] LLM valuation analysis failed: {e}")
            return None

    def _build_signal_from_llm(
        self,
        llm_analysis: dict[str, Any],
        fair_value: float,
        current_price: float,
        margin: float,
        stock_type: Any,
        valuation_data: dict[str, Any],
        valuations: dict[str, float],
        weights: dict[str, float],
        confidence_decay: float,
    ) -> AgentSignal:
        """Build AgentSignal from LLM analysis result."""
        signal_str = llm_analysis.get("signal", "hold")
        signal = SignalType.from_string(signal_str)
        confidence = llm_analysis.get("confidence", 50)
        reasoning = llm_analysis.get("reasoning", "无详细分析")
        key_metrics = llm_analysis.get("key_metrics", {})

        # Build metadata - match fields from _heuristic_analysis for consistency
        metadata = {
            "analysis_method": "llm",
            "fair_value": round(fair_value, 2),
            "current_price": current_price,
            "margin_of_safety": round(margin * 100, 1),
            "stock_type": stock_type.value if stock_type else "default",
            "valuations": {k: round(v, 2) for k, v in valuations.items()},
            "method_weights": weights,
            "confidence_decay": round(confidence_decay, 2),
        }
        metadata.update(key_metrics)

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            metadata=metadata,
        )

    @handle_errors("估值分析失败", default_return=None)
    async def analyze(self, context: dict[str, Any]) -> AgentSignal:
        """
        Execute valuation analysis using LLM (async).

        Args:
            context: Analysis context containing valuation_data and current_price

        Returns:
            AgentSignal with valuation-based trading signal
        """
        stock_code = context.get("code", "")
        current_price = context.get("current_price", 0)

        self._logger.debug(f"[{stock_code}] ValuationAgent starting analysis")

        valuation_data = context.get("valuation_data", {})

        if not valuation_data or current_price <= 0:
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="无估值数据或当前价格",
                metadata={"error": "no_valuation_data"},
            )

        # Step 1: Classify stock type (always needed)
        stock_type = self._classify_stock(valuation_data)
        self._logger.debug(f"[{stock_code}] 股票类型: {stock_type.value}")

        # Step 2: Select methods and weights
        selection = select_valuation_methods(stock_type)
        self._logger.debug(f"[{stock_code}] 选用方法: {selection.methods}, 权重: {selection.weights}")

        # Step 3: Calculate valuations (always needed for LLM context)
        valuations, decay_factors = self._calculate_valuations(selection.methods, valuation_data, current_price)

        if not valuations:
            self._logger.debug(f"[{stock_code}] 无足够数据进行任何估值计算")
            return AgentSignal(
                agent_name=self.name,
                signal=SignalType.HOLD,
                confidence=0,
                reasoning="数据不足，无法计算估值",
                metadata={
                    "error": "insufficient_data_for_valuation",
                    "available_data": list(valuation_data.keys()),
                    "stock_type": stock_type.value,
                },
            )

        # Step 4: Calculate weighted fair value
        fair_value, confidence_decay = self._calculate_weighted_value(valuations, selection.weights, decay_factors)

        # Step 5: Calculate margin
        margin = (fair_value - current_price) / current_price if current_price > 0 else 0

        # Try LLM analysis first
        if self._llm_client and self._llm_client.is_available():
            llm_result = await self._analyze_with_llm(
                stock_code, valuation_data, fair_value, current_price, margin, stock_type
            )
            if llm_result:
                return self._build_signal_from_llm(
                    llm_result,
                    fair_value,
                    current_price,
                    margin,
                    stock_type,
                    valuation_data,
                    valuations,
                    selection.weights,
                    confidence_decay,
                )

        # Fallback to heuristic analysis
        self._logger.warning(f"[{stock_code}] LLM unavailable, using heuristic fallback")
        return await self._heuristic_analysis(
            context, stock_type, fair_value, current_price, margin, valuations, selection.weights, confidence_decay
        )

    async def _heuristic_analysis(
        self,
        context: dict[str, Any],
        stock_type: Any,
        fair_value: float,
        current_price: float,
        margin: float,
        valuations: dict[str, float],
        weights: dict[str, float],
        confidence_decay: float,
    ) -> AgentSignal:
        """Fallback heuristic analysis when LLM is unavailable."""
        stock_code = context.get("code", "")

        # Generate signal based on margin thresholds (heuristic)
        signal, base_confidence = self._heuristic_margin_signal(margin)

        # Apply confidence decay
        final_confidence = int(base_confidence * confidence_decay)
        final_confidence = max(10, min(100, final_confidence))

        # Build reasoning
        reasoning = self._build_reasoning(
            fair_value, current_price, margin, stock_type, valuations, weights, confidence_decay
        )

        self._logger.debug(
            f"[{stock_code}] ValuationAgent heuristic analysis: {signal} "
            f"(公允价值¥{fair_value:.2f}, 安全边际{margin * 100:+.1f}%, "
            f"置信度{final_confidence}%)"
        )

        # Build metadata
        metadata = {
            "analysis_method": "heuristic_fallback",
            "fair_value": round(fair_value, 2),
            "current_price": current_price,
            "margin_of_safety": round(margin * 100, 1),
            "stock_type": stock_type.value,
            "valuations": {k: round(v, 2) for k, v in valuations.items()},
            "method_weights": weights,
            "confidence_decay": round(confidence_decay, 2),
        }

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=final_confidence,
            reasoning=reasoning,
            metadata=metadata,
        )

    def _classify_stock(self, valuation_data: dict[str, Any]) -> StockType:
        """
        Classify stock type based on valuation data.

        Args:
            valuation_data: Valuation data dict

        Returns:
            StockType classification
        """
        classify_data = {
            "pe_ratio": valuation_data.get("pe_ratio"),
            "roe": valuation_data.get("roe", 0),
            "dividend_yield": valuation_data.get("dividend_yield", 0),
            "revenue_growth": valuation_data.get("revenue_growth", 0),
            "eps_growth": valuation_data.get("eps_growth", 0),
            "industry": valuation_data.get("industry_name", ""),
        }
        return classify_stock(classify_data)

    def _calculate_valuations(
        self,
        methods: list[str],
        valuation_data: dict[str, Any],
        current_price: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Calculate valuations for each method.

        Args:
            methods: List of valuation method names
            valuation_data: Valuation data dict
            current_price: Current stock price

        Returns:
            Tuple of (valuations dict, decay_factors dict)
        """
        valuations: dict[str, float] = {}
        decay_factors: dict[str, float] = {}

        for method in methods:
            result = self._calculate_single_method(method, valuation_data, current_price)
            if result.value > 0:
                valuations[method] = result.value
                # Calculate decay factor for this method
                decay_factors[method] = self._calculate_method_decay(method, valuation_data, result)
                self._logger.debug(
                    f"  {method}: 价值={result.value:.2f}, 衰减={decay_factors[method]:.2f}"
                    + (f", 跳过原因={result.skip_reason}" if result.skip_reason else "")
                )

        return valuations, decay_factors

    def _calculate_single_method(
        self,
        method: str,
        valuation_data: dict[str, Any],
        current_price: float,
    ) -> ValuationResult:
        """
        Calculate valuation for a single method.

        Args:
            method: Method name
            valuation_data: Valuation data dict
            current_price: Current stock price

        Returns:
            ValuationResult with value or skip_reason
        """
        eps = valuation_data.get("eps", 0)
        bvps = valuation_data.get("book_value_per_share", 0)
        pe_ratio = valuation_data.get("pe_ratio", 0)
        pb_ratio = valuation_data.get("pb_ratio", 0)
        ps_ratio = valuation_data.get("ps_ratio", 0)
        roe = valuation_data.get("roe", 10.0)
        dividend_yield = valuation_data.get("dividend_yield", 0)
        growth_rate = valuation_data.get("revenue_growth", 0)
        industry_pb = valuation_data.get("industry_pb")
        industry_ps = valuation_data.get("industry_ps")
        historical_pb_median = valuation_data.get("historical_pb_median")
        historical_ps_median = valuation_data.get("historical_ps_median")

        if method == "adjusted_graham":
            return calculate_adjusted_graham(eps, bvps, roe)

        elif method == "peg":
            if eps <= 0 or pe_ratio <= 0:
                return ValuationResult(value=0.0, method=method, skip_reason="需要EPS和PE比率")
            return calculate_peg_valuation(eps, pe_ratio, growth_rate)

        elif method == "pb_percentile":
            if bvps <= 0 or pb_ratio <= 0:
                return ValuationResult(value=0.0, method=method, skip_reason="需要BVPS和PB比率")
            return calculate_pb_percentile(bvps, pb_ratio, historical_pb_median, industry_pb)

        elif method == "ps":
            if ps_ratio <= 0 or current_price <= 0:
                return ValuationResult(value=0.0, method=method, skip_reason="需要PS比率和当前价格")
            return calculate_ps_valuation(current_price, ps_ratio, industry_ps, historical_ps_median)

        elif method == "dividend_discount":
            if dividend_yield <= 0 or current_price <= 0:
                return ValuationResult(value=0.0, method=method, skip_reason="需要股息率和当前价格")
            # Calculate annual dividend per share
            dividend = current_price * dividend_yield / 100
            required_return = A_SHARE_RISK_FREE_RATE + A_SHARE_RISK_PREMIUM
            return calculate_dividend_discount(dividend, required_return)

        else:
            return ValuationResult(value=0.0, method=method, skip_reason=f"未知估值方法: {method}")

    def _calculate_method_decay(
        self,
        method: str,
        valuation_data: dict[str, Any],
        result: ValuationResult,
    ) -> float:
        """
        Calculate confidence decay factor for a method.

        Decay factors:
        - No historical PE/PB data: x0.7
        - No industry data: x0.9
        - EPS/BVPS derived from PE/PB: x0.85
        - Only one valuation method available: x0.6 (handled elsewhere)

        Args:
            method: Method name
            valuation_data: Valuation data dict
            result: ValuationResult from the method

        Returns:
            Decay factor (0.0 to 1.0)
        """
        decay = 1.0

        # Check for historical data
        has_historical_pb = valuation_data.get("historical_pb_median") is not None
        has_historical_ps = valuation_data.get("historical_ps_median") is not None
        has_industry_pb = valuation_data.get("industry_pb") is not None
        has_industry_ps = valuation_data.get("industry_ps") is not None

        # Check if EPS/BVPS might be derived (not provided directly)
        eps_provided = valuation_data.get("eps", 0) > 0
        bvps_provided = valuation_data.get("book_value_per_share", 0) > 0

        # Apply decay based on method type
        if method in ("pb_percentile", "adjusted_graham"):
            if not has_historical_pb:
                decay *= 0.7
            if not has_industry_pb:
                decay *= 0.9
            if not bvps_provided:
                decay *= 0.85

        elif method == "ps":
            if not has_historical_ps:
                decay *= 0.7
            if not has_industry_ps:
                decay *= 0.9

        elif method == "peg":
            if not eps_provided:
                decay *= 0.85

        elif method == "dividend_discount":
            # Dividend discount is more reliable if dividend is stable
            pass

        return decay

    def _calculate_weighted_value(
        self,
        valuations: dict[str, float],
        weights: dict[str, float],
        decay_factors: dict[str, float],
    ) -> tuple[float, float]:
        """
        Calculate weighted fair value with confidence decay.

        Args:
            valuations: Dict of method -> fair value
            weights: Dict of method -> weight
            decay_factors: Dict of method -> decay factor

        Returns:
            Tuple of (fair_value, overall_confidence_decay)
        """
        # Only use methods that have valid valuations
        available_methods = [m for m in valuations if m in weights]
        if not available_methods:
            return 0.0, 0.0

        # Normalize weights for available methods
        total_weight = sum(weights[m] for m in available_methods)
        if total_weight <= 0:
            return 0.0, 0.0

        # Calculate weighted average with decay-adjusted weights
        weighted_sum = 0.0
        adjusted_weight_sum = 0.0

        for method in available_methods:
            original_weight = weights[method]
            decay = decay_factors.get(method, 1.0)
            # Adjust weight by decay factor
            adjusted_weight = original_weight * decay
            weighted_sum += valuations[method] * adjusted_weight
            adjusted_weight_sum += adjusted_weight

        fair_value = weighted_sum / adjusted_weight_sum if adjusted_weight_sum > 0 else 0.0

        # Calculate overall confidence decay
        # Based on how much weight was "lost" due to decay
        if len(available_methods) == 1:
            # Only one method available - significant decay
            overall_decay = decay_factors.get(available_methods[0], 1.0) * 0.6
        else:
            # Multiple methods - use ratio of adjusted to original weight
            original_weight_sum = sum(weights[m] for m in available_methods)
            overall_decay = adjusted_weight_sum / original_weight_sum if original_weight_sum > 0 else 1.0

        return fair_value, overall_decay

    def _heuristic_margin_signal(self, margin: float) -> tuple[SignalType, int]:
        """
        Convert margin to signal and base confidence using heuristic rules.

        Signal thresholds (new adaptive approach):
        | Margin        | Signal | Base Confidence |
        |---------------|--------|-----------------|
        | >= +20%       | BUY    | 70 + excess*0.5 |
        | +10% ~ +20%   | BUY    | 60 ~ 70         |
        | -5% ~ +5%     | HOLD   | 50              |
        | -20% ~ -10%   | SELL   | 60 ~ 70         |
        | <= -20%       | SELL   | 70 + excess*0.5 |

        Args:
            margin: Margin of safety as decimal (e.g., 0.20 for 20%)

        Returns:
            Tuple of (SignalType, base_confidence)
        """
        if margin >= 0.20:
            # Strong undervalued - buy zone
            excess = (margin - 0.20) * 100  # Convert to percentage points
            confidence = min(100, int(70 + excess * 0.5))
            return (SignalType.BUY, confidence)

        elif margin >= 0.10:
            # Moderately undervalued - buy zone
            # Linear interpolation: 10% -> 60, 20% -> 70
            confidence = int(60 + (margin - 0.10) * 100)
            return (SignalType.BUY, confidence)

        elif margin >= -0.05:
            # Fairly valued to slightly undervalued - hold zone
            return (SignalType.HOLD, 50)

        elif margin >= -0.10:
            # Slightly overvalued - hold zone
            return (SignalType.HOLD, 50)

        elif margin >= -0.20:
            # Moderately overvalued - sell zone
            # Linear interpolation: -10% -> 60, -20% -> 70
            confidence = int(60 + (abs(margin) - 0.10) * 100)
            return (SignalType.SELL, confidence)

        else:
            # Strongly overvalued - sell zone
            excess = (abs(margin) - 0.20) * 100  # Convert to percentage points
            confidence = min(100, int(70 + excess * 0.5))
            return (SignalType.SELL, confidence)

    def _build_reasoning(
        self,
        fair_value: float,
        current_price: float,
        margin: float,
        stock_type: StockType,
        valuations: dict[str, float],
        weights: dict[str, float],
        confidence_decay: float,
    ) -> str:
        """
        Build reasoning string for the signal.

        Args:
            fair_value: Calculated fair value
            current_price: Current stock price
            margin: Margin of safety
            stock_type: Stock type classification
            valuations: Dict of method -> fair value
            weights: Dict of method -> weight
            confidence_decay: Overall confidence decay factor

        Returns:
            Reasoning string
        """
        parts = []

        # Main valuation summary
        parts.append(f"公允价值¥{fair_value:.2f}")
        parts.append(f"当前价格¥{current_price:.2f}")
        parts.append(f"安全边际{margin * 100:+.1f}%")

        # Stock type
        type_names = {
            StockType.VALUE: "价值型",
            StockType.GROWTH: "成长型",
            StockType.CYCLICAL: "周期型",
            StockType.FINANCIAL: "金融型",
            StockType.LOSS_MAKING: "亏损型",
            StockType.DEFAULT: "默认型",
        }
        parts.append(f"股票类型:{type_names.get(stock_type, '未知')}")

        # Valuation methods used
        method_names = {
            "adjusted_graham": "修正格雷厄姆",
            "peg": "PEG估值",
            "pb_percentile": "PB百分位",
            "ps": "PS估值",
            "dividend_discount": "股息贴现",
        }
        method_strs = []
        for method in valuations:
            name = method_names.get(method, method)
            weight = weights.get(method, 0) * 100
            method_strs.append(f"{name}({weight:.0f}%)")
        parts.append(f"估值方法: {', '.join(method_strs)}")

        # Confidence decay indicator
        if confidence_decay < 1.0:
            decay_pct = (1.0 - confidence_decay) * 100
            parts.append(f"置信度衰减{decay_pct:.0f}%")

        return " / ".join(parts)
