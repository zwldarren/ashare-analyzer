"""Tests for TechnicalAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.ai.agents.technical_agent import TechnicalAgent
from ashare_analyzer.models import SignalType

LLM_CLIENT_PATH = "ashare_analyzer.ai.clients.get_llm_client"


class TestTechnicalAgent:
    """Tests for TechnicalAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()
            assert agent.name == "TechnicalAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_bullish_trend_returns_buy(self, sample_analysis_context):
        """Test bullish trend analysis returns BUY signal."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()

            context = sample_analysis_context.copy()
            context["today"] = context["today"].copy()
            context["today"]["ma5"] = 1795.0
            context["today"]["ma10"] = 1780.0
            context["today"]["ma20"] = 1770.0
            context["today"]["close"] = 1800.0
            context["ma_status"] = "多头排列"

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence > 0
            assert "多头排列" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_bearish_trend_returns_sell(self, sample_analysis_context):
        """Test bearish trend analysis returns SELL signal."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()

            context = sample_analysis_context.copy()
            context["today"] = context["today"].copy()
            context["today"]["close"] = 1760.0
            context["today"]["ma5"] = 1770.0
            context["today"]["ma10"] = 1780.0
            context["today"]["ma20"] = 1790.0
            context["ma_status"] = "空头排列"

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert "空头" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_missing_price_data_returns_hold(self):
        """Test analysis with missing price data returns HOLD."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()

            result = await agent.analyze(
                {
                    "code": "600519",
                    "today": {"close": 0, "ma5": 0},
                    "ma_status": "unknown",
                }
            )

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0

    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self, sample_analysis_context, mock_acompletion):
        """Test LLM-based analysis returns parsed result."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "Strong bullish trend",
                "trend_assessment": "bullish",
                "trend_strength": 80,
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = TechnicalAgent()

            result = await agent.analyze(sample_analysis_context)

            assert result.signal == SignalType.BUY
            assert result.confidence == 85
            assert "trend_assessment" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_exception_returns_hold(self):
        """Test exception during analysis returns HOLD."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = TechnicalAgent()

            result = await agent.analyze({"code": "600519", "today": None})

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "error" in result.metadata


class TestIndicatorInterpretation:
    """Tests for indicator interpretation functions."""

    def test_interpret_volume(self):
        """Test volume interpretation."""
        from ashare_analyzer.analysis.indicators import interpret_volume

        assert interpret_volume(2.5) == "显著放量"
        assert interpret_volume(1.0) == "量能正常"
        assert interpret_volume(0.3) == "明显缩量"

    def test_interpret_rsi_cn(self):
        """Test Chinese RSI interpretation."""
        from ashare_analyzer.analysis.indicators import interpret_rsi_cn

        assert interpret_rsi_cn(75) == "超买"
        assert interpret_rsi_cn(15) == "严重超卖"

    def test_interpret_adx_cn(self):
        """Test Chinese ADX interpretation."""
        from ashare_analyzer.analysis.indicators import interpret_adx_cn

        assert interpret_adx_cn(45) == "很强趋势"
        assert interpret_adx_cn(15) == "无趋势"
