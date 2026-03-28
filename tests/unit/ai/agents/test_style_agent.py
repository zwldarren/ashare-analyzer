"""Tests for StyleAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.ai.agents.style_agent import StyleAgent
from ashare_analyzer.models import SignalType

LLM_CLIENT_PATH = "ashare_analyzer.ai.clients.get_llm_client"


class TestStyleAgent:
    """Tests for StyleAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()
            assert agent.name == "StyleAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_heuristic_with_full_data(self, sample_analysis_context):
        """Test heuristic analysis with full data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 25.0,
                "roa": 12.0,
                "net_margin": 30.0,
                "gross_margin": 50.0,
                "debt_to_equity": 0.2,
                "revenue_growth": 25.0,
                "earnings_growth": 20.0,
            }
            context["valuation_data"] = {
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
                "dividend_yield": 3.0,
            }
            context["growth_data"] = {
                "revenue_cagr": 20.0,
                "eps_cagr": 18.0,
            }
            context["technical_data"] = {
                "adx": 25.0,
                "volume_momentum": 130,
            }
            context["price_data"] = {"close": [100, 105, 110, 115, 120]}

            result = await agent.analyze(context)

            # Test that heuristic fallback is used
            assert result.confidence > 0
            assert "analysis_method" in result.metadata
            assert result.metadata["analysis_method"] == "heuristic_fallback"
            assert "value_score" in result.metadata
            assert "growth_score" in result.metadata
            assert "momentum_score" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_heuristic_weak_all_styles(self, sample_analysis_context):
        """Test heuristic analysis with weak metrics returns appropriate signal."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 3.0,  # Weak
                "net_margin": 3.0,  # Weak
                "debt_to_equity": 2.0,  # High debt
            }
            context["valuation_data"] = {
                "pe_ratio": 80.0,  # High PE
            }
            context["growth_data"] = {}
            context["technical_data"] = {}
            context["price_data"] = {}

            result = await agent.analyze(context)

            assert result.confidence > 0
            assert "analysis_method" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self, sample_analysis_context):
        """Test LLM-based analysis returns parsed result."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "Strong value characteristics with margin of safety",
                "key_metrics": {"dominant_style": "value"},
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = StyleAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 20.0,
                "net_margin": 25.0,
            }
            context["valuation_data"] = {
                "pe_ratio": 15.0,
                "pb_ratio": 2.0,
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence == 85
            assert result.metadata["analysis_method"] == "llm"

    @pytest.mark.asyncio
    async def test_analyze_llm_fallback_to_heuristic(self, sample_analysis_context):
        """Test fallback to heuristic when LLM fails."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(return_value=None)  # LLM returns None

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = StyleAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {"roe": 20.0}
            context["valuation_data"] = {"pe_ratio": 15.0}
            context["growth_data"] = {}
            context["technical_data"] = {}
            context["price_data"] = {}

            result = await agent.analyze(context)

            # Should fallback to heuristic
            assert result.metadata.get("analysis_method") == "heuristic_fallback"

    @pytest.mark.asyncio
    async def test_analyze_llm_invalid_format_fallback(self, sample_analysis_context):
        """Test fallback when LLM returns invalid format."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(return_value={"invalid": "format"})  # No 'signal' key

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = StyleAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {"roe": 20.0}
            context["valuation_data"] = {}
            context["growth_data"] = {}
            context["technical_data"] = {}
            context["price_data"] = {}

            result = await agent.analyze(context)

            # Should fallback to heuristic
            assert result.metadata.get("analysis_method") == "heuristic_fallback"

    def test_build_style_prompt(self):
        """Test prompt building includes all style dimensions."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()

            context = {
                "financial_data": {
                    "roe": 18.5,
                    "revenue_growth": 15.0,
                    "earnings_growth": 20.0,
                    "debt_to_equity": 0.35,
                },
                "valuation_data": {
                    "pe_ratio": 15.0,
                    "pb_ratio": 2.5,
                    "margin_of_safety": 25.0,
                    "dividend_yield": 3.5,
                },
                "technical_data": {
                    "rsi_14": 55.0,
                    "macd_hist": 0.05,
                },
                "today": {
                    "close": 100.0,
                    "ma5": 95.0,
                    "pct_chg": 2.5,
                    "volume_ratio": 1.2,
                },
            }

            prompt = agent._build_style_prompt("600519", context)

            assert "600519" in prompt
            # Value attributes
            assert "PE Ratio: 15.00" in prompt
            assert "PB Ratio: 2.50" in prompt
            assert "Margin of Safety: 25.0%" in prompt
            assert "Dividend Yield: 3.50%" in prompt
            assert "Debt-to-Equity: 0.35" in prompt
            # Growth attributes
            assert "Revenue Growth: 15.00%" in prompt
            assert "EPS Growth: 20.00%" in prompt
            assert "ROE: 18.50%" in prompt
            # Momentum attributes
            assert "Price vs MA5:" in prompt
            assert "Price Change: 2.50%" in prompt
            assert "RSI(14): 55.0" in prompt
            assert "analyze_signal" in prompt

    def test_build_style_prompt_empty_data(self):
        """Test prompt building with empty data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = StyleAgent()

            prompt = agent._build_style_prompt("600519", {})

            assert "600519" in prompt
            assert "Limited value data available" in prompt
            assert "Limited growth data available" in prompt
            assert "Limited momentum data available" in prompt
