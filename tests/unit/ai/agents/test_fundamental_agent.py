"""Tests for FundamentalAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.ai.agents.fundamental_agent import FundamentalAgent
from ashare_analyzer.models import SignalType

LLM_CLIENT_PATH = "ashare_analyzer.ai.clients.get_llm_client"


class TestFundamentalAgent:
    """Tests for FundamentalAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()
            assert agent.name == "FundamentalAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_missing_financial_data_returns_hold(self):
        """Test analysis with missing financial data returns HOLD."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            result = await agent.analyze({"code": "600519"})

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "无财务数据" in result.reasoning

    @pytest.mark.asyncio
    async def test_analyze_heuristic_with_financial_data(self, sample_analysis_context):
        """Test heuristic analysis uses financial data correctly."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 25.0,
                "roa": 12.0,
                "net_margin": 30.0,
                "gross_margin": 50.0,
                "debt_to_equity": 0.2,
                "current_ratio": 2.5,
                "revenue_growth": 30.0,
            }

            result = await agent.analyze(context)

            # Test that heuristic fallback is used and metadata is set correctly
            assert result.confidence > 0
            assert "analysis_method" in result.metadata
            assert result.metadata["analysis_method"] == "heuristic"
            assert "total_score" in result.metadata
            assert "profitability_score" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_heuristic_weak_fundamentals_returns_sell(self, sample_analysis_context):
        """Test heuristic analysis with weak fundamentals returns SELL signal."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 2.0,  # Weak ROE < 5%
                "net_margin": 2.0,  # Weak margin < 5%
                "debt_to_equity": 1.5,  # High debt
                "current_ratio": 0.8,  # Poor liquidity
                "revenue_growth": -15.0,  # Declining revenue
                "earnings_growth": -20.0,
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_analyze_heuristic_basic_data_only(self, sample_analysis_context):
        """Test heuristic analysis with only PE/PB data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "pe_ratio": 12.0,  # Low PE
                "pb_ratio": 1.0,  # Low PB
                "volatility": 15.0,
            }

            result = await agent.analyze(context)

            assert result.signal in [SignalType.BUY, SignalType.HOLD]
            assert result.metadata["analysis_type"] == "basic"

    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self, sample_analysis_context):
        """Test LLM-based analysis returns parsed result."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "Strong fundamentals with ROE > 15%",
                "key_metrics": {"roe": 20.5},
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 20.0,
                "net_margin": 25.0,
                "debt_to_equity": 0.3,
                "current_ratio": 2.5,
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
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 20.0,
                "net_margin": 25.0,
                "debt_to_equity": 0.3,
            }

            result = await agent.analyze(context)

            # Should fallback to heuristic
            assert result.metadata.get("analysis_method") == "heuristic"

    @pytest.mark.asyncio
    async def test_analyze_llm_invalid_format_fallback(self, sample_analysis_context):
        """Test fallback when LLM returns invalid format."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(return_value={"invalid": "format"})  # No 'signal' key

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = FundamentalAgent()

            context = sample_analysis_context.copy()
            context["financial_data"] = {
                "roe": 20.0,
                "net_margin": 25.0,
            }

            result = await agent.analyze(context)

            # Should fallback to heuristic
            assert result.metadata.get("analysis_method") == "heuristic"

    def test_build_fundamental_prompt(self):
        """Test prompt building includes all metrics."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            financial_data = {
                "roe": 18.5,
                "roa": 12.3,
                "net_margin": 22.0,
                "gross_margin": 45.0,
                "debt_to_equity": 0.35,
                "current_ratio": 2.1,
                "interest_coverage": 8.5,
                "revenue_growth": 15.0,
                "earnings_growth": 20.0,
                "eps_consistency": 85.0,
                "revenue_consistency": 75.0,
            }

            prompt = agent._build_fundamental_prompt("600519", financial_data)

            assert "600519" in prompt
            assert "ROE: 18.50%" in prompt
            assert "ROA: 12.30%" in prompt
            assert "Net Margin: 22.00%" in prompt
            assert "Gross Margin: 45.00%" in prompt
            assert "Debt-to-Equity: 0.35" in prompt
            assert "Current Ratio: 2.10" in prompt
            assert "Interest Coverage: 8.50" in prompt
            assert "Revenue Growth: 15.00%" in prompt
            assert "Earnings Growth: 20.00%" in prompt
            assert "EPS Consistency: 85.0%" in prompt
            assert "Revenue Consistency: 75.0%" in prompt
            assert "analyze_signal" in prompt

    def test_build_fundamental_prompt_empty_data(self):
        """Test prompt building with empty financial data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = FundamentalAgent()

            prompt = agent._build_fundamental_prompt("600519", {})

            assert "600519" in prompt
            assert "No profitability data available" in prompt
            assert "No health metrics available" in prompt
            assert "No growth data available" in prompt
            assert "No consistency data available" in prompt
