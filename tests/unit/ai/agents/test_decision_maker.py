"""Tests for DecisionMakerAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.ai.agents.decision_maker import DecisionMakerAgent
from ashare_analyzer.models import SignalType

LLM_CLIENT_PATH = "ashare_analyzer.ai.clients.get_llm_client"


class TestDecisionMakerAgent:
    """Tests for DecisionMakerAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()
            assert agent.name == "DecisionMakerAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_no_analysis_reports_returns_hold(self):
        """Test analysis with no analysis reports returns HOLD with error."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [],
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "error" in result.metadata
            assert result.metadata["error"] == "missing_analysis_reports"

    @pytest.mark.asyncio
    async def test_analyze_llm_unavailable_returns_hold(self):
        """Test analysis returns HOLD when LLM is unavailable."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [{"analyst": "Test", "conclusion": "buy", "confidence": 80}],
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert result.metadata.get("error") == "llm_unavailable"

    @pytest.mark.asyncio
    async def test_analyze_with_llm_buy_decision(self):
        """Test LLM-based analysis returns BUY decision."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "decision": "buy",
                "confidence": 85,
                "reasoning": "Strong fundamentals and technical alignment",
                "position_ratio": 0.25,
                "key_considerations": ["趋势向上", "估值合理"],
                "risks_identified": ["市场波动"],
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [
                    {"analyst": "TechnicalAgent", "conclusion": "buy", "confidence": 80, "reasoning": "上升趋势"}
                ],
                "portfolio": {},
                "risk_limits": {"max_position": 0.25},
                "market_data": {"close": 1800.0},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence == 85
            assert result.metadata["decision"] == "buy"
            assert result.metadata["position_ratio"] == 0.25
            assert "key_considerations" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_with_llm_sell_decision(self):
        """Test LLM-based analysis returns SELL decision."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "decision": "sell",
                "confidence": 75,
                "reasoning": "Technical breakdown",
                "position_ratio": 1.0,
                "key_considerations": ["趋势向下"],
                "risks_identified": ["止损"],
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [
                    {"analyst": "TechnicalAgent", "conclusion": "sell", "confidence": 75, "reasoning": "下跌趋势"}
                ],
                "portfolio": {"has_position": True},
                "risk_limits": {"max_position": 0.25},
                "market_data": {"close": 1700.0},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.confidence == 75
            assert result.metadata["decision"] == "sell"

    @pytest.mark.asyncio
    async def test_analyze_with_llm_hold_decision(self):
        """Test LLM-based analysis returns HOLD decision."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "decision": "hold",
                "confidence": 50,
                "reasoning": "Mixed signals, wait for clarity",
                "position_ratio": 0,
                "key_considerations": ["信号不明确"],
                "risks_identified": ["市场不确定性"],
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [
                    {"analyst": "TechnicalAgent", "conclusion": "hold", "confidence": 50, "reasoning": "震荡"}
                ],
                "portfolio": {},
                "risk_limits": {"max_position": 0.25},
                "market_data": {},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.confidence == 50
            assert result.metadata["decision"] == "hold"

    @pytest.mark.asyncio
    async def test_analyze_respects_max_position_limit(self):
        """Test position ratio is capped by risk limit."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "decision": "buy",
                "confidence": 90,
                "reasoning": "Strong buy",
                "position_ratio": 0.5,
                "key_considerations": ["强势"],
                "risks_identified": [],
            }
        )

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [{"analyst": "TechnicalAgent", "conclusion": "buy", "confidence": 90}],
                "portfolio": {},
                "risk_limits": {"max_position": 0.10},
                "market_data": {},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.metadata["position_ratio"] == 0.10
            assert result.metadata["max_position_limit"] == 0.10

    @pytest.mark.asyncio
    async def test_analyze_llm_returns_invalid_format(self):
        """Test analysis handles invalid LLM response format."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(return_value=None)

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [{"analyst": "Test", "conclusion": "buy", "confidence": 80}],
                "portfolio": {},
                "risk_limits": {"max_position": 0.25},
                "market_data": {},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert "LLM 返回格式无效" in result.reasoning
            assert "risks_identified" in result.metadata

    @pytest.mark.asyncio
    async def test_analyze_llm_exception_returns_hold(self):
        """Test analysis handles LLM exception gracefully."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(side_effect=Exception("API Error"))

        with patch(LLM_CLIENT_PATH, return_value=mock_llm_client):
            agent = DecisionMakerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "analysis_reports": [{"analyst": "Test", "conclusion": "buy", "confidence": 80}],
                "portfolio": {},
                "risk_limits": {"max_position": 0.25},
                "market_data": {},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert "API Error" in result.reasoning
            assert "risks_identified" in result.metadata


class TestDecisionMakerHelpers:
    """Tests for DecisionMakerAgent helper methods."""

    def test_decision_to_signal_buy(self):
        """Test converting buy decision to SignalType."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            assert agent._decision_to_signal("buy") == SignalType.BUY
            assert agent._decision_to_signal("BUY") == SignalType.BUY
            assert agent._decision_to_signal("Buy") == SignalType.BUY

    def test_decision_to_signal_sell(self):
        """Test converting sell decision to SignalType."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            assert agent._decision_to_signal("sell") == SignalType.SELL
            assert agent._decision_to_signal("SELL") == SignalType.SELL
            assert agent._decision_to_signal("Sell") == SignalType.SELL

    def test_decision_to_signal_hold(self):
        """Test converting hold decision to SignalType."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            assert agent._decision_to_signal("hold") == SignalType.HOLD
            assert agent._decision_to_signal("HOLD") == SignalType.HOLD
            assert agent._decision_to_signal("Hold") == SignalType.HOLD

    def test_decision_to_signal_unknown_returns_hold(self):
        """Test unknown decision returns HOLD."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            assert agent._decision_to_signal("UNKNOWN") == SignalType.HOLD
            assert agent._decision_to_signal("") == SignalType.HOLD
            assert agent._decision_to_signal("invalid") == SignalType.HOLD

    def test_make_hold_decision(self):
        """Test creating hold decision."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            result = agent._make_hold_decision("Test reason")

            assert result["decision"] == "hold"
            assert result["confidence"] == 0
            assert result["position_ratio"] == 0
            assert result["reasoning"] == "Test reason"
            assert result["key_considerations"] == []
            assert result["risks_identified"] == ["Test reason"]

    def test_format_analysis_reports(self):
        """Test formatting analysis reports."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            reports = [
                {
                    "analyst": "TechnicalAgent",
                    "conclusion": "buy",
                    "confidence": 80,
                    "reasoning": "上升趋势",
                    "key_metrics": {"trend": "bullish"},
                },
                {
                    "analyst": "ValuationAgent",
                    "conclusion": "hold",
                    "confidence": 60,
                    "reasoning": "估值合理",
                },
            ]

            result = agent._format_analysis_reports(reports)

            assert "--- TechnicalAgent ---" in result
            assert "BUY" in result
            assert "80%" in result
            assert "上升趋势" in result
            assert "--- ValuationAgent ---" in result
            assert "HOLD" in result
            assert "60%" in result

    def test_format_analysis_reports_empty(self):
        """Test formatting empty reports."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            result = agent._format_analysis_reports([])
            assert result == ""

    def test_format_portfolio_empty(self):
        """Test formatting empty portfolio."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            result = agent._format_portfolio({})
            assert result == "无持仓信息"

    def test_format_portfolio_no_position(self):
        """Test formatting portfolio without position."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            result = agent._format_portfolio({"has_position": False})
            assert result == "当前无持仓"

    def test_format_portfolio_with_position(self):
        """Test formatting portfolio with position."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            portfolio = {
                "has_position": True,
                "position_quantity": 100,
                "position_cost_price": 1800.0,
                "current_price": 1850.0,
                "current_profit_loss_pct": 2.5,
                "total_value": 100000.0,
                "current_position_ratio": 0.18,
            }

            result = agent._format_portfolio(portfolio)

            assert "100 股" in result
            assert "1800" in result
            assert "1850" in result
            assert "2.50%" in result

    def test_format_market_data_empty(self):
        """Test formatting empty market data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            result = agent._format_market_data({})
            assert result == "无市场数据"

    def test_format_market_data_with_data(self):
        """Test formatting market data."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            market_data = {
                "close": 1800.0,
                "pct_chg": 2.5,
                "volume": 1000000,
                "amount": 1800000000,
            }

            result = agent._format_market_data(market_data)

            assert "1800" in result
            assert "2.50%" in result
            assert "1,000,000" in result

    def test_build_decision_prompt(self):
        """Test building decision prompt."""
        with patch(LLM_CLIENT_PATH, return_value=None):
            agent = DecisionMakerAgent()

            prompt = agent._build_decision_prompt(
                stock_code="600519",
                stock_name="贵州茅台",
                analysis_reports=[{"analyst": "Test", "conclusion": "buy", "confidence": 80, "reasoning": "测试"}],
                portfolio={"has_position": False},
                market_data={"close": 1800.0},
                max_position=0.25,
            )

            assert "600519" in prompt
            assert "贵州茅台" in prompt
            assert "Test" in prompt
            assert "25%" in prompt
