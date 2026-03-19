"""Tests for PortfolioManagerAgent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.ai.agents.portfolio_manager import PortfolioManagerAgent
from ashare_analyzer.models import SignalType


class TestPortfolioManagerAgent:
    """Tests for PortfolioManagerAgent."""

    def test_init(self):
        """Test agent initialization."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()
            assert agent.name == "PortfolioManagerAgent"

    def test_is_available_returns_true(self):
        """Test is_available always returns True."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()
            assert agent.is_available() is True

    @pytest.mark.asyncio
    async def test_analyze_no_agent_signals_returns_hold(self):
        """Test analysis with no agent signals returns HOLD with error."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {},  # Empty signals
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 50},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "error" in result.metadata
            assert result.metadata["error"] == "missing_agent_signals"

    @pytest.mark.asyncio
    async def test_analyze_buy_decision_rule_based(self):
        """Test BUY decision when weighted_score >= 30."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 80},
                    "FundamentalAgent": {"signal": "buy", "confidence": 70},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 50, "risk_flags": []},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence > 0
            assert "买入" in result.reasoning
            assert result.metadata["action"] == "BUY"
            assert result.metadata["position_ratio"] > 0

    @pytest.mark.asyncio
    async def test_analyze_sell_decision_rule_based(self):
        """Test SELL decision when weighted_score <= -30."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 80},
                    "FundamentalAgent": {"signal": "sell", "confidence": 70},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},  # Allow full exit
                "consensus_data": {"weighted_score": -50, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 100,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.confidence > 0
            assert "卖出" in result.reasoning or "看空" in result.reasoning
            assert result.metadata["action"] == "SELL"
            assert result.metadata["position_ratio"] == 1.0  # Full exit

    @pytest.mark.asyncio
    async def test_analyze_hold_decision_rule_based(self):
        """Test HOLD decision when -30 < weighted_score < 30."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "hold", "confidence": 50},
                    "FundamentalAgent": {"signal": "hold", "confidence": 50},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 0, "risk_flags": []},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0  # abs(0) = 0
            assert "观望" in result.reasoning
            assert result.metadata["action"] == "HOLD"
            assert result.metadata["position_ratio"] == 0.0

    @pytest.mark.asyncio
    async def test_analyze_respects_max_position_limit(self):
        """Test position ratio is capped by risk manager limit."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # High confidence buy would normally give high position
            # But risk manager limits to 10%
            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 90},
                    "FundamentalAgent": {"signal": "buy", "confidence": 90},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.10}},  # 10% limit
                "consensus_data": {"weighted_score": 90, "risk_flags": []},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            # Position should be capped at 10% (0.10)
            assert result.metadata["position_ratio"] <= 0.10
            assert result.metadata["max_position_limit"] == 0.10

    @pytest.mark.asyncio
    async def test_analyze_with_llm_success(self):
        """Test LLM-based analysis returns parsed result."""
        # Create a mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "Strong fundamentals and technical alignment",
                "position_ratio": 0.5,
                "key_factors": ["趋势向上", "估值合理"],
                "risk_level": "medium",
            }
        )

        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=mock_llm_client):
            agent = PortfolioManagerAgent()

            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 80, "reasoning": "上升趋势"},
                    "FundamentalAgent": {"signal": "buy", "confidence": 70, "reasoning": "估值合理"},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 50, "consensus_level": 0.8, "risk_flags": []},
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.confidence == 85
            assert "key_factors" in result.metadata
            assert result.metadata["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_analyze_with_risk_flags_reduces_confidence(self):
        """Test that risk flags reduce confidence in rule-based decision."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context_no_risk = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 80},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 50, "risk_flags": []},
            }

            context_with_risk = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 80},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 50, "risk_flags": ["flag1", "flag2"]},
            }

            result_no_risk = await agent.analyze(context_no_risk)
            result_with_risk = await agent.analyze(context_with_risk)

            # Confidence should be reduced with risk flags
            assert result_with_risk.confidence < result_no_risk.confidence
            assert "风险" in result_with_risk.reasoning

    @pytest.mark.asyncio
    async def test_analyze_exception_returns_hold(self):
        """Test exception during analysis returns HOLD."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # Context that will cause exception (missing required keys)
            result = await agent.analyze({"code": "600519"})

            assert result.signal == SignalType.HOLD
            assert result.confidence == 0
            assert "error" in result.metadata


class TestPortfolioManagerHelpers:
    """Tests for PortfolioManagerAgent helper methods."""

    def test_get_max_position_limit_from_signal(self):
        """Test extracting max position limit from risk manager signal."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            risk_signal = {"metadata": {"max_position_size": 0.15}}
            result = agent._get_max_position_limit(risk_signal)

            assert result == 0.15

    def test_get_max_position_limit_default(self):
        """Test default max position limit when no signal provided."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # No signal
            result = agent._get_max_position_limit({})
            assert result == 0.25  # Default conservative limit

            # Empty signal (None would fail type check, but method handles falsy values)
            result = agent._get_max_position_limit({})  # Already tested above, but verifies falsy handling
            assert result == 0.25

            # Signal without metadata
            result = agent._get_max_position_limit({"other_key": "value"})
            assert result == 0.25

            # Signal with empty metadata
            result = agent._get_max_position_limit({"metadata": {}})
            assert result == 0.25

    def test_action_to_signal_buy(self):
        """Test converting BUY action to SignalType."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            assert agent._action_to_signal("BUY") == SignalType.BUY
            assert agent._action_to_signal("buy") == SignalType.BUY
            assert agent._action_to_signal("Buy") == SignalType.BUY

    def test_action_to_signal_sell(self):
        """Test converting SELL action to SignalType."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            assert agent._action_to_signal("SELL") == SignalType.SELL
            assert agent._action_to_signal("sell") == SignalType.SELL
            assert agent._action_to_signal("Sell") == SignalType.SELL

    def test_action_to_signal_hold(self):
        """Test converting HOLD action to SignalType."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            assert agent._action_to_signal("HOLD") == SignalType.HOLD
            assert agent._action_to_signal("hold") == SignalType.HOLD
            assert agent._action_to_signal("Hold") == SignalType.HOLD

    def test_action_to_signal_unknown_returns_hold(self):
        """Test unknown action returns HOLD."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            assert agent._action_to_signal("UNKNOWN") == SignalType.HOLD
            assert agent._action_to_signal("") == SignalType.HOLD
            assert agent._action_to_signal("invalid") == SignalType.HOLD

    def test_summarize_signals(self):
        """Test summarizing agent signals."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            agent_signals = {
                "TechnicalAgent": {"signal": "buy", "confidence": 80},
                "FundamentalAgent": {"signal": "hold", "confidence": 50},
                "SentimentAgent": {"signal": "sell", "confidence": 30},
            }

            result = agent._summarize_signals(agent_signals)

            assert result == {
                "TechnicalAgent": "buy(80)",
                "FundamentalAgent": "hold(50)",
                "SentimentAgent": "sell(30)",
            }

    def test_summarize_signals_empty(self):
        """Test summarizing empty signals."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            result = agent._summarize_signals({})

            assert result == {}

    def test_summarize_signals_missing_fields(self):
        """Test summarizing signals with missing fields."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            agent_signals = {
                "Agent1": {},  # Missing signal and confidence
                "Agent2": {"signal": "buy"},  # Missing confidence
                "Agent3": {"confidence": 70},  # Missing signal
            }

            result = agent._summarize_signals(agent_signals)

            assert result == {
                "Agent1": "unknown(0)",
                "Agent2": "buy(0)",
                "Agent3": "unknown(70)",
            }

    def test_make_rule_based_decision_buy_boundary(self):
        """Test rule-based decision at BUY boundary (weighted_score = 30)."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "buy", "confidence": 50}},
                consensus_data={"weighted_score": 30, "risk_flags": []},
                max_position=0.25,
            )

            assert result["action"] == "BUY"
            assert result["confidence"] == 30  # abs(30)

    def test_make_rule_based_decision_sell_boundary(self):
        """Test rule-based decision at SELL boundary (weighted_score = -30)."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "sell", "confidence": 50}},
                consensus_data={"weighted_score": -30, "risk_flags": []},
                max_position=0.25,
            )

            assert result["action"] == "SELL"
            assert result["confidence"] == 30  # abs(-30)

    def test_make_rule_based_decision_hold_range(self):
        """Test rule-based decision in HOLD range (-29 to 29)."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # Test at 0
            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "hold", "confidence": 50}},
                consensus_data={"weighted_score": 0, "risk_flags": []},
                max_position=0.25,
            )
            assert result["action"] == "HOLD"
            assert result["position_ratio"] == 0.0

            # Test at 29
            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "hold", "confidence": 50}},
                consensus_data={"weighted_score": 29, "risk_flags": []},
                max_position=0.25,
            )
            assert result["action"] == "HOLD"

            # Test at -29
            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "hold", "confidence": 50}},
                consensus_data={"weighted_score": -29, "risk_flags": []},
                max_position=0.25,
            )
            assert result["action"] == "HOLD"


class TestTradeQuantityCalculation:
    """Tests for trade quantity calculation in PortfolioManagerAgent."""

    @pytest.mark.asyncio
    async def test_sell_high_confidence_closes_position(self):
        """Test SELL with high confidence (>=70) closes entire position."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 80},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},
                "consensus_data": {"weighted_score": -80, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 400,
                    "position_cost_price": 23.85,
                    "current_price": 23.80,
                    "total_value": 100000,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.metadata["trade_quantity"] == 400  # All 400 shares
            assert result.metadata["position_action"] == "close_position"
            assert "清仓" in result.reasoning

    @pytest.mark.asyncio
    async def test_sell_medium_confidence_reduces_half(self):
        """Test SELL with medium confidence (50-69) sells 50% of position."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 60},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},
                "consensus_data": {"weighted_score": -60, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 400,
                    "position_cost_price": 23.85,
                    "current_price": 23.80,
                    "total_value": 100000,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.metadata["trade_quantity"] == 200  # 50% of 400 = 200
            assert result.metadata["position_action"] == "reduce_position"
            assert "减半仓" in result.reasoning

    @pytest.mark.asyncio
    async def test_sell_low_confidence_reduces_30_percent(self):
        """Test SELL with low confidence (<50) sells 30% of position."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 40},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},
                "consensus_data": {"weighted_score": -40, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 400,
                    "position_cost_price": 23.85,
                    "current_price": 23.80,
                    "total_value": 100000,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.metadata["trade_quantity"] == 120  # 30% of 400 = 120
            assert result.metadata["position_action"] == "reduce_position"
            assert "减仓" in result.reasoning

    @pytest.mark.asyncio
    async def test_sell_without_position_returns_no_action(self):
        """Test SELL without position returns no_action."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 80},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},
                "consensus_data": {"weighted_score": -80, "risk_flags": []},
                "portfolio": {
                    "has_position": False,
                    "position_quantity": 0,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.metadata["trade_quantity"] == 0
            assert result.metadata["position_action"] == "no_action"
            assert result.metadata["position_ratio"] == 0.0

    @pytest.mark.asyncio
    async def test_hold_with_position_keeps_position(self):
        """Test HOLD with existing position returns keep_position."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "hold", "confidence": 50},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 0, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 400,
                    "position_cost_price": 23.85,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.metadata["trade_quantity"] == 0
            assert result.metadata["position_action"] == "keep_position"

    @pytest.mark.asyncio
    async def test_hold_without_position_returns_no_action(self):
        """Test HOLD without position returns no_action."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "hold", "confidence": 50},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 0, "risk_flags": []},
                "portfolio": {
                    "has_position": False,
                    "position_quantity": 0,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.metadata["trade_quantity"] == 0
            assert result.metadata["position_action"] == "no_action"

    @pytest.mark.asyncio
    async def test_buy_with_position_adds_position(self):
        """Test BUY with existing position calculates add_position quantity."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # Total value = 100000, current position = 400 shares @ 23.80 = 9520 (9.52%)
            # Target position ratio = 40% (confidence 50% * 0.8)
            # Target value = 40000, target shares = 40000/23.80 = 1680
            # Trade quantity = 1680 - 400 = 1280
            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 50},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.5}},
                "consensus_data": {"weighted_score": 50, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 400,
                    "position_cost_price": 23.85,
                    "current_price": 23.80,
                    "total_value": 100000,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.metadata["trade_quantity"] > 0
            assert result.metadata["position_action"] == "add_position"

    @pytest.mark.asyncio
    async def test_buy_without_position_opens_position(self):
        """Test BUY without existing position calculates open_position quantity."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            # Total value = 100000, target position ratio = 40%
            # Target value = 40000, target shares = 40000/23.80 = 1680
            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 50},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.5}},
                "consensus_data": {"weighted_score": 50, "risk_flags": []},
                "portfolio": {
                    "has_position": False,
                    "position_quantity": 0,
                    "current_price": 23.80,
                    "total_value": 100000,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.metadata["trade_quantity"] > 0
            assert result.metadata["position_action"] == "open_position"

    def test_make_rule_based_decision_sell_with_portfolio(self):
        """Test rule-based SELL decision with portfolio context."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "sell", "confidence": 80}},
                consensus_data={"weighted_score": -80, "risk_flags": []},
                max_position=1.0,
                portfolio={
                    "has_position": True,
                    "position_quantity": 400,
                },
            )

            assert result["action"] == "SELL"
            assert result["trade_quantity"] == 400
            assert result["position_action"] == "close_position"
            assert "清仓" in result["reasoning"]

    def test_make_rule_based_decision_sell_without_portfolio(self):
        """Test rule-based SELL decision without portfolio context."""
        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=None):
            agent = PortfolioManagerAgent()

            result = agent._make_rule_based_decision(
                agent_signals={"Test": {"signal": "sell", "confidence": 80}},
                consensus_data={"weighted_score": -80, "risk_flags": []},
                max_position=1.0,
                portfolio=None,
            )

            assert result["action"] == "SELL"
            assert result["trade_quantity"] == 0
            assert result["position_action"] == "no_action"
