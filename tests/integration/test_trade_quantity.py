"""Integration tests for trade quantity feature."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ashare_analyzer.models import SignalType


class TestTradeQuantityIntegration:
    """Integration tests for trade quantity decisions."""

    @pytest.mark.asyncio
    async def test_sell_decision_with_quantity(self):
        """Test that SELL decision includes specific quantity."""
        # Mock LLM client to return specific trade quantity
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "sell",
                "confidence": 65,
                "reasoning": "技术面走弱，建议减仓50%",
                "position_ratio": 0.0,
                "trade_quantity": 500,
                "position_action": "reduce_position",
                "key_factors": ["趋势下行"],
                "risk_level": "medium",
            }
        )

        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=mock_llm_client):
            from ashare_analyzer.ai.agents import PortfolioManagerAgent

            agent = PortfolioManagerAgent()
            context = {
                "code": "000338",
                "stock_name": "潍柴动力",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "sell", "confidence": 70, "reasoning": "趋势走弱"},
                    "FundamentalAgent": {"signal": "sell", "confidence": 80, "reasoning": "基本面恶化"},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 1.0}},
                "consensus_data": {"weighted_score": -50, "consensus_level": 0.7, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 1000,
                    "position_cost_price": 24.55,
                    "current_price": 24.55,
                    "current_profit_loss_pct": 0.0,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.SELL
            assert result.metadata["trade_quantity"] == 500
            assert result.metadata["position_action"] == "reduce_position"

    @pytest.mark.asyncio
    async def test_buy_decision_with_quantity(self):
        """Test that BUY decision includes specific quantity."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "buy",
                "confidence": 85,
                "reasoning": "技术面突破，建议建仓",
                "position_ratio": 0.15,
                "trade_quantity": 800,
                "position_action": "open_position",
                "key_factors": ["突破阻力位"],
                "risk_level": "low",
            }
        )

        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=mock_llm_client):
            from ashare_analyzer.ai.agents import PortfolioManagerAgent

            agent = PortfolioManagerAgent()
            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "buy", "confidence": 90, "reasoning": "突破"},
                    "FundamentalAgent": {"signal": "buy", "confidence": 80, "reasoning": "估值合理"},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 70, "consensus_level": 0.85, "risk_flags": []},
                "portfolio": {
                    "has_position": False,
                    "position_quantity": 0,
                    "position_cost_price": 0,
                    "current_price": 1800.0,
                    "current_profit_loss_pct": None,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.BUY
            assert result.metadata["trade_quantity"] == 800
            assert result.metadata["position_action"] == "open_position"

    @pytest.mark.asyncio
    async def test_hold_decision_with_position(self):
        """Test that HOLD decision with position returns keep_position."""
        mock_llm_client = MagicMock()
        mock_llm_client.is_available.return_value = True
        mock_llm_client.generate_with_tool = AsyncMock(
            return_value={
                "signal": "hold",
                "confidence": 50,
                "reasoning": "信号中性，维持现有仓位",
                "position_ratio": 0.0,
                "trade_quantity": 0,
                "position_action": "keep_position",
                "key_factors": [],
                "risk_level": "low",
            }
        )

        with patch("ashare_analyzer.ai.agents.portfolio_manager.get_llm_client", return_value=mock_llm_client):
            from ashare_analyzer.ai.agents import PortfolioManagerAgent

            agent = PortfolioManagerAgent()
            context = {
                "code": "600519",
                "stock_name": "贵州茅台",
                "agent_signals": {
                    "TechnicalAgent": {"signal": "hold", "confidence": 50, "reasoning": "震荡"},
                    "FundamentalAgent": {"signal": "hold", "confidence": 50, "reasoning": "估值合理"},
                },
                "risk_manager_signal": {"metadata": {"max_position_size": 0.25}},
                "consensus_data": {"weighted_score": 0, "consensus_level": 0.5, "risk_flags": []},
                "portfolio": {
                    "has_position": True,
                    "position_quantity": 500,
                    "position_cost_price": 1800.0,
                    "current_price": 1800.0,
                    "current_profit_loss_pct": 0.0,
                },
            }

            result = await agent.analyze(context)

            assert result.signal == SignalType.HOLD
            assert result.metadata["trade_quantity"] == 0
            assert result.metadata["position_action"] == "keep_position"
