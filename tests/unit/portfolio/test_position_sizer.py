"""Unit tests for PositionSizer."""

import pytest

from ashare_analyzer.portfolio.models import Portfolio, Position
from ashare_analyzer.portfolio.position_sizer import (
    PositionDecision,
    PositionSizer,
    RiskLimits,
)


class TestRiskLimits:
    """Tests for RiskLimits dataclass."""

    def test_default_values(self):
        """Test default risk limits."""
        limits = RiskLimits()
        assert limits.max_single_position_pct == 0.25
        assert limits.max_total_exposure_pct == 0.95
        assert limits.min_confidence_to_trade == 30
        assert limits.min_sell_confidence_full == 70
        assert limits.min_sell_confidence_half == 50

    def test_custom_values(self):
        """Test custom risk limits."""
        limits = RiskLimits(
            max_single_position_pct=0.30,
            min_confidence_to_trade=50,
        )
        assert limits.max_single_position_pct == 0.30
        assert limits.min_confidence_to_trade == 50


class TestPositionDecision:
    """Tests for PositionDecision dataclass."""

    def test_buy_decision(self):
        """Test creating a BUY decision."""
        decision = PositionDecision(
            action="BUY",
            position_action="open_position",
            position_ratio=0.20,
            trade_quantity=100,
            reasoning="Strong buy signal",
            risk_adjusted=True,
        )
        assert decision.action == "BUY"
        assert decision.position_action == "open_position"
        assert decision.trade_quantity == 100

    def test_sell_decision(self):
        """Test creating a SELL decision."""
        decision = PositionDecision(
            action="SELL",
            position_action="close_position",
            position_ratio=0.0,
            trade_quantity=50,
            reasoning="High confidence sell",
            risk_adjusted=False,
        )
        assert decision.action == "SELL"
        assert decision.trade_quantity == 50


class TestPositionSizer:
    """Tests for PositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create a PositionSizer with default limits."""
        return PositionSizer()

    @pytest.fixture
    def sizer_conservative(self):
        """Create a conservative PositionSizer."""
        return PositionSizer(
            risk_limits=RiskLimits(
                max_single_position_pct=0.15,
                min_confidence_to_trade=50,
            )
        )

    # --- BUY Signal Tests ---

    def test_buy_no_position(self, sizer):
        """Test BUY signal when no existing position."""
        decision = sizer.calculate(
            signal="BUY",
            confidence=70,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "BUY"
        assert decision.position_action == "open_position"
        assert decision.position_ratio > 0
        assert decision.trade_quantity > 0
        assert decision.risk_adjusted is True

    def test_buy_with_existing_position(self, sizer):
        """Test BUY signal when already have position."""
        position = Position(code="600519", quantity=50, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=50000.0)

        decision = sizer.calculate(
            signal="BUY",
            confidence=60,
            portfolio=portfolio,
            position=position,
            current_price=110.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "BUY"
        assert decision.position_action == "add_position"
        assert decision.trade_quantity > 0

    def test_buy_respects_max_position_limit(self, sizer):
        """Test that BUY respects max position limit."""
        decision = sizer.calculate(
            signal="BUY",
            confidence=90,  # High confidence would want more
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.10,  # Limited to 10%
        )

        assert decision.position_ratio <= 0.10
        # 10% of 100000 = 10000, at price 100 = 100 shares max
        assert decision.trade_quantity <= 100

    def test_buy_low_confidence_no_trade(self, sizer):
        """Test that low confidence prevents trading."""
        decision = sizer.calculate(
            signal="BUY",
            confidence=20,  # Below min_confidence_to_trade
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "HOLD"
        assert decision.trade_quantity == 0
        assert "置信度不足" in decision.reasoning

    # --- SELL Signal Tests ---

    def test_sell_high_confidence_close_position(self, sizer):
        """Test SELL with high confidence closes entire position."""
        position = Position(code="600519", quantity=100, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=100000.0)

        decision = sizer.calculate(
            signal="SELL",
            confidence=80,  # High confidence
            portfolio=portfolio,
            position=position,
            current_price=110.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "SELL"
        assert decision.position_action == "close_position"
        assert decision.trade_quantity == 100

    def test_sell_medium_confidence_partial(self, sizer):
        """Test SELL with medium confidence reduces position."""
        position = Position(code="600519", quantity=100, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=100000.0)

        decision = sizer.calculate(
            signal="SELL",
            confidence=50,  # Medium confidence
            portfolio=portfolio,
            position=position,
            current_price=110.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "SELL"
        assert decision.position_action == "reduce_position"
        assert decision.trade_quantity < 100  # Partial sell
        assert decision.trade_quantity > 0

    def test_sell_no_position(self, sizer):
        """Test SELL signal when no position."""
        decision = sizer.calculate(
            signal="SELL",
            confidence=80,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "HOLD"
        assert decision.trade_quantity == 0
        assert "无持仓" in decision.reasoning

    def test_sell_low_confidence_small_reduction(self, sizer):
        """Test SELL with low confidence does small reduction."""
        position = Position(code="600519", quantity=100, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=100000.0)

        decision = sizer.calculate(
            signal="SELL",
            confidence=25,  # Below min_sell_confidence_partial (30)
            portfolio=portfolio,
            position=position,
            current_price=110.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        # With confidence 25 (< min_confidence_to_trade=30), should return HOLD
        assert decision.action == "HOLD"
        assert decision.trade_quantity == 0

    # --- HOLD Signal Tests ---

    def test_hold_no_position(self, sizer):
        """Test HOLD signal when no position."""
        decision = sizer.calculate(
            signal="HOLD",
            confidence=50,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "HOLD"
        assert decision.position_action == "no_action"
        assert decision.trade_quantity == 0

    def test_hold_with_position(self, sizer):
        """Test HOLD signal when have position."""
        position = Position(code="600519", quantity=100, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=100000.0)

        decision = sizer.calculate(
            signal="HOLD",
            confidence=50,
            portfolio=portfolio,
            position=position,
            current_price=110.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.action == "HOLD"
        assert decision.position_action == "keep_position"
        assert decision.trade_quantity == 0

    # --- Validation Tests ---

    def test_validate_trade_within_limits(self, sizer):
        """Test validation passes for trade within limits."""
        position = Position(code="600519", quantity=50, cost_price=100.0)
        portfolio = Portfolio(positions=[position], total_value=50000.0)

        decision = PositionDecision(
            action="BUY",
            position_action="add_position",
            position_ratio=0.20,
            trade_quantity=100,
            reasoning="Test",
            risk_adjusted=True,
        )

        result = sizer.validate_trade(decision, portfolio, total_value=100000.0)
        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_validate_trade_exceeds_single_position_limit(self, sizer_conservative):
        """Test validation fails when exceeding single position limit."""
        portfolio = Portfolio(positions=[], total_value=0)

        decision = PositionDecision(
            action="BUY",
            position_action="open_position",
            position_ratio=0.20,  # Exceeds 0.15 limit
            trade_quantity=200,
            reasoning="Test",
            risk_adjusted=False,
        )

        result = sizer_conservative.validate_trade(decision, portfolio, total_value=100000.0)
        assert result.is_valid is False
        assert any("单股仓位" in v for v in result.violations)

    def test_validate_trade_exceeds_total_exposure(self, sizer):
        """Test validation fails when exceeding total exposure."""
        # Portfolio with 90% exposure
        position = Position(
            code="600000",
            quantity=900,
            cost_price=100.0,
            current_price=100.0,
            market_value=90000.0,
        )
        portfolio = Portfolio(positions=[position], total_value=90000.0)

        decision = PositionDecision(
            action="BUY",
            position_action="open_position",
            position_ratio=0.20,  # Would push over 95%
            trade_quantity=200,
            reasoning="Test",
            risk_adjusted=False,
        )

        result = sizer.validate_trade(decision, portfolio, total_value=100000.0)
        assert result.is_valid is False
        assert any("总仓位" in v for v in result.violations)

    # --- Edge Cases ---

    def test_zero_price(self, sizer):
        """Test handling zero price."""
        decision = sizer.calculate(
            signal="BUY",
            confidence=70,
            portfolio=None,
            position=None,
            current_price=0.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        assert decision.trade_quantity == 0

    def test_zero_total_value(self, sizer):
        """Test handling zero total value."""
        decision = sizer.calculate(
            signal="BUY",
            confidence=70,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=0.0,
            max_position_limit=0.25,
        )

        assert decision.trade_quantity == 0

    def test_confidence_scaling(self, sizer):
        """Test that higher confidence leads to larger position."""
        # Create a sizer with higher risk limit to see the confidence scaling
        from ashare_analyzer.portfolio.position_sizer import RiskLimits

        relaxed_sizer = PositionSizer(risk_limits=RiskLimits(max_single_position_pct=0.50))

        decision_low = relaxed_sizer.calculate(
            signal="BUY",
            confidence=40,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.50,
        )

        decision_high = relaxed_sizer.calculate(
            signal="BUY",
            confidence=80,
            portfolio=None,
            position=None,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.50,
        )

        # Higher confidence should result in higher position ratio
        # confidence 40 -> 0.32 (40% * 0.8), confidence 80 -> 0.40 (80% * 0.8, capped at 0.5)
        assert decision_high.position_ratio > decision_low.position_ratio
