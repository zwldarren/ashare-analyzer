"""Unit tests for portfolio models."""

from ashare_analyzer.portfolio.models import Portfolio, Position


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation_basic(self):
        """Test creating a position with basic fields."""
        pos = Position(code="600519", quantity=100, cost_price=1800.0)
        assert pos.code == "600519"
        assert pos.quantity == 100
        assert pos.cost_price == 1800.0
        assert pos.current_price is None
        assert pos.market_value is None
        assert pos.profit_loss is None
        assert pos.profit_loss_pct is None

    def test_position_to_dict(self):
        """Test converting position to dictionary."""
        pos = Position(
            code="600519",
            quantity=100,
            cost_price=1800.0,
            current_price=1850.0,
            market_value=185000.0,
            profit_loss=5000.0,
            profit_loss_pct=2.78,
        )
        result = pos.to_dict()
        assert result["code"] == "600519"
        assert result["quantity"] == 100
        assert result["cost_price"] == 1800.0
        assert result["current_price"] == 1850.0
        assert result["market_value"] == 185000.0
        assert result["profit_loss"] == 5000.0
        assert result["profit_loss_pct"] == 2.78


class TestPortfolio:
    """Tests for Portfolio dataclass."""

    def test_portfolio_creation(self):
        """Test creating a portfolio with positions."""
        positions = [
            Position(code="600519", quantity=100, cost_price=1800.0, market_value=185000.0),
            Position(code="300750", quantity=200, cost_price=200.0, market_value=42000.0),
        ]
        portfolio = Portfolio(positions=positions)
        assert len(portfolio.positions) == 2
        assert portfolio.total_value == 227000.0

    def test_portfolio_empty(self):
        """Test creating an empty portfolio."""
        portfolio = Portfolio(positions=[])
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 0.0
