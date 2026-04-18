"""Unit tests for PortfolioService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ashare_analyzer.portfolio.models import Position
from ashare_analyzer.portfolio.repository import PortfolioRepository
from ashare_analyzer.portfolio.service import PortfolioService


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    manager = MagicMock()
    manager.get_realtime_quote = AsyncMock()
    manager.prefetch_realtime_quotes = AsyncMock()
    return manager


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    return MagicMock(spec=PortfolioRepository)


@pytest.fixture
def service(mock_data_manager, mock_repository):
    """Create service with mocks."""
    return PortfolioService(mock_data_manager, mock_repository)


class TestPortfolioService:
    """Tests for PortfolioService."""

    def test_load_from_config_empty(self, service, mock_repository):
        """Test loading empty config."""
        from ashare_analyzer.config import PortfolioConfig

        config = PortfolioConfig(positions={})
        service.load_from_config(config)

        mock_repository.upsert.assert_not_called()

    def test_load_from_config_with_positions(self, service, mock_repository):
        """Test loading positions from config."""
        from ashare_analyzer.config import PortfolioConfig, PositionConfig

        config = PortfolioConfig(
            positions={
                "600519": PositionConfig(quantity=100, cost_price=1800.0),
                "300750": PositionConfig(quantity=200, cost_price=200.0),
            }
        )

        service.load_from_config(config)

        assert mock_repository.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, service, mock_repository):
        """Test getting positions when empty."""
        mock_repository.get_all.return_value = []

        positions = await service.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_with_prices(self, service, mock_repository, mock_data_manager):
        """Test getting positions with price enrichment."""
        from ashare_analyzer.storage import PortfolioState

        mock_position = MagicMock(spec=PortfolioState)
        mock_position.code = "600519"
        mock_position.quantity = 100
        mock_position.cost_price = 1800.0
        mock_repository.get_all.return_value = [mock_position]

        mock_quote = MagicMock()
        mock_quote.price = 1850.0
        mock_data_manager.get_realtime_quote.return_value = mock_quote

        positions = await service.get_positions()

        assert len(positions) == 1
        assert positions[0].code == "600519"
        assert positions[0].current_price == 1850.0
        assert positions[0].market_value == 185000.0
        assert positions[0].profit_loss == 5000.0
        assert abs(positions[0].profit_loss_pct - 2.78) < 0.01

    @pytest.mark.asyncio
    async def test_get_positions_price_fetch_failure(self, service, mock_repository, mock_data_manager):
        """Test getting positions when price fetch fails."""
        from ashare_analyzer.storage import PortfolioState

        mock_position = MagicMock(spec=PortfolioState)
        mock_position.code = "600519"
        mock_position.quantity = 100
        mock_position.cost_price = 1800.0
        mock_repository.get_all.return_value = [mock_position]

        mock_data_manager.get_realtime_quote.return_value = None

        positions = await service.get_positions()

        assert len(positions) == 1
        assert positions[0].current_price is None
        assert positions[0].market_value is None

    def test_enrich_with_prices(self, service):
        """Test enriching positions with price data."""
        positions = [
            Position(code="600519", quantity=100, cost_price=1800.0),
        ]
        price_map = {"600519": 1850.0}

        service.enrich_with_prices(positions, price_map)

        assert positions[0].current_price == 1850.0
        assert positions[0].market_value == 185000.0
        assert positions[0].profit_loss == 5000.0
        assert positions[0].profit_loss_pct is not None
        assert abs(positions[0].profit_loss_pct - 2.78) < 0.01
