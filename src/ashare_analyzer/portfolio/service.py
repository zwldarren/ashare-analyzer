"""
Portfolio service for position management.

Orchestrates repository and data manager for position operations.
"""

import logging
from typing import TYPE_CHECKING

from ashare_analyzer.portfolio.models import Position

if TYPE_CHECKING:
    from ashare_analyzer.config import PortfolioConfig
    from ashare_analyzer.data.manager import DataManager
    from ashare_analyzer.portfolio.repository import PortfolioRepository

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service for portfolio position management."""

    def __init__(self, data_manager: "DataManager", repository: "PortfolioRepository"):
        """Initialize service with dependencies.

        Args:
            data_manager: DataManager for fetching realtime quotes
            repository: PortfolioRepository for database operations
        """
        self._data_manager = data_manager
        self._repository = repository

    def load_from_config(self, portfolio_config: "PortfolioConfig") -> None:
        """Load positions from config into database.

        Args:
            portfolio_config: Portfolio configuration with positions
        """
        if not portfolio_config.positions:
            logger.debug("No positions in config to load")
            return

        for code, pos in portfolio_config.positions.items():
            success = self._repository.upsert(code, pos.quantity, pos.cost_price)
            if success:
                logger.info(f"Loaded position from config: {code} qty={pos.quantity} cost={pos.cost_price}")
            else:
                logger.warning(f"Failed to load position from config: {code}")

    async def get_positions(self) -> list[Position]:
        """Get all positions enriched with current prices.

        Returns:
            List of Position objects with runtime fields populated
        """
        db_positions = self._repository.get_all()

        if not db_positions:
            return []

        positions = [Position(code=p.code, quantity=p.quantity, cost_price=p.cost_price) for p in db_positions]  # ty:ignore[invalid-argument-type]

        codes = [p.code for p in positions]
        await self._data_manager.prefetch_realtime_quotes(codes)

        for position in positions:
            try:
                quote = await self._data_manager.get_realtime_quote(position.code)
                if quote and quote.price:
                    position.current_price = float(quote.price)
                    position.market_value = position.quantity * position.current_price
                    position.profit_loss = (position.current_price - position.cost_price) * position.quantity
                    if position.cost_price > 0:
                        position.profit_loss_pct = (
                            (position.current_price - position.cost_price) / position.cost_price * 100
                        )
            except Exception as e:
                logger.warning(f"Failed to get price for {position.code}: {e}")

        return positions

    def enrich_with_prices(self, positions: list[Position], price_map: dict[str, float]) -> None:
        """Enrich positions with price data from a price map.

        Args:
            positions: List of positions to enrich (modified in place)
            price_map: Dict mapping stock codes to current prices
        """
        for position in positions:
            price = price_map.get(position.code)
            if price:
                position.current_price = price
                position.market_value = position.quantity * price
                position.profit_loss = (price - position.cost_price) * position.quantity
                if position.cost_price > 0:
                    position.profit_loss_pct = (price - position.cost_price) / position.cost_price * 100
