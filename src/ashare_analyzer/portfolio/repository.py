"""
Portfolio repository for database operations.

Provides CRUD operations for portfolio positions.
"""

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from ashare_analyzer.storage import PortfolioState

if TYPE_CHECKING:
    from ashare_analyzer.storage import DatabaseManager

logger = logging.getLogger(__name__)


class PortfolioRepository:
    """Repository for portfolio position database operations."""

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize repository with database manager.

        Args:
            db_manager: DatabaseManager instance for DB operations
        """
        self._db_manager = db_manager

    def get_all(self) -> list[PortfolioState]:
        """Get all portfolio positions.

        Returns:
            List of PortfolioState objects
        """
        with self._db_manager.get_session() as session:
            results = session.execute(select(PortfolioState)).scalars().all()
            return list(results)

    def get_by_code(self, code: str) -> PortfolioState | None:
        """Get a position by stock code.

        Args:
            code: Stock code

        Returns:
            PortfolioState or None if not found
        """
        with self._db_manager.get_session() as session:
            result = session.execute(select(PortfolioState).where(PortfolioState.code == code)).scalar_one_or_none()
            return result

    def upsert(self, code: str, quantity: int, cost_price: float) -> bool:
        """Insert or update a position.

        Args:
            code: Stock code
            quantity: Number of shares
            cost_price: Average cost price

        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime

        try:
            with self._db_manager.get_session() as session:
                existing = session.execute(
                    select(PortfolioState).where(PortfolioState.code == code)
                ).scalar_one_or_none()

                if existing:
                    existing.quantity = quantity
                    existing.cost_price = cost_price
                    existing.updated_at = datetime.now()
                    logger.debug(f"Updated position: {code} qty={quantity} cost={cost_price}")
                else:
                    position = PortfolioState(
                        code=code,
                        quantity=quantity,
                        cost_price=cost_price,
                    )
                    session.add(position)
                    logger.debug(f"Inserted position: {code} qty={quantity} cost={cost_price}")

                session.commit()
                return True

        except SQLAlchemyError as e:
            logger.error(f"Failed to upsert position {code}: {e}")
            return False

    def delete(self, code: str) -> bool:
        """Delete a position by stock code.

        Args:
            code: Stock code

        Returns:
            True if deleted, False if not found or error
        """
        try:
            with self._db_manager.get_session() as session:
                existing = session.execute(
                    select(PortfolioState).where(PortfolioState.code == code)
                ).scalar_one_or_none()

                if existing:
                    session.delete(existing)
                    session.commit()
                    logger.debug(f"Deleted position: {code}")
                    return True

                return False

        except SQLAlchemyError as e:
            logger.error(f"Failed to delete position {code}: {e}")
            return False
