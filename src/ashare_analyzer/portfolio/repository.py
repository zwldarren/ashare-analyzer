"""
Portfolio repository for database operations.

Provides CRUD operations for portfolio positions and trade history.
"""

import json
import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from ashare_analyzer.storage import PortfolioState, TradeHistory

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
                    existing.quantity = quantity  # ty:ignore[invalid-assignment]
                    existing.cost_price = cost_price  # ty:ignore[invalid-assignment]
                    existing.updated_at = datetime.now()  # ty:ignore[invalid-assignment]
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


class TradeHistoryRepository:
    """Repository for trade history database operations."""

    def __init__(self, db_manager: "DatabaseManager"):
        """Initialize repository with database manager.

        Args:
            db_manager: DatabaseManager instance for DB operations
        """
        self._db_manager = db_manager

    def save(
        self,
        code: str,
        action: str,
        quantity: int,
        price: float | None = None,
        name: str | None = None,
        signal_source: str | None = None,
        confidence: int | None = None,
        weighted_score: float | None = None,
        position_action: str | None = None,
        agent_signals: dict[str, Any] | None = None,
        is_paper: bool = True,
    ) -> int | None:
        """Save a trade record.

        Args:
            code: Stock code
            action: Trade action (BUY/SELL)
            quantity: Number of shares
            price: Trade price
            name: Stock name
            signal_source: Source of signal (llm/rule_based)
            confidence: Signal confidence level
            weighted_score: Weighted score from analysis
            position_action: Position action type
            agent_signals: Dict of agent signals
            is_paper: Whether this is paper trading

        Returns:
            Trade ID if successful, None otherwise
        """
        try:
            with self._db_manager.get_session() as session:
                trade = TradeHistory(
                    code=code,
                    name=name,
                    action=action.upper(),
                    quantity=quantity,
                    price=price,
                    signal_source=signal_source,
                    confidence=confidence,
                    weighted_score=weighted_score,
                    position_action=position_action,
                    agent_signals=json.dumps(agent_signals) if agent_signals else None,
                    is_paper=1 if is_paper else 0,
                    status="pending",
                )
                session.add(trade)
                session.commit()

                logger.debug(f"Saved trade: {action} {quantity} {code} @ {price}")
                # SQLAlchemy column type not recognized by ty
                return trade.id  # ty:ignore[invalid-return-type]

        except SQLAlchemyError as e:
            logger.error(f"Failed to save trade for {code}: {e}")
            return None

    def get_by_code(self, code: str) -> list[TradeHistory]:
        """Get trades by stock code.

        Args:
            code: Stock code

        Returns:
            List of TradeHistory records
        """
        with self._db_manager.get_session() as session:
            results = (
                session.execute(
                    select(TradeHistory).where(TradeHistory.code == code).order_by(TradeHistory.created_at.desc())
                )
                .scalars()
                .all()
            )
            return list(results)

    def get_recent(self, limit: int = 10) -> list[TradeHistory]:
        """Get recent trades.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of TradeHistory records ordered by created_at desc
        """
        with self._db_manager.get_session() as session:
            results = (
                session.execute(select(TradeHistory).order_by(TradeHistory.created_at.desc()).limit(limit))
                .scalars()
                .all()
            )
            return list(results)

    def update_status(self, trade_id: int, status: str, result_pnl: float | None = None) -> bool:
        """Update trade status.

        Args:
            trade_id: Trade ID
            status: New status (pending/executed/cancelled)
            result_pnl: Realized PnL (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._db_manager.get_session() as session:
                trade = session.execute(select(TradeHistory).where(TradeHistory.id == trade_id)).scalar_one_or_none()

                if trade:
                    trade.status = status  # ty:ignore[invalid-assignment]
                    if result_pnl is not None:
                        trade.result_pnl = result_pnl  # ty:ignore[invalid-assignment]
                    trade.updated_at = datetime.now()  # ty:ignore[invalid-assignment]
                    session.commit()
                    logger.debug(f"Updated trade {trade_id} status to {status}")
                    return True

                return False

        except SQLAlchemyError as e:
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False

    def get_pending_trades(self) -> list[TradeHistory]:
        """Get all pending trades.

        Returns:
            List of pending TradeHistory records
        """
        with self._db_manager.get_session() as session:
            results = (
                session.execute(
                    select(TradeHistory)
                    .where(TradeHistory.status == "pending")
                    .order_by(TradeHistory.created_at.desc())
                )
                .scalars()
                .all()
            )
            return list(results)

    def get_by_date_range(self, start_date: date, end_date: date) -> list[TradeHistory]:
        """Get trades within a date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of TradeHistory records
        """
        with self._db_manager.get_session() as session:
            results = (
                session.execute(
                    select(TradeHistory)
                    .where(TradeHistory.created_at >= datetime.combine(start_date, datetime.min.time()))
                    .where(TradeHistory.created_at <= datetime.combine(end_date, datetime.max.time()))
                    .order_by(TradeHistory.created_at.desc())
                )
                .scalars()
                .all()
            )
            return list(results)

    def get_trade_stats(self) -> dict[str, Any]:
        """Get trade statistics.

        Returns:
            Dict with trade statistics
        """
        with self._db_manager.get_session() as session:
            # Total trades
            total = session.execute(select(func.count(TradeHistory.id))).scalar() or 0

            # Executed trades
            executed = (
                session.execute(select(func.count(TradeHistory.id)).where(TradeHistory.status == "executed")).scalar()
                or 0
            )

            # Get all executed trades with PnL
            executed_trades = (
                session.execute(
                    select(TradeHistory)
                    .where(TradeHistory.status == "executed")
                    .where(TradeHistory.result_pnl.isnot(None))
                )
                .scalars()
                .all()
            )

            total_pnl = sum(t.result_pnl or 0 for t in executed_trades)
            win_count = sum(1 for t in executed_trades if (t.result_pnl or 0) > 0)

            return {
                "total_trades": total,
                "executed_trades": executed,
                "pending_trades": total - executed,
                "total_pnl": total_pnl,
                "win_count": win_count,
                "win_rate": win_count / executed if executed > 0 else 0,
            }

    def delete(self, trade_id: int) -> bool:
        """Delete a trade record.

        Args:
            trade_id: Trade ID

        Returns:
            True if deleted, False if not found or error
        """
        try:
            with self._db_manager.get_session() as session:
                trade = session.execute(select(TradeHistory).where(TradeHistory.id == trade_id)).scalar_one_or_none()

                if trade:
                    session.delete(trade)
                    session.commit()
                    logger.debug(f"Deleted trade {trade_id}")
                    return True

                return False

        except SQLAlchemyError as e:
            logger.error(f"Failed to delete trade {trade_id}: {e}")
            return False
