"""Unit tests for TradeHistory model and repository."""

from datetime import datetime

import pytest

from ashare_analyzer.portfolio.repository import TradeHistoryRepository
from ashare_analyzer.storage import DatabaseManager
from ashare_analyzer.storage.models import TradeHistory


class TestTradeHistoryModel:
    """Tests for TradeHistory ORM model."""

    def test_create_trade_history(self):
        """Test creating a TradeHistory instance."""
        trade = TradeHistory(
            code="600519",
            name="贵州茅台",
            action="BUY",
            quantity=100,
            price=1800.0,
            signal_source="llm",
            confidence=75,
            weighted_score=35.5,
            position_action="open_position",
            agent_signals='{"TechnicalAgent": "BUY(80)", "ChipAgent": "BUY(75)"}',
            is_paper=1,
            status="pending",
        )

        assert trade.code == "600519"
        assert trade.action == "BUY"
        assert trade.quantity == 100
        assert trade.is_paper == 1
        assert trade.status == "pending"

    def test_to_dict(self):
        """Test converting TradeHistory to dictionary."""
        trade = TradeHistory(
            code="600519",
            name="贵州茅台",
            action="SELL",
            quantity=50,
            price=1850.0,
            signal_source="rule_based",
            confidence=60,
            weighted_score=-40.0,
            position_action="reduce_position",
            agent_signals='{"TechnicalAgent": "SELL(70)"}',
            is_paper=0,
            status="executed",
        )

        result = trade.to_dict()

        assert result["code"] == "600519"
        assert result["action"] == "SELL"
        assert result["quantity"] == 50
        assert result["is_paper"] is False


class TestTradeHistoryRepository:
    """Tests for TradeHistoryRepository."""

    @pytest.fixture
    def db_manager(self):
        """Create in-memory database for testing."""
        # Reset singleton to ensure fresh database for each test
        DatabaseManager._instance = None
        yield DatabaseManager("sqlite:///:memory:")
        DatabaseManager._instance = None

    @pytest.fixture
    def repository(self, db_manager):
        """Create repository with test database."""
        return TradeHistoryRepository(db_manager)

    def test_save_trade(self, repository):
        """Test saving a trade record."""
        trade_id = repository.save(
            code="600519",
            name="贵州茅台",
            action="BUY",
            quantity=100,
            price=1800.0,
            signal_source="llm",
            confidence=75,
            weighted_score=35.5,
            position_action="open_position",
            agent_signals={"TechnicalAgent": "BUY(80)"},
            is_paper=True,
        )

        assert trade_id is not None
        assert trade_id > 0

    def test_get_by_code(self, repository):
        """Test getting trades by stock code."""
        # Use unique codes to avoid test isolation issues
        repository.save(
            code="TEST001",
            action="BUY",
            quantity=100,
            price=1800.0,
            is_paper=True,
        )
        repository.save(
            code="TEST002",
            action="BUY",
            quantity=200,
            price=15.0,
            is_paper=True,
        )
        repository.save(
            code="TEST001",
            action="SELL",
            quantity=50,
            price=1850.0,
            is_paper=True,
        )

        trades = repository.get_by_code("TEST001")

        assert len(trades) == 2
        assert all(t.code == "TEST001" for t in trades)

    def test_get_recent(self, repository):
        """Test getting recent trades."""
        # Use unique codes to avoid test isolation issues
        for i in range(5):
            repository.save(
                code=f"RECENT{i}",
                action="BUY",
                quantity=100,
                price=100.0 + i * 10,
                is_paper=True,
            )

        recent = repository.get_recent(limit=3)

        assert len(recent) == 3
        # Should be ordered by created_at desc
        assert recent[0].code == "RECENT4"  # Most recent

    def test_update_status(self, repository):
        """Test updating trade status."""
        trade_id = repository.save(
            code="UPD001",
            action="BUY",
            quantity=100,
            price=1800.0,
            is_paper=True,
        )

        result = repository.update_status(trade_id, "executed", result_pnl=500.0)

        assert result is True

        trades = repository.get_by_code("UPD001")
        assert trades[0].status == "executed"
        assert trades[0].result_pnl == 500.0

    def test_get_pending_trades(self, repository):
        """Test getting pending trades."""
        repository.save(code="PEND001", action="BUY", quantity=100, price=1800.0, is_paper=True)
        repository.save(code="PEND002", action="SELL", quantity=50, price=15.0, is_paper=True)

        trade_id = repository.save(code="PEND003", action="BUY", quantity=200, price=10.0, is_paper=True)
        repository.update_status(trade_id, "executed")

        pending = repository.get_pending_trades()

        assert len(pending) == 2
        assert all(t.status == "pending" for t in pending)

    def test_get_trade_stats(self, repository):
        """Test getting trade statistics."""
        # Use unique codes for this test
        id1 = repository.save(code="STAT001", action="BUY", quantity=100, price=1800.0, is_paper=False)
        repository.update_status(id1, "executed", result_pnl=500.0)

        id2 = repository.save(code="STAT002", action="SELL", quantity=50, price=1850.0, is_paper=False)
        repository.update_status(id2, "executed", result_pnl=-200.0)

        id3 = repository.save(code="STAT003", action="BUY", quantity=200, price=15.0, is_paper=False)
        repository.update_status(id3, "executed", result_pnl=300.0)

        stats = repository.get_trade_stats()

        assert stats["executed_trades"] == 3
        assert stats["total_pnl"] == 600.0  # 500 - 200 + 300
        assert stats["win_count"] == 2  # 500 and 300 are positive

    def test_get_by_date_range(self, repository):
        """Test getting trades within a date range."""
        repository.save(code="DATE001", action="BUY", quantity=100, price=1800.0, is_paper=True)

        # Get today's trades
        today = datetime.now().date()
        trades = repository.get_by_date_range(today, today)

        # Find our trade in the results
        our_trades = [t for t in trades if t.code == "DATE001"]
        assert len(our_trades) == 1

        # Get future trades (should be empty for our code)
        from datetime import timedelta

        tomorrow = today + timedelta(days=1)
        future_trades = repository.get_by_date_range(tomorrow, tomorrow)
        our_future_trades = [t for t in future_trades if t.code == "DATE001"]
        assert len(our_future_trades) == 0
