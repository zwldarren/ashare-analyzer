"""Unit tests for PortfolioRepository."""

import pytest

from ashare_analyzer.portfolio.repository import PortfolioRepository
from ashare_analyzer.storage import DatabaseManager


@pytest.fixture
def db_manager():
    """Create in-memory database for testing."""
    return DatabaseManager("sqlite:///:memory:")


@pytest.fixture
def repository(db_manager):
    """Create repository with test database."""
    return PortfolioRepository(db_manager)


class TestPortfolioRepository:
    """Tests for PortfolioRepository."""

    def test_get_all_empty(self, repository):
        """Test getting all positions when empty."""
        positions = repository.get_all()
        assert positions == []

    def test_upsert_new_position(self, repository):
        """Test inserting a new position."""
        result = repository.upsert("600519", 100, 1800.0)
        assert result is True

        positions = repository.get_all()
        assert len(positions) == 1
        assert positions[0].code == "600519"
        assert positions[0].quantity == 100
        assert positions[0].cost_price == 1800.0

    def test_upsert_update_position(self, repository):
        """Test updating an existing position."""
        repository.upsert("600519", 100, 1800.0)
        repository.upsert("600519", 150, 1750.0)

        positions = repository.get_all()
        assert len(positions) == 1
        assert positions[0].quantity == 150
        assert positions[0].cost_price == 1750.0

    def test_get_by_code(self, repository):
        """Test getting a position by code."""
        repository.upsert("600519", 100, 1800.0)

        position = repository.get_by_code("600519")
        assert position is not None
        assert position.code == "600519"

    def test_get_by_code_not_found(self, repository):
        """Test getting a non-existent position."""
        position = repository.get_by_code("999999")
        assert position is None

    def test_delete(self, repository):
        """Test deleting a position."""
        repository.upsert("600519", 100, 1800.0)

        result = repository.delete("600519")
        assert result is True

        positions = repository.get_all()
        assert len(positions) == 0

    def test_delete_not_found(self, repository):
        """Test deleting a non-existent position."""
        result = repository.delete("999999")
        assert result is False
