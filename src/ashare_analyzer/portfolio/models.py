"""
Portfolio models for position tracking.

Contains Position and Portfolio dataclasses for representing stock holdings.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Position:
    """Single stock position with runtime-calculated metrics."""

    code: str
    quantity: int
    cost_price: float

    current_price: float | None = None
    market_value: float | None = None
    profit_loss: float | None = None
    profit_loss_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for context passing."""
        return {
            "code": self.code,
            "quantity": self.quantity,
            "cost_price": self.cost_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
        }


@dataclass
class Portfolio:
    """Collection of positions with aggregated metrics."""

    positions: list[Position]
    total_value: float = 0.0

    def __post_init__(self) -> None:
        """Calculate total value from positions."""
        self.total_value = sum(p.market_value or 0 for p in self.positions)
