"""
Position Sizer Module

Provides independent position sizing calculation with risk limits enforcement.

This module separates the position sizing logic from the PortfolioManagerAgent,
enabling:
- Testable position calculations
- Configurable risk limits
- Risk validation before execution
"""

from dataclasses import dataclass, field

from ashare_analyzer.portfolio.models import Portfolio, Position


@dataclass
class RiskLimits:
    """
    Risk limits configuration for position sizing.

    Attributes:
        max_single_position_pct: Maximum allocation for a single stock (0-1)
        max_total_exposure_pct: Maximum total portfolio exposure (0-1)
        min_confidence_to_trade: Minimum confidence level to execute trades
        min_sell_confidence_full: Minimum confidence for full position close (>= this = sell all)
        min_sell_confidence_half: Minimum confidence for 50% position reduction
    """

    max_single_position_pct: float = 0.25
    max_total_exposure_pct: float = 0.95
    min_confidence_to_trade: int = 30
    min_sell_confidence_full: int = 70
    min_sell_confidence_half: int = 50

    def __post_init__(self) -> None:
        """Validate risk limits."""
        if not 0 < self.max_single_position_pct <= 1:
            raise ValueError("max_single_position_pct must be between 0 and 1")
        if not 0 < self.max_total_exposure_pct <= 1:
            raise ValueError("max_total_exposure_pct must be between 0 and 1")
        if not 0 <= self.min_confidence_to_trade <= 100:
            raise ValueError("min_confidence_to_trade must be between 0 and 100")


@dataclass
class PositionDecision:
    """
    Position sizing decision result.

    Attributes:
        action: Trading action (BUY/SELL/HOLD)
        position_action: Specific position action
        position_ratio: Target position ratio (0-1)
        trade_quantity: Number of shares to trade
        reasoning: Explanation for the decision
        risk_adjusted: Whether decision was adjusted for risk limits
    """

    action: str
    position_action: str
    position_ratio: float
    trade_quantity: int
    reasoning: str
    risk_adjusted: bool = False


@dataclass
class ValidationResult:
    """
    Result of trade validation against risk limits.

    Attributes:
        is_valid: Whether the trade passes all risk checks
        violations: List of risk limit violations
        adjusted_quantity: Suggested adjusted quantity (if applicable)
    """

    is_valid: bool
    violations: list[str] = field(default_factory=list)
    adjusted_quantity: int | None = None


class PositionSizer:
    """
    Independent position sizing calculator.

    Calculates trade quantities based on signals, confidence, and risk limits.
    Separates position sizing logic from decision-making for testability.

    Example:
        sizer = PositionSizer()
        decision = sizer.calculate(
            signal="BUY",
            confidence=70,
            portfolio=current_portfolio,
            position=current_position,
            current_price=100.0,
            total_value=100000.0,
            max_position_limit=0.25,
        )

        result = sizer.validate_trade(decision, portfolio, total_value)
        if not result.is_valid:
            # Handle risk violation
    """

    def __init__(self, risk_limits: RiskLimits | None = None):
        """
        Initialize PositionSizer with optional risk limits.

        Args:
            risk_limits: Custom risk limits configuration. Uses defaults if None.
        """
        self.risk_limits = risk_limits or RiskLimits()

    def calculate(
        self,
        signal: str,
        confidence: int,
        portfolio: Portfolio | None,
        position: Position | None,
        current_price: float,
        total_value: float,
        max_position_limit: float = 0.25,
    ) -> PositionDecision:
        """
        Calculate position size and trade quantity.

        Args:
            signal: Trading signal (BUY/SELL/HOLD)
            confidence: Signal confidence level (0-100)
            portfolio: Current portfolio state
            position: Current position for this stock (if any)
            current_price: Current stock price
            total_value: Total portfolio value
            max_position_limit: Maximum position ratio from risk manager

        Returns:
            PositionDecision with trade details
        """
        signal = signal.upper()

        # Check minimum confidence
        if confidence < self.risk_limits.min_confidence_to_trade and signal != "HOLD":
            return PositionDecision(
                action="HOLD",
                position_action="no_action",
                position_ratio=0.0,
                trade_quantity=0,
                reasoning=f"置信度不足 ({confidence}% < {self.risk_limits.min_confidence_to_trade}%)",
                risk_adjusted=False,
            )

        if signal == "BUY":
            return self._calculate_buy(confidence, position, current_price, total_value, max_position_limit)
        elif signal == "SELL":
            return self._calculate_sell(confidence, position)
        else:
            return self._calculate_hold(position)

    def _calculate_buy(
        self,
        confidence: int,
        position: Position | None,
        current_price: float,
        total_value: float,
        max_position_limit: float,
    ) -> PositionDecision:
        """Calculate BUY position sizing."""
        # Handle edge cases
        if current_price <= 0 or total_value <= 0:
            return PositionDecision(
                action="HOLD",
                position_action="no_action",
                position_ratio=0.0,
                trade_quantity=0,
                reasoning="价格或总资产无效",
                risk_adjusted=False,
            )

        # Calculate base position ratio from confidence
        # Higher confidence = larger position (max 80% of limit at 100% confidence)
        confidence_factor = confidence / 100.0
        base_position = confidence_factor * 0.8

        # Apply risk limits
        effective_limit = min(max_position_limit, self.risk_limits.max_single_position_pct)
        position_ratio = min(base_position, effective_limit)

        # Calculate target shares
        target_value = total_value * position_ratio
        target_shares = int(target_value / current_price)

        # Determine position action
        has_position = position is not None and position.quantity > 0

        if has_position and position is not None:
            current_shares = position.quantity
            trade_quantity = max(0, target_shares - current_shares)
            position_action = "add_position" if trade_quantity > 0 else "keep_position"
            reasoning = f"增持至目标仓位 {position_ratio * 100:.1f}%"
        else:
            trade_quantity = target_shares
            position_action = "open_position"
            reasoning = f"新建仓位 {position_ratio * 100:.1f}%"

        # Check if risk adjusted
        risk_adjusted = base_position > effective_limit

        return PositionDecision(
            action="BUY",
            position_action=position_action,
            position_ratio=position_ratio,
            trade_quantity=trade_quantity,
            reasoning=reasoning,
            risk_adjusted=risk_adjusted,
        )

    def _calculate_sell(self, confidence: int, position: Position | None) -> PositionDecision:
        """Calculate SELL position sizing."""
        # No position to sell
        if position is None or position.quantity <= 0:
            return PositionDecision(
                action="HOLD",
                position_action="no_action",
                position_ratio=0.0,
                trade_quantity=0,
                reasoning="无持仓可卖出",
                risk_adjusted=False,
            )

        quantity = position.quantity

        # Determine sell amount based on confidence
        if confidence >= self.risk_limits.min_sell_confidence_full:
            # High confidence (>=70): close entire position
            trade_quantity = quantity
            position_action = "close_position"
            reasoning = f"高置信度 ({confidence}%)，清仓卖出"
        elif confidence >= self.risk_limits.min_sell_confidence_half:
            # Medium confidence (>=50): sell 50%
            trade_quantity = int(quantity * 0.5)
            position_action = "reduce_position"
            reasoning = f"中等置信度 ({confidence}%)，减半仓"
        else:
            # Low confidence (<50): sell 30%
            trade_quantity = int(quantity * 0.3)
            position_action = "reduce_position"
            reasoning = f"低置信度 ({confidence}%)，小幅减仓"

        return PositionDecision(
            action="SELL",
            position_action=position_action,
            position_ratio=0.0,
            trade_quantity=trade_quantity,
            reasoning=reasoning,
            risk_adjusted=False,
        )

    def _calculate_hold(self, position: Position | None) -> PositionDecision:
        """Calculate HOLD (no trade)."""
        has_position = position is not None and position.quantity > 0

        if has_position:
            position_action = "keep_position"
            reasoning = "维持现有持仓"
        else:
            position_action = "no_action"
            reasoning = "观望，无操作"

        return PositionDecision(
            action="HOLD",
            position_action=position_action,
            position_ratio=0.0,
            trade_quantity=0,
            reasoning=reasoning,
            risk_adjusted=False,
        )

    def validate_trade(
        self,
        decision: PositionDecision,
        portfolio: Portfolio | None,
        total_value: float,
    ) -> ValidationResult:
        """
        Validate trade against risk limits.

        Args:
            decision: Position decision to validate
            portfolio: Current portfolio state
            total_value: Total portfolio value

        Returns:
            ValidationResult with validation status and any violations
        """
        violations: list[str] = []

        if decision.action == "HOLD":
            return ValidationResult(is_valid=True, violations=[])

        # Check single position limit
        if decision.position_ratio > self.risk_limits.max_single_position_pct:
            violations.append(
                f"单股仓位 {decision.position_ratio * 100:.1f}% 超过限制 "
                f"{self.risk_limits.max_single_position_pct * 100:.1f}%"
            )

        # Check total exposure limit
        if portfolio and decision.action == "BUY":
            current_exposure = portfolio.total_value / total_value if total_value > 0 else 0
            new_exposure = current_exposure + decision.position_ratio

            if new_exposure > self.risk_limits.max_total_exposure_pct:
                violations.append(
                    f"总仓位 {new_exposure * 100:.1f}% 将超过限制 {self.risk_limits.max_total_exposure_pct * 100:.1f}%"
                )

        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
        )

    def calculate_safe_quantity(
        self,
        target_value: float,
        current_price: float,
        lot_size: int = 100,
    ) -> int:
        """
        Calculate safe trade quantity rounded to lot size.

        Args:
            target_value: Target position value
            current_price: Current stock price
            lot_size: Trading lot size (default 100 for A-shares)

        Returns:
            Number of shares rounded down to lot size
        """
        if current_price <= 0:
            return 0

        raw_shares = int(target_value / current_price)
        # Round down to lot size
        return (raw_shares // lot_size) * lot_size
