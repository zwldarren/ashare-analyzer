"""
ConsoleRenderer - Rich terminal output renderer.

Renders Report objects to the terminal using Rich library for
beautiful table-based display.
"""

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from ashare_analyzer.constants import get_signal_emoji

from ..constants import (
    COL_AGENT,
    COL_CONFIDENCE,
    COL_DECISION,
    COL_POSITION,
    COL_REASON,
    COL_SIGNAL,
    COL_STOCK,
    COL_TREND,
    TEXT_DASHBOARD,
    TEXT_DECISION_REASON,
    TEXT_NO_RESULTS,
)
from ..models import Report, StockReport
from ..renderer_base import ReportRenderer


class ConsoleRenderer(ReportRenderer):
    """
    Rich console renderer for Report objects.

    Provides both string-based render methods (returns empty string)
    and side-effect display methods (prints to console).
    """

    def __init__(self, console: Console):
        """
        Initialize the renderer with a Rich console.

        Args:
            console: Rich Console instance for output
        """
        self._console = console

    def render_full(self, report: Report) -> str:
        """
        Render full report (returns empty string, actual output via display_full).

        Args:
            report: Report to render

        Returns:
            Empty string (actual output is via display_full side effect)
        """
        self.display_full(report)
        return ""

    def render_overview(self, report: Report) -> str:
        """
        Render overview table (returns empty string, actual output via display_overview).

        Args:
            report: Report to render

        Returns:
            Empty string (actual output is via display_overview side effect)
        """
        self.display_overview(report)
        return ""

    def render_single_stock(self, stock: StockReport) -> str:
        """
        Render single stock report (returns empty string, actual output via _display_stock_detail).

        Args:
            stock: StockReport to render

        Returns:
            Empty string (actual output is via _display_stock_detail side effect)
        """
        self._display_stock_detail(stock)
        return ""

    def _build_action_description(self, stock: StockReport) -> str:
        """Build human-readable action description.

        Args:
            stock: StockReport to describe

        Returns:
            Formatted action description string with Rich markup
        """
        if stock.action.upper() == "HOLD":
            if stock.has_position:
                return "持有当前仓位"
            return "观望（无持仓）"

        if stock.action.upper() == "SELL":
            if stock.has_position and stock.position_quantity > 0:
                if stock.action_quantity > 0:
                    # Calculate percentage of position being sold
                    pct = (stock.action_quantity / stock.position_quantity * 100) if stock.position_quantity > 0 else 0
                    # Determine action type
                    if stock.position_action == "close_position":
                        return f"[red]卖出 {stock.action_quantity} 股[/]（清仓，100%）"
                    elif stock.position_action == "reduce_position":
                        return f"[red]卖出 {stock.action_quantity} 股[/]（减仓{pct:.0f}%）"
                    else:
                        return f"[red]卖出 {stock.action_quantity} 股[/]（占持仓{pct:.0f}%）"
                else:
                    # No quantity specified, show general recommendation
                    return "[red]卖出[/]（建议减仓）"
            return "[red]卖出[/]（无持仓）"

        if stock.action.upper() == "BUY":
            if stock.action_quantity > 0:
                if stock.has_position:
                    return f"[green]买入 {stock.action_quantity} 股[/]（加仓）"
                return f"[green]买入 {stock.action_quantity} 股[/]（新开仓）"
            return "[green]买入[/]"

        return ""

    def display_full(self, report: Report) -> None:
        """
        Display full report to console (side effect: prints to console).

        Args:
            report: Report to display
        """
        if not report.stocks:
            self._console.print(f"[yellow]{TEXT_NO_RESULTS}[/yellow]")
            return

        self._console.print()
        self.display_overview(report)

        for stock in report.stocks:
            self._console.print()
            self._display_stock_detail(stock)

    def display_overview(self, report: Report) -> None:
        """
        Display overview table to console (side effect: prints to console).

        Args:
            report: Report to display overview for
        """
        table = Table(
            title=f"📊 {TEXT_DASHBOARD}",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            title_style="bold",
            expand=False,
        )
        table.add_column(COL_STOCK, style="white", no_wrap=True)
        table.add_column(COL_DECISION, justify="center", no_wrap=True)
        table.add_column(COL_CONFIDENCE, justify="right", no_wrap=True)
        table.add_column(COL_POSITION, justify="right", no_wrap=True)
        table.add_column(COL_TREND, no_wrap=True)

        for stock in report.stocks:
            emoji = get_signal_emoji(stock.action)
            position = f"{stock.position_ratio * 100:.0f}%" if stock.position_ratio > 0 else "-"
            trend = stock.trend_prediction or "-"
            action_style = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(stock.action.upper(), "white")

            table.add_row(
                f"{stock.name}({stock.code})",
                f"[{action_style}]{emoji} {stock.action.upper()}[/]",
                f"{stock.confidence}%",
                position,
                trend,
            )

        self._console.print(table)

    def _display_stock_detail(self, stock: StockReport) -> None:
        """
        Display single stock detail to console (internal method).

        Args:
            stock: StockReport to display
        """
        emoji = get_signal_emoji(stock.action)
        action_style = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(stock.action.upper(), "white")

        # Title with stock info
        position_str = f" | 仓位: {stock.position_ratio * 100:.0f}%" if stock.position_ratio > 0 else ""
        holding_str = ""
        if stock.has_position:
            if stock.position_cost_price > 0:
                holding_str = f" [dim](持有{stock.position_quantity}股@{stock.position_cost_price:.2f})[/]"
            else:
                holding_str = f" [dim](持有{stock.position_quantity}股)[/]"
        title_text = (
            f"{emoji} {stock.name} ({stock.code})"
            f" — [{action_style}]{stock.action.upper()}[/] | 置信度: {stock.confidence}%{position_str}{holding_str}"
        )
        self._console.print(Rule(title_text, style="dim"))

        # Trade action recommendation
        action_desc = self._build_action_description(stock)
        if action_desc:
            self._console.print(f"[bold]建议操作:[/] {action_desc}")

        # 决策理由
        if stock.decision_reasoning:
            self._console.print(f"[bold]{TEXT_DECISION_REASON}:[/] {stock.decision_reasoning}")

        # Agent consensus table
        if stock.agent_opinions:
            table = Table(
                show_header=True,
                header_style="bold dim",
                border_style="dim",
                expand=True,
                pad_edge=False,
            )
            table.add_column(COL_AGENT, style="cyan", width=20, no_wrap=True)
            table.add_column(COL_SIGNAL, justify="center", width=10, no_wrap=True)
            table.add_column(COL_CONFIDENCE, justify="right", width=8, no_wrap=True)
            table.add_column(COL_REASON, ratio=1, overflow="fold")

            for opinion in stock.agent_opinions:
                signal_emoji = get_signal_emoji(opinion.signal)
                signal_style = {"buy": "green", "sell": "red", "hold": "yellow"}.get(opinion.signal.lower(), "white")
                table.add_row(
                    opinion.name,
                    f"[{signal_style}]{signal_emoji} {opinion.signal.upper()}[/]",
                    f"{opinion.confidence}%",
                    opinion.reasoning,
                )

            self._console.print(table)

        # Key factors and risk warning
        extras = []
        if stock.key_factors:
            extras.append(f"🔑 关键因子: {', '.join(stock.key_factors[:3])}")
        if stock.risk_warning:
            extras.append(f"⚠️  风险: {stock.risk_warning}")

        if extras:
            self._console.print(" | ".join(extras))
