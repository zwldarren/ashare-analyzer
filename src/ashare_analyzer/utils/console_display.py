"""
Rich Console Display Module

Provides Rich-based terminal display for analysis progress and reports.
"""

from typing import TYPE_CHECKING

from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from ashare_analyzer.constants import get_signal_emoji
from ashare_analyzer.models import AnalysisResult

if TYPE_CHECKING:
    pass

from ashare_analyzer.utils.logging_config import (
    clear_live_display,
    get_console,
    set_live_display,
)

__all__ = ["RichConsoleDisplay"]


class RichConsoleDisplay:
    """
    Rich-based terminal display manager.
    """

    def __init__(self, console: Console | None = None):
        self.console = console or get_console()
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._stock_tasks: dict[str, TaskID] = {}
        self._stock_codes: list[str] = []

    def start_analysis(self, stock_codes: list[str]) -> None:
        """Initialize progress display for analysis using Live."""
        self._stock_codes = stock_codes

        # 创建进度条
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        )

        # 为每只股票创建任务
        for code in stock_codes:
            task_id = self._progress.add_task(f"○ {code} 等待中", total=None)
            self._stock_tasks[code] = task_id

        self._live = Live(
            self._progress,
            console=self.console,
            refresh_per_second=4,
            screen=False,
        )
        self._live.start()

        # 通知日志系统 Live 已启动
        set_live_display(self._live)

    def finish_analysis(self) -> None:
        """Stop progress display."""
        # 通知日志系统 Live 已停止
        clear_live_display()

        if self._live:
            self._live.stop()
            self._live = None

        if self._progress:
            self._progress.stop()
            self._progress = None

        self._stock_tasks.clear()
        self._stock_codes.clear()

    def update_stock_progress(self, code: str, status: str, name: str = "") -> None:
        """Update stock progress status.

        Args:
            code: Stock code
            status: One of 'waiting', 'analyzing', 'completed', 'error'
            name: Stock name (optional, for display)
        """
        if not self._progress or code not in self._stock_tasks:
            return

        task_id = self._stock_tasks[code]
        display_name = f"{name}({code})" if name else code

        status_map = {
            "waiting": ("○", "等待中", "dim"),
            "analyzing": ("◌", "分析中...", "cyan"),
            "completed": ("✓", "完成", "green"),
            "error": ("✗", "失败", "red"),
        }
        symbol, text, style = status_map.get(status, ("○", "", "white"))
        self._progress.update(task_id, description=f"[{style}]{symbol} {display_name} {text}[/{style}]")

    def start_agent(self, agent_name: str) -> None:
        """Display agent execution start - 简化版本不显示 Agent 进度."""
        pass

    def complete_agent(self, agent_name: str, signal: str, confidence: int, error: str | None = None) -> None:
        """Display agent completion with result - 简化版本不显示 Agent 进度."""
        pass

    def show_stock_result(self, result: AnalysisResult) -> None:
        """Display single stock analysis result summary."""
        action = result.final_action or "HOLD"
        emoji = get_signal_emoji(action)
        self.console.print(f"  {emoji} {result.name}({result.code}): {action} | 置信度 {result.sentiment_score}%")
