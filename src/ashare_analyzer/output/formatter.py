"""统一输出格式化器"""

from typing import TYPE_CHECKING

from rich.console import Console

from ashare_analyzer.utils.logging_config import get_console

from .builder import ReportBuilder
from .models import Report, StockReport
from .renderers.console import ConsoleRenderer
from .renderers.markdown import MarkdownRenderer

if TYPE_CHECKING:
    from ashare_analyzer.models import AnalysisResult


class OutputFormatter:
    """统一输出格式化器"""

    def __init__(self, console: Console | None = None):
        self._console = console or get_console()
        self._console_renderer = ConsoleRenderer(self._console)
        self._markdown_renderer = MarkdownRenderer()
        self._builder = ReportBuilder()

    def build_report(self, results: list["AnalysisResult"], report_date: str | None = None) -> Report:
        """
        构建报告数据模型。

        Args:
            results: AnalysisResult 列表
            report_date: 报告日期，默认今天

        Returns:
            结构化的 Report 对象
        """
        return self._builder.build(results, report_date)

    def display_report(self, report: Report) -> None:
        """在终端显示完整报告（副作用：直接输出到控制台）"""
        self._console_renderer.display_full(report)

    def display_overview(self, report: Report) -> None:
        """在终端显示概览表格（副作用：直接输出到控制台）"""
        self._console_renderer.display_overview(report)

    def to_markdown(self, report: Report) -> str:
        """转换为 Markdown 文本（返回字符串，无副作用）"""
        return self._markdown_renderer.render_full(report)

    def to_single_stock_markdown(self, stock: StockReport) -> str:
        """转换单只股票为 Markdown（返回字符串，无副作用）"""
        return self._markdown_renderer.render_single_stock(stock)

    def to_overview_markdown(self, report: Report) -> str:
        """转换概览为 Markdown（返回字符串，无副作用）"""
        return self._markdown_renderer.render_overview(report)


def get_output_formatter(console: Console | None = None) -> OutputFormatter:
    """获取 OutputFormatter 实例（工厂函数）"""
    return OutputFormatter(console)
