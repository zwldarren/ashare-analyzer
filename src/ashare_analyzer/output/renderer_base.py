"""
Renderer base class for unified output system.

Defines the abstract interface that all renderers must implement.
"""

from abc import ABC, abstractmethod

from .models import Report, StockReport


class ReportRenderer(ABC):
    """
    Base class for report renderers.

    All renderers must implement the three core rendering methods:
    - render_full: Complete report
    - render_single_stock: Single stock detail
    - render_overview: Overview table section
    """

    @abstractmethod
    def render_full(self, report: Report) -> str:
        """
        Render complete report.

        Args:
            report: Report object to render

        Returns:
            Rendered string (format depends on renderer)
        """
        pass

    @abstractmethod
    def render_single_stock(self, stock: StockReport) -> str:
        """
        Render single stock report.

        Args:
            stock: StockReport object to render

        Returns:
            Rendered string (format depends on renderer)
        """
        pass

    @abstractmethod
    def render_overview(self, report: Report) -> str:
        """
        Render overview table.

        Args:
            report: Report object (uses report.stocks for overview)

        Returns:
            Rendered string (format depends on renderer)
        """
        pass
