"""Tests for output renderers."""

import pytest

from ashare_analyzer.output.models import AgentOpinion, MarketSnapshot, Report, ReportSummary, StockReport
from ashare_analyzer.output.renderers.markdown import MarkdownRenderer


def test_build_action_description_sell_with_quantity():
    """Test action description for sell with quantity."""
    from rich.console import Console

    from ashare_analyzer.output.renderers.console import ConsoleRenderer

    console = Console()
    renderer = ConsoleRenderer(console)

    stock = StockReport(
        code="600519",
        name="贵州茅台",
        action="SELL",
        confidence=75,
        position_ratio=0.0,
        action_quantity=500,
        position_action="reduce_position",
        trend_prediction="看空",
        decision_reasoning="技术面走弱",
        agent_opinions=[],
        consensus_level="50%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=True,
        position_quantity=1000,
        position_cost_price=1800.0,
    )

    desc = renderer._build_action_description(stock)
    assert "卖出 500 股" in desc
    assert "50%" in desc


def test_build_action_description_buy_with_quantity():
    """Test action description for buy with quantity."""
    from rich.console import Console

    from ashare_analyzer.output.renderers.console import ConsoleRenderer

    console = Console()
    renderer = ConsoleRenderer(console)

    stock = StockReport(
        code="600519",
        name="贵州茅台",
        action="BUY",
        confidence=85,
        position_ratio=0.15,
        action_quantity=800,
        position_action="open_position",
        trend_prediction="看多",
        decision_reasoning="突破阻力位",
        agent_opinions=[],
        consensus_level="85%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=False,
        position_quantity=0,
        position_cost_price=0.0,
    )

    desc = renderer._build_action_description(stock)
    assert "买入 800 股" in desc
    assert "新开仓" in desc


def test_render_stock_section_with_trade_quantity():
    """Test markdown rendering with trade quantity."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000338",
        name="潍柴动力",
        action="SELL",
        confidence=65,
        position_ratio=0.0,
        action_quantity=500,
        position_action="reduce_position",
        trend_prediction="看空",
        decision_reasoning="基本面恶化",
        agent_opinions=[],
        consensus_level="50%",
        key_factors=["趋势走弱"],
        risk_warning="风险等级: low",
        market_snapshot=None,
        has_position=True,
        position_quantity=1000,
        position_cost_price=24.55,
    )

    report = Report(
        report_date="2026-03-16",
        stocks=[stock],
        summary=ReportSummary(total_count=1, buy_count=0, hold_count=0, sell_count=1),
    )

    result = renderer.render_full(report)
    assert "卖出 500 股" in result
    assert "50%" in result


def test_build_action_description_md_hold_with_position():
    """Test markdown action description for hold with position."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="600519",
        name="贵州茅台",
        action="HOLD",
        confidence=60,
        position_ratio=0.0,
        action_quantity=0,
        position_action="keep_position",
        trend_prediction="震荡",
        decision_reasoning="观望",
        agent_opinions=[],
        consensus_level="40%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=True,
        position_quantity=500,
        position_cost_price=1800.0,
    )

    desc = renderer._build_action_description_md(stock)
    assert desc == "持有当前仓位"


def test_build_action_description_hold_no_position():
    """Test markdown action description for hold without position."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000001",
        name="平安银行",
        action="HOLD",
        confidence=50,
        position_ratio=0.0,
        action_quantity=0,
        position_action="keep_position",
        trend_prediction="震荡",
        decision_reasoning="观望",
        agent_opinions=[],
        consensus_level="N/A",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=False,
        position_quantity=0,
        position_cost_price=0.0,
    )

    desc = renderer._build_action_description_md(stock)
    assert desc == "观望（无持仓）"


def test_build_action_description_md_buy_new_position():
    """Test markdown action description for buy with new position."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000858",
        name="五粮液",
        action="BUY",
        confidence=75,
        position_ratio=0.1,
        action_quantity=800,
        position_action="open_position",
        trend_prediction="看多",
        decision_reasoning="突破阻力位",
        agent_opinions=[],
        consensus_level="75%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=False,
        position_quantity=0,
        position_cost_price=0.0,
    )

    desc = renderer._build_action_description_md(stock)
    assert desc == "买入 800 股（新开仓）"


def test_build_action_description_md_buy_add_position():
    """Test markdown action description for buy adding to position."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000858",
        name="五粮液",
        action="BUY",
        confidence=75,
        position_ratio=0.1,
        action_quantity=400,
        position_action="add_position",
        trend_prediction="看多",
        decision_reasoning="回调加仓",
        agent_opinions=[],
        consensus_level="75%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=True,
        position_quantity=500,
        position_cost_price=120.0,
    )

    desc = renderer._build_action_description_md(stock)
    assert desc == "买入 400 股（加仓）"


def test_render_stock_section_displays_action_description():
    """Test that action description appears in rendered output."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000888",
        name="峨眉山A",
        action="SELL",
        confidence=70,
        position_ratio=0.0,
        action_quantity=300,
        position_action="reduce_position",
        trend_prediction="看空",
        decision_reasoning="技术面恶化",
        agent_opinions=[],
        consensus_level="70%",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=True,
        position_quantity=600,
        position_cost_price=15.0,
    )

    result = renderer.render_single_stock(stock)
    assert "建议操作" in result
    assert "卖出 300 股" in result
    assert "50%" in result  # percentage of position


def test_render_stock_section_no_action_for_hold():
    """Test that action description is not shown for hold without position."""
    renderer = MarkdownRenderer()

    stock = StockReport(
        code="000001",
        name="平安银行",
        action="HOLD",
        confidence=50,
        position_ratio=0.0,
        action_quantity=0,
        position_action="keep_position",
        trend_prediction="震荡",
        decision_reasoning="观望",
        agent_opinions=[],
        consensus_level="N/A",
        key_factors=[],
        risk_warning=None,
        market_snapshot=None,
        has_position=False,
        position_quantity=0,
        position_cost_price=0.0,
    )

    result = renderer.render_single_stock(stock)
    # No action description for hold without position
    assert "建议操作" not in result


class TestMarkdownRenderer:
    """Tests for MarkdownRenderer class."""

    @pytest.fixture
    def renderer(self):
        """Create a MarkdownRenderer instance."""
        return MarkdownRenderer()

    @pytest.fixture
    def sample_stock(self):
        """Create a sample StockReport for testing."""
        return StockReport(
            code="600519",
            name="贵州茅台",
            action="buy",
            confidence=80,
            position_ratio=0.15,
            trend_prediction="上涨趋势",
            decision_reasoning="技术面向好，基本面优秀",
            agent_opinions=[
                AgentOpinion(name="TechnicalAgent", signal="buy", confidence=85, reasoning="MA金叉"),
                AgentOpinion(name="RiskAgent", signal="hold", confidence=50, reasoning="估值偏高"),
            ],
            consensus_level="high",
            key_factors=["MA金叉", "放量突破", "业绩超预期"],
            risk_warning="估值偏高，注意回调风险",
            market_snapshot=MarketSnapshot(
                close="1800.50",
                prev_close="1790.00",
                open="1795.00",
                high="1810.00",
                low="1785.00",
                pct_chg="+0.58%",
                price="1805.00",
                volume_ratio="1.25",
                turnover_rate="0.85%",
            ),
            success=True,
        )

    @pytest.fixture
    def sample_report(self, sample_stock):
        """Create a sample Report for testing."""
        return Report(
            report_date="2024-01-15",
            stocks=[sample_stock],
            summary=ReportSummary(total_count=1, buy_count=1, hold_count=0, sell_count=0),
        )

    def test_markdown_renderer_empty_report(self, renderer):
        """Test MarkdownRenderer should handle empty report."""
        report = Report(
            report_date="2024-01-15",
            stocks=[],
            summary=ReportSummary(total_count=0, buy_count=0, hold_count=0, sell_count=0),
        )

        output = renderer.render_full(report)

        assert "2024-01-15" in output
        assert "共分析" in output
        assert "买入:0" in output
        assert "持有:0" in output
        assert "卖出:0" in output

    def test_markdown_renderer_single_stock(self, renderer, sample_stock):
        """Test MarkdownRenderer should render single stock correctly."""
        output = renderer.render_single_stock(sample_stock)

        assert "贵州茅台" in output
        assert "600519" in output
        assert "buy" in output.lower() or "BUY" in output
        assert "80%" in output
        assert "15%" in output  # position ratio
        assert "---" in output  # separator

    def test_markdown_renderer_overview(self, renderer, sample_report):
        """Test MarkdownRenderer should render overview correctly."""
        output = renderer.render_overview(sample_report)

        assert "贵州茅台" in output
        assert "600519" in output
        assert "buy" in output  # Action is lowercase

    def test_markdown_renderer_market_snapshot(self, renderer, sample_stock):
        """Test MarkdownRenderer should render market snapshot."""
        output = renderer.render_single_stock(sample_stock)

        assert "1800.50" in output  # close price
        assert "1790.00" in output  # prev_close
        assert "+0.58%" in output  # pct_chg
        assert "1.25" in output  # volume_ratio

    def test_markdown_renderer_full_report(self, renderer, sample_report):
        """Test MarkdownRenderer should render full report correctly."""
        output = renderer.render_full(sample_report)

        # Check title
        assert "2024-01-15" in output
        assert "AI投资决策报告" in output

        # Check summary
        assert "共分析" in output
        assert "买入:1" in output

        # Check stock section
        assert "贵州茅台" in output
        assert "600519" in output

        # Check agent opinions
        assert "TechnicalAgent" in output
        assert "RiskAgent" in output

        # Check key factors
        assert "MA金叉" in output

        # Check risk warning
        assert "估值偏高" in output

    def test_markdown_renderer_multiple_stocks(self, renderer):
        """Test MarkdownRenderer should render multiple stocks."""
        stocks = [
            StockReport(
                code="600519",
                name="贵州茅台",
                action="buy",
                confidence=80,
                position_ratio=0.15,
                trend_prediction="上涨",
                decision_reasoning="看好",
                agent_opinions=[],
                consensus_level="high",
                key_factors=["Factor1"],
                risk_warning=None,
                market_snapshot=None,
            ),
            StockReport(
                code="000001",
                name="平安银行",
                action="sell",
                confidence=70,
                position_ratio=0.0,
                trend_prediction="下跌",
                decision_reasoning="看空",
                agent_opinions=[],
                consensus_level="low",
                key_factors=[],
                risk_warning="风险较大",
                market_snapshot=None,
            ),
        ]

        report = Report(
            report_date="2024-01-15",
            stocks=stocks,
            summary=ReportSummary(total_count=2, buy_count=1, hold_count=0, sell_count=1),
        )

        output = renderer.render_full(report)

        assert "贵州茅台" in output
        assert "平安银行" in output
        assert "买入:1" in output
        assert "卖出:1" in output

    def test_markdown_renderer_agent_opinions_table(self, renderer, sample_stock):
        """Test that agent opinions are rendered as table."""
        output = renderer.render_single_stock(sample_stock)

        assert "| Agent | 信号 | 置信度 | 原因 |" in output
        assert "|-------|------|--------|------|" in output
        assert "| TechnicalAgent |" in output
        assert "| RiskAgent |" in output

    def test_markdown_renderer_without_position_ratio(self, renderer):
        """Test rendering stock with zero position ratio."""
        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="hold",
            confidence=50,
            position_ratio=0.0,
            trend_prediction="震荡",
            decision_reasoning="观望",
            agent_opinions=[],
            consensus_level="N/A",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        # Position ratio should not appear in output when 0
        assert "仓位" not in output or "0%" not in output

    def test_markdown_renderer_without_agent_opinions(self, renderer):
        """Test rendering stock without agent opinions."""
        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="hold",
            confidence=50,
            position_ratio=0.0,
            trend_prediction="",
            decision_reasoning="",
            agent_opinions=[],
            consensus_level="N/A",
            key_factors=["Factor1"],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        # Should not contain agent consensus section
        assert "Agent 共识分析" not in output
        assert "| Agent |" not in output

    def test_markdown_renderer_without_risk_warning(self, renderer):
        """Test rendering stock without risk warning."""
        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="buy",
            confidence=80,
            position_ratio=0.1,
            trend_prediction="上涨",
            decision_reasoning="看好",
            agent_opinions=[],
            consensus_level="high",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        # Should not contain risk warning section
        assert "风险提示" not in output

    def test_markdown_renderer_report_time(self, renderer, sample_report):
        """Test that report time is included in output."""
        output = renderer.render_full(sample_report)

        assert "报告生成时间" in output


class TestMarkdownRendererEdgeCases:
    """Edge case tests for MarkdownRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create a MarkdownRenderer instance."""
        return MarkdownRenderer()

    def test_renderer_with_special_characters_in_name(self, renderer):
        """Test rendering stock with special characters in name."""
        stock = StockReport(
            code="600519",
            name='测试&<>"股票',
            action="buy",
            confidence=80,
            position_ratio=0.1,
            trend_prediction="上涨",
            decision_reasoning="测试",
            agent_opinions=[],
            consensus_level="high",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        assert '测试&<>"股票' in output

    def test_renderer_with_long_decision_reasoning(self, renderer):
        """Test rendering stock with long decision reasoning."""
        long_reasoning = (
            "这是一个非常长的决策理由，包含了多个方面的分析，包括技术面、基本面、市场情绪等多个维度的综合考量。"
        )

        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="buy",
            confidence=80,
            position_ratio=0.1,
            trend_prediction="上涨",
            decision_reasoning=long_reasoning,
            agent_opinions=[],
            consensus_level="high",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        assert long_reasoning in output

    def test_renderer_with_many_key_factors(self, renderer):
        """Test rendering stock with many key factors."""
        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="buy",
            confidence=80,
            position_ratio=0.1,
            trend_prediction="上涨",
            decision_reasoning="看好",
            agent_opinions=[],
            consensus_level="high",
            key_factors=["Factor1", "Factor2", "Factor3", "Factor4"],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        # Key factors should be joined with comma
        assert "Factor1" in output
        assert "Factor2" in output

    def test_renderer_with_empty_strings(self, renderer):
        """Test rendering stock with empty string values."""
        stock = StockReport(
            code="600519",
            name="",
            action="hold",
            confidence=0,
            position_ratio=0.0,
            trend_prediction="",
            decision_reasoning="",
            agent_opinions=[],
            consensus_level="",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
        )

        output = renderer.render_single_stock(stock)

        # Should still render without error
        assert "600519" in output
