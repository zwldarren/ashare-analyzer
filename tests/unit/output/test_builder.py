"""Tests for ReportBuilder."""

import pytest

from ashare_analyzer.models import AnalysisResult
from ashare_analyzer.output.builder import ReportBuilder
from ashare_analyzer.output.models import Report


class TestReportBuilder:
    """Tests for ReportBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a ReportBuilder instance."""
        return ReportBuilder()

    @pytest.fixture
    def sample_result(self):
        """Create a sample AnalysisResult for testing."""
        return AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="上涨趋势",
            operation_advice="逢低买入",
            decision_type="buy",
            confidence_level="高",
            final_action="buy",
            position_ratio=0.15,
            decision_reasoning="技术面向好，基本面优秀",
            dashboard={
                "agent_reports": {
                    "signals": {"TechnicalAgent": "buy", "RiskAgent": "hold"},
                    "confidences": {"TechnicalAgent": 80, "RiskAgent": 50},
                    "reasonings": {
                        "TechnicalAgent": "MA金叉，量价配合",
                        "RiskAgent": "估值偏高，风险中性",
                    },
                    "consensus_level": "high",
                },
                "key_considerations": ["MA金叉", "放量突破", "业绩超预期"],
            },
            market_snapshot={
                "close": "1800.50",
                "prev_close": "1790.00",
                "open": "1795.00",
                "high": "1810.00",
                "low": "1785.00",
                "pct_chg": "+0.58%",
            },
            risk_warning="估值偏高，注意回调风险",
            success=True,
        )

    def test_build_empty_results(self, builder):
        """Test building report with empty results."""
        report = builder.build([])

        assert isinstance(report, Report)
        assert report.report_date  # Should have a date
        assert len(report.stocks) == 0
        assert report.summary.total_count == 0
        assert report.summary.buy_count == 0
        assert report.summary.hold_count == 0
        assert report.summary.sell_count == 0

    def test_build_single_result(self, builder, sample_result):
        """Test building report with single result."""
        report = builder.build([sample_result])

        assert len(report.stocks) == 1
        assert report.summary.total_count == 1

        stock = report.stocks[0]
        assert stock.code == "600519"
        assert stock.name == "贵州茅台"
        assert stock.action == "buy"
        assert stock.confidence == 75
        assert stock.position_ratio == 0.15
        assert stock.trend_prediction == "上涨趋势"
        assert stock.decision_reasoning == "技术面向好，基本面优秀"
        assert len(stock.agent_opinions) == 2
        assert stock.consensus_level == "high"
        assert len(stock.key_factors) == 3
        assert stock.risk_warning == "估值偏高，注意回调风险"

    def test_build_multiple_results_sorted(self, builder):
        """Test that results are sorted by confidence descending."""
        results = [
            AnalysisResult(
                code="600519",
                name="Stock A",
                sentiment_score=50,
                trend_prediction="",
                operation_advice="",
            ),
            AnalysisResult(
                code="000001",
                name="Stock B",
                sentiment_score=90,
                trend_prediction="",
                operation_advice="",
            ),
            AnalysisResult(
                code="000002",
                name="Stock C",
                sentiment_score=70,
                trend_prediction="",
                operation_advice="",
            ),
        ]

        report = builder.build(results)

        assert len(report.stocks) == 3
        # Should be sorted by sentiment_score descending
        assert report.stocks[0].code == "000001"  # 90
        assert report.stocks[1].code == "000002"  # 70
        assert report.stocks[2].code == "600519"  # 50

    def test_build_handles_default_name(self, builder):
        """Test that builder handles missing/placeholder names."""
        result = AnalysisResult(
            code="600519",
            name="",  # Empty name
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
        )

        report = builder.build([result])

        assert report.stocks[0].name == "股票600519"

    def test_build_handles_placeholder_name(self, builder):
        """Test that builder handles placeholder names."""
        result = AnalysisResult(
            code="600519",
            name="股票",  # Placeholder name
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
        )

        report = builder.build([result])

        assert report.stocks[0].name == "股票600519"

    def test_build_handles_none_dashboard(self, builder):
        """Test that builder handles None dashboard gracefully."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            dashboard=None,
        )

        report = builder.build([result])

        assert len(report.stocks[0].agent_opinions) == 0
        assert report.stocks[0].consensus_level == "N/A"
        assert len(report.stocks[0].key_factors) == 0

    def test_build_handles_empty_dashboard(self, builder):
        """Test that builder handles empty dashboard gracefully."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            dashboard={},
        )

        report = builder.build([result])

        assert len(report.stocks[0].agent_opinions) == 0

    def test_build_counts_actions_correctly(self, builder):
        """Test that builder counts BUY/HOLD/SELL correctly."""
        results = [
            AnalysisResult(
                code="001", name="A", sentiment_score=80, decision_type="buy", trend_prediction="", operation_advice=""
            ),
            AnalysisResult(
                code="002", name="B", sentiment_score=70, decision_type="buy", trend_prediction="", operation_advice=""
            ),
            AnalysisResult(
                code="003", name="C", sentiment_score=60, decision_type="hold", trend_prediction="", operation_advice=""
            ),
            AnalysisResult(
                code="004", name="D", sentiment_score=50, decision_type="sell", trend_prediction="", operation_advice=""
            ),
        ]

        report = builder.build(results)

        assert report.summary.total_count == 4
        assert report.summary.buy_count == 2
        assert report.summary.hold_count == 1
        assert report.summary.sell_count == 1

    def test_build_uses_final_action_over_decision_type(self, builder):
        """Test that builder uses final_action when available."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            decision_type="hold",
            final_action="buy",
        )

        report = builder.build([result])

        assert report.stocks[0].action == "buy"

    def test_build_custom_report_date(self, builder, sample_result):
        """Test that builder uses custom report date."""
        custom_date = "2024-12-25"
        report = builder.build([sample_result], report_date=custom_date)

        assert report.report_date == custom_date

    def test_build_with_market_snapshot(self, builder, sample_result):
        """Test that builder correctly processes market snapshot."""
        report = builder.build([sample_result])

        assert report.stocks[0].market_snapshot is not None
        assert report.stocks[0].market_snapshot.close == "1800.50"
        assert report.stocks[0].market_snapshot.prev_close == "1790.00"

    def test_build_without_market_snapshot(self, builder):
        """Test that builder handles missing market snapshot."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            market_snapshot=None,
        )

        report = builder.build([result])

        assert report.stocks[0].market_snapshot is None

    def test_build_agent_opinions(self, builder, sample_result):
        """Test that builder correctly extracts agent opinions."""
        report = builder.build([sample_result])

        opinions = report.stocks[0].agent_opinions
        assert len(opinions) == 2

        # Check first opinion
        tech_opinion = next((o for o in opinions if o.name == "TechnicalAgent"), None)
        assert tech_opinion is not None
        assert tech_opinion.signal == "buy"
        assert tech_opinion.confidence == 80
        assert tech_opinion.reasoning == "MA金叉，量价配合"

        # Check second opinion
        risk_opinion = next((o for o in opinions if o.name == "RiskAgent"), None)
        assert risk_opinion is not None
        assert risk_opinion.signal == "hold"
        assert risk_opinion.confidence == 50

    def test_build_limits_key_factors(self, builder):
        """Test that builder limits key_factors to 3 items."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            dashboard={
                "key_considerations": ["Factor1", "Factor2", "Factor3", "Factor4", "Factor5"],
            },
        )

        report = builder.build([result])

        assert len(report.stocks[0].key_factors) == 3

    def test_build_handles_string_confidence(self, builder):
        """Test that builder handles string confidence values."""
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=75,
            trend_prediction="",
            operation_advice="",
            dashboard={
                "agent_reports": {
                    "signals": {"Agent1": "buy"},
                    "confidences": {"Agent1": "85"},  # String instead of int
                    "reasonings": {"Agent1": "Test"},
                },
            },
        )

        report = builder.build([result])

        assert report.stocks[0].agent_opinions[0].confidence == 85
