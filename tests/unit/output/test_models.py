"""Tests for output models."""

from ashare_analyzer.output.models import AgentOpinion, MarketSnapshot, Report, ReportSummary, StockReport


class TestAgentOpinion:
    """Tests for AgentOpinion dataclass."""

    def test_agent_opinion_creation(self):
        """Test AgentOpinion should be created correctly."""
        opinion = AgentOpinion(
            name="TechnicalAgent",
            signal="buy",
            confidence=80,
            reasoning="Strong uptrend detected",
        )

        assert opinion.name == "TechnicalAgent"
        assert opinion.signal == "buy"
        assert opinion.confidence == 80
        assert opinion.reasoning == "Strong uptrend detected"

    def test_agent_opinion_with_different_signals(self):
        """Test AgentOpinion with different signal types."""
        sell_opinion = AgentOpinion(name="RiskAgent", signal="sell", confidence=60, reasoning="High risk")
        assert sell_opinion.signal == "sell"
        assert sell_opinion.confidence == 60

        hold_opinion = AgentOpinion(name="FundAgent", signal="hold", confidence=50, reasoning="Neutral")
        assert hold_opinion.signal == "hold"


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_market_snapshot_from_dict(self):
        """Test MarketSnapshot should be created from dict."""
        data = {
            "close": "1800.50",
            "prev_close": "1790.00",
            "open": "1795.00",
            "high": "1810.00",
            "low": "1785.00",
            "pct_chg": "+0.58%",
            "price": "1805.00",
            "volume_ratio": "1.25",
            "turnover_rate": "0.85%",
        }

        snapshot = MarketSnapshot.from_dict(data)

        assert snapshot is not None
        assert snapshot.close == "1800.50"
        assert snapshot.prev_close == "1790.00"
        assert snapshot.open == "1795.00"
        assert snapshot.high == "1810.00"
        assert snapshot.low == "1785.00"
        assert snapshot.pct_chg == "+0.58%"
        assert snapshot.price == "1805.00"
        assert snapshot.volume_ratio == "1.25"
        assert snapshot.turnover_rate == "0.85%"

    def test_market_snapshot_from_none(self):
        """Test MarketSnapshot.from_dict should return None for None input."""
        snapshot = MarketSnapshot.from_dict(None)
        assert snapshot is None

    def test_market_snapshot_from_empty_dict(self):
        """Test MarketSnapshot.from_dict with empty dict."""
        snapshot = MarketSnapshot.from_dict({})

        assert snapshot is not None
        assert snapshot.close == "N/A"
        assert snapshot.prev_close == "N/A"
        assert snapshot.open == "N/A"

    def test_market_snapshot_cleans_markdown(self):
        """Test MarketSnapshot should remove markdown formatting."""
        data = {
            "close": "**1800.50**",
            "prev_close": "*1790.00*",
            "open": "1795.00",
            "high": "**1810.00**",
            "low": "1785.00",
            "pct_chg": "+0.58%",
        }

        snapshot = MarketSnapshot.from_dict(data)

        assert snapshot is not None
        assert snapshot.close == "1800.50"
        assert snapshot.prev_close == "1790.00"
        assert snapshot.high == "1810.00"

    def test_market_snapshot_with_partial_data(self):
        """Test MarketSnapshot with only required fields."""
        data = {
            "close": "1800.50",
            "prev_close": "1790.00",
            "open": "1795.00",
            "high": "1810.00",
            "low": "1785.00",
            "pct_chg": "+0.58%",
        }

        snapshot = MarketSnapshot.from_dict(data)

        assert snapshot is not None
        assert snapshot.close == "1800.50"
        assert snapshot.price is None
        assert snapshot.volume_ratio is None
        assert snapshot.turnover_rate is None


class TestStockReport:
    """Tests for StockReport dataclass."""

    def test_stock_report_creation(self):
        """Test StockReport should be created correctly."""
        opinions = [
            AgentOpinion(name="TechAgent", signal="buy", confidence=80, reasoning="Uptrend"),
            AgentOpinion(name="RiskAgent", signal="hold", confidence=50, reasoning="Neutral"),
        ]

        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="buy",
            confidence=75,
            position_ratio=0.15,
            trend_prediction="上涨趋势",
            decision_reasoning="技术面向好",
            agent_opinions=opinions,
            consensus_level="high",
            key_factors=["MA金叉", "放量突破"],
            risk_warning="估值偏高",
            market_snapshot=None,
        )

        assert stock.code == "600519"
        assert stock.name == "贵州茅台"
        assert stock.action == "buy"
        assert stock.confidence == 75
        assert stock.position_ratio == 0.15
        assert stock.trend_prediction == "上涨趋势"
        assert stock.decision_reasoning == "技术面向好"
        assert len(stock.agent_opinions) == 2
        assert stock.consensus_level == "high"
        assert stock.key_factors == ["MA金叉", "放量突破"]
        assert stock.risk_warning == "估值偏高"
        assert stock.success is True
        assert stock.error_message is None

    def test_stock_report_with_error(self):
        """Test StockReport with error state."""
        stock = StockReport(
            code="600519",
            name="贵州茅台",
            action="hold",
            confidence=0,
            position_ratio=0.0,
            trend_prediction="",
            decision_reasoning="",
            agent_opinions=[],
            consensus_level="N/A",
            key_factors=[],
            risk_warning=None,
            market_snapshot=None,
            success=False,
            error_message="数据获取失败",
        )

        assert stock.success is False
        assert stock.error_message == "数据获取失败"


class TestReportSummary:
    """Tests for ReportSummary dataclass."""

    def test_report_summary_counts(self):
        """Test ReportSummary should count actions correctly."""
        summary = ReportSummary(total_count=5, buy_count=2, hold_count=2, sell_count=1)

        assert summary.total_count == 5
        assert summary.buy_count == 2
        assert summary.hold_count == 2
        assert summary.sell_count == 1

    def test_report_summary_zero_counts(self):
        """Test ReportSummary with zero counts."""
        summary = ReportSummary(total_count=0, buy_count=0, hold_count=0, sell_count=0)

        assert summary.total_count == 0
        assert summary.buy_count == 0
        assert summary.hold_count == 0
        assert summary.sell_count == 0


class TestReport:
    """Tests for Report dataclass."""

    def test_report_creation(self):
        """Test Report should be created with stocks and summary."""
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
                key_factors=["MA金叉"],
                risk_warning=None,
                market_snapshot=None,
            ),
            StockReport(
                code="000001",
                name="平安银行",
                action="hold",
                confidence=50,
                position_ratio=0.0,
                trend_prediction="震荡",
                decision_reasoning="观望",
                agent_opinions=[],
                consensus_level="low",
                key_factors=[],
                risk_warning=None,
                market_snapshot=None,
            ),
        ]

        summary = ReportSummary(total_count=2, buy_count=1, hold_count=1, sell_count=0)

        report = Report(report_date="2024-01-15", stocks=stocks, summary=summary)

        assert report.report_date == "2024-01-15"
        assert len(report.stocks) == 2
        assert report.summary.total_count == 2
        assert report.summary.buy_count == 1

    def test_report_empty_stocks(self):
        """Test Report with empty stocks list."""
        summary = ReportSummary(total_count=0, buy_count=0, hold_count=0, sell_count=0)
        report = Report(report_date="2024-01-15", stocks=[], summary=summary)

        assert len(report.stocks) == 0
        assert report.summary.total_count == 0
