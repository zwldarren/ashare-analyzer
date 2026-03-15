"""输出系统数据模型"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentOpinion:
    """单个 Agent 的分析意见"""

    name: str
    signal: str
    confidence: int
    reasoning: str


@dataclass(slots=True)
class MarketSnapshot:
    """行情快照"""

    close: str
    prev_close: str
    open: str
    high: str
    low: str
    pct_chg: str
    price: str | None = None
    volume_ratio: str | None = None
    turnover_rate: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MarketSnapshot | None":
        """从字典创建，处理 None 和格式化"""
        if data is None:
            return None

        def clean(val: Any) -> str:
            return str(val).replace("**", "").replace("*", "") if val else "N/A"

        return cls(
            close=clean(data.get("close")),
            prev_close=clean(data.get("prev_close")),
            open=clean(data.get("open")),
            high=clean(data.get("high")),
            low=clean(data.get("low")),
            pct_chg=clean(data.get("pct_chg")),
            price=clean(data.get("price")) if data.get("price") else None,
            volume_ratio=clean(data.get("volume_ratio")) if data.get("volume_ratio") else None,
            turnover_rate=clean(data.get("turnover_rate")) if data.get("turnover_rate") else None,
        )


@dataclass(slots=True)
class StockReport:
    """单只股票的报告"""

    code: str
    name: str
    action: str
    confidence: int
    position_ratio: float
    trend_prediction: str
    decision_reasoning: str
    agent_opinions: list[AgentOpinion]
    consensus_level: str
    key_factors: list[str]
    risk_warning: str | None
    market_snapshot: MarketSnapshot | None
    success: bool = True
    error_message: str | None = None
    data_sources: str = ""


@dataclass(slots=True)
class ReportSummary:
    """报告统计摘要"""

    total_count: int
    buy_count: int
    hold_count: int
    sell_count: int


@dataclass(slots=True)
class Report:
    """完整报告"""

    report_date: str
    stocks: list[StockReport]
    summary: ReportSummary
