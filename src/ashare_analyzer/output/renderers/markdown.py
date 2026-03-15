"""Markdown 渲染器"""

from __future__ import annotations

from datetime import datetime

from ashare_analyzer.constants import REPORT_EMOJI, get_signal_emoji

from ..constants import (
    TEXT_AGENT_CONSENSUS,
    TEXT_BUY,
    TEXT_CONSENSUS_LEVEL,
    TEXT_DASHBOARD,
    TEXT_DECISION_REASON,
    TEXT_DISCLAIMER,
    TEXT_HOLD,
    TEXT_KEY_FACTORS,
    TEXT_MARKET_SNAPSHOT,
    TEXT_REPORT_TIME,
    TEXT_RISK_WARNING,
    TEXT_SELL,
    TEXT_STOCK_SECTION,
    TEXT_SUGGESTED_POSITION,
    TEXT_TITLE,
    TEXT_TOTAL_STOCKS,
)
from ..models import MarketSnapshot, Report, StockReport
from ..renderer_base import ReportRenderer


class MarkdownRenderer(ReportRenderer):
    """Markdown 渲染器"""

    def render_full(self, report: Report) -> str:
        """渲染完整报告为 Markdown"""
        lines = [
            f"# {REPORT_EMOJI['title']} {report.report_date} {TEXT_TITLE}",
            "",
            f"> {TEXT_TOTAL_STOCKS} **{report.summary.total_count}** 只股票 | "
            f"{get_signal_emoji('BUY')}{TEXT_BUY}:{report.summary.buy_count} "
            f"{get_signal_emoji('HOLD')}{TEXT_HOLD}:{report.summary.hold_count} "
            f"{get_signal_emoji('SELL')}{TEXT_SELL}:{report.summary.sell_count}",
            "",
            "---",
            "",
        ]

        if report.stocks:
            lines.extend(self.render_overview(report).split("\n"))
            lines.extend(["", "---", ""])

        for stock in report.stocks:
            lines.extend(self._render_stock_section(stock))

        lines.extend(["", f"*{TEXT_REPORT_TIME}：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"])

        return "\n".join(lines)

    def render_overview(self, report: Report) -> str:
        """渲染概览部分为 Markdown"""
        lines = [f"## {REPORT_EMOJI['dashboard']} {TEXT_DASHBOARD}", ""]

        for stock in report.stocks:
            emoji = get_signal_emoji(stock.action)
            position_str = f" | 仓位{stock.position_ratio * 100:.0f}%" if stock.position_ratio > 0 else ""
            line = f"{emoji} **{stock.name}({stock.code})**: **{stock.action}** | "
            line += f"置信度{stock.confidence}%{position_str}"
            lines.append(line)

        return "\n".join(lines)

    def render_single_stock(self, stock: StockReport) -> str:
        """渲染单只股票为 Markdown"""
        lines = self._render_stock_section(stock)
        lines.extend(["---", f"*{TEXT_DISCLAIMER}*"])
        return "\n".join(lines)

    def _render_stock_section(self, stock: StockReport) -> list[str]:
        """渲染单只股票部分"""
        emoji = get_signal_emoji(stock.action)

        lines = [
            f"## {emoji} {stock.name} ({stock.code})",
            "",
            f"### {REPORT_EMOJI['title']} {TEXT_STOCK_SECTION}",
            "",
            f"**{emoji} {stock.action}** | 置信度: {stock.confidence}%",
            "",
        ]

        # 建议仓位
        if stock.position_ratio > 0:
            lines.append(f"{REPORT_EMOJI['money']} **{TEXT_SUGGESTED_POSITION}**: {stock.position_ratio * 100:.0f}%")
            lines.append("")

        # 决策理由
        if stock.decision_reasoning:
            lines.extend([f"**{TEXT_DECISION_REASON}**: {stock.decision_reasoning}", ""])

        # Agent 共识
        if stock.agent_opinions:
            lines.extend([f"### {REPORT_EMOJI['ai']} {TEXT_AGENT_CONSENSUS}", ""])
            lines.append("| Agent | 信号 | 置信度 | 原因 |")
            lines.append("|-------|------|--------|------|")

            for opinion in stock.agent_opinions:
                signal_emoji = get_signal_emoji(opinion.signal)
                row = f"| {opinion.name} | {signal_emoji} {opinion.signal.upper()} | "
                row += f"{opinion.confidence}% | {opinion.reasoning} |"
                lines.append(row)

            lines.append("")
            lines.append(f"**{TEXT_CONSENSUS_LEVEL}**: {stock.consensus_level}")
            lines.append("")

        # 行情快照
        if stock.market_snapshot:
            lines.extend(self._render_market_snapshot(stock.market_snapshot))

        # 关键因子
        if stock.key_factors:
            lines.append(f"🔑 **{TEXT_KEY_FACTORS}**: {', '.join(stock.key_factors)}")
            lines.append("")

        # 风险提示
        if stock.risk_warning:
            lines.extend([f"{REPORT_EMOJI['risk']} **{TEXT_RISK_WARNING}**: {stock.risk_warning}", ""])

        lines.extend(["---", ""])
        return lines

    def _render_market_snapshot(self, snapshot: MarketSnapshot) -> list[str]:
        """渲染行情快照"""
        lines = [
            f"### {REPORT_EMOJI['market']} {TEXT_MARKET_SNAPSHOT}",
            "",
            f"**收盘**: {snapshot.close} | **昨收**: {snapshot.prev_close} | **开盘**: {snapshot.open}",
            f"**最高**: {snapshot.high} | **最低**: {snapshot.low} | **涨跌幅**: {snapshot.pct_chg}",
            "",
        ]

        if snapshot.price:
            price_line = f"**当前价**: {snapshot.price} | **量比**: {snapshot.volume_ratio}"
            price_line += f" | **换手率**: {snapshot.turnover_rate}"
            lines.append(price_line)
            lines.append("")

        return lines
