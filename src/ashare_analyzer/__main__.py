"""
===================================
A股自选股智能分析系统 - 主调度程序 (Simplified)
===================================

使用方式：
    python -m ashare_analyzer              # 正常运行
    python -m ashare_analyzer --debug      # 调试模式
    python -m ashare_analyzer --dry-run    # 仅获取数据不分析
"""

import asyncio
import atexit
import logging
import os
import sys
import warnings
from datetime import datetime

import click

from ashare_analyzer.analysis import batch_analyze
from ashare_analyzer.config import Config, check_config_valid, get_config, get_config_safe
from ashare_analyzer.dependencies import get_data_manager
from ashare_analyzer.infrastructure import aiohttp_session_manager
from ashare_analyzer.notification import get_notification_service
from ashare_analyzer.output import get_output_formatter
from ashare_analyzer.utils import get_console
from ashare_analyzer.utils.logging_config import setup_logging
from ashare_analyzer.utils.stock_code import StockType, detect_stock_type

logger = logging.getLogger(__name__)

# Suppress warnings from external libraries before importing them
warnings.filterwarnings("ignore", message="invalid escape sequence", category=SyntaxWarning)
warnings.filterwarnings("ignore", message="enable_cleanup_closed", category=DeprecationWarning)

# Suppress tqdm progress bars from efinance library
os.environ.setdefault("TQDM_DISABLE", "1")


def _cleanup_resources() -> None:
    pass


atexit.register(_cleanup_resources)


def _print_banner() -> None:
    """使用 Rich 输出启动横幅，替代 logger.info 避免格式混乱"""
    console = get_console()
    console.print()
    console.rule("[bold cyan]A股自选股智能分析系统[/bold cyan]", style="cyan")
    console.print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
    console.print()


def _print_analysis_header(stock_codes: list[str], max_workers: int, dry_run: bool) -> None:
    """使用 Rich 输出分析配置摘要"""
    console = get_console()
    mode = "仅获取数据" if dry_run else "完整分析"
    console.print(
        f"📋 分析 [bold]{len(stock_codes)}[/bold] 只股票: [cyan]{', '.join(stock_codes)}[/cyan]"
        f"  (并发: {max_workers}, 模式: {mode})"
    )
    console.print()


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="启用调试模式，输出详细日志")
@click.option("--dry-run", is_flag=True, help="仅获取数据，不进行 AI 分析")
@click.option("--stocks", type=str, help="指定要分析的股票代码，逗号分隔（覆盖配置文件）")
@click.option("--no-notify", is_flag=True, help="不发送推送通知")
@click.option(
    "--single-notify",
    is_flag=True,
    help="启用单股推送模式：每分析完一只股票立即推送，而不是汇总推送",
)
@click.option("--workers", type=int, default=None, help="并发线程数（默认使用配置值）")
@click.option("--schedule", is_flag=True, help="启用定时任务模式，每日定时执行")
@click.option("--api-key", type=str, help="AI 模型 API 密钥 (LLM_API_KEY)")
@click.option("--base-url", type=str, help="AI 模型 API 基础 URL (LLM_BASE_URL)")
@click.option("--model", type=str, help="AI 模型名称 (LLM_MODEL)")
@click.pass_context
def cli(
    ctx: click.Context,
    debug: bool,
    dry_run: bool,
    stocks: str | None,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    schedule: bool,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> int:
    """A股自选股智能分析系统"""
    if ctx.invoked_subcommand is None:
        return asyncio.run(
            run_main_async(
                debug,
                dry_run,
                stocks,
                no_notify,
                single_notify,
                workers,
                schedule,
                api_key,
                base_url,
                model,
            )
        )
    return 0


def _apply_cli_overrides(
    config: Config,
    stocks: str | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> None:
    """Apply CLI options to Config instance directly."""
    if stocks:
        valid_codes = []
        for code in stocks.split(","):
            code = code.strip()
            if code:
                stock_type = detect_stock_type(code)
                if stock_type != StockType.UNKNOWN:
                    valid_codes.append(code)
                else:
                    logger.warning(f"无效的股票代码格式: {code}，已跳过")
        if valid_codes:
            object.__setattr__(config, "stock_list_str", ",".join(valid_codes))

    if api_key:
        config.ai.llm_api_key = api_key
    if base_url:
        config.ai.llm_base_url = base_url
    if model:
        config.ai.llm_model = model


async def run_main_async(
    debug: bool,
    dry_run: bool,
    stocks: str | None,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    schedule: bool,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> int:
    """Async main program."""
    config, errors = get_config_safe()
    is_valid, missing = check_config_valid(config)

    if not is_valid:
        from rich.console import Console

        console = Console()
        console.print("\n[bold yellow]⚠️ 未检测到有效配置[/bold yellow]")
        console.print("\n[dim]缺少以下必需配置:[/dim]")
        for item in missing:
            console.print(f"  - {item}")
        console.print("\n[dim]请运行以下命令完成初始化:[/dim]")
        console.print("  [bold cyan]ashare-analyzer init[/bold cyan]")
        return 1

    config = get_config()

    _apply_cli_overrides(config, stocks, api_key, base_url, model)

    if os.getenv("GITHUB_ACTIONS") != "true":
        if config.system.http_proxy:
            os.environ["http_proxy"] = config.system.http_proxy
            logger.debug(f"已设置 http_proxy: {config.system.http_proxy}")
        if config.system.https_proxy:
            os.environ["https_proxy"] = config.system.https_proxy
            logger.debug(f"已设置 https_proxy: {config.system.https_proxy}")

    effective_debug = debug or config.system.debug
    setup_logging(debug=effective_debug, log_dir=config.log_dir)

    _print_banner()

    warnings = config.validate_config()
    for warning in warnings:
        logger.warning(warning)

    stock_codes = config.stock_list if stocks else None

    try:
        async with aiohttp_session_manager():
            if schedule or config.schedule.schedule_enabled:
                logger.info("模式: 定时任务")
                logger.info(f"每日执行时间: {config.schedule.schedule_time}")

                from ashare_analyzer.scheduler import run_with_schedule_async

                async def scheduled_task():
                    await run_full_analysis_async(
                        config,
                        stock_codes,
                        dry_run,
                        no_notify,
                        single_notify,
                        workers,
                        debug,
                    )

                await run_with_schedule_async(
                    task=scheduled_task,
                    schedule_time=config.schedule.schedule_time,
                    run_immediately=True,
                )
                return 0

            await run_full_analysis_async(
                config,
                stock_codes,
                dry_run,
                no_notify,
                single_notify,
                workers,
                debug,
            )

            console = get_console()
            console.print()
            console.rule("[dim]执行完成[/dim]", style="dim")
            console.print()

            return 0

    except KeyboardInterrupt:
        logger.info("用户中断，程序退出")
        return 130

    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        return 1


async def run_full_analysis_async(
    config: Config,
    stock_codes: list[str] | None,
    dry_run: bool,
    no_notify: bool,
    single_notify: bool,
    workers: int | None,
    debug: bool = False,
) -> list:
    """
    执行完整的分析流程 (Async version)
    """
    try:
        if single_notify:
            config.notification_message.single_stock_notify = True

        if stock_codes is None:
            config.refresh_stock_list()
            stock_codes = config.stock_list

        if not stock_codes:
            logger.error("未配置自选股列表，请在 .env 文件中设置 STOCK_LIST")
            return []

        max_workers = workers or config.system.max_workers

        _print_analysis_header(stock_codes, max_workers, dry_run)

        await _prefetch_realtime_quotes_async(stock_codes)

        results = []

        if dry_run:
            results = await _run_dry_mode_async(stock_codes, max_workers)
        else:
            results = await _run_analysis_mode_async(
                stock_codes,
                config,
                max_workers,
                no_notify,
            )

        if results:
            formatter = get_output_formatter()
            report = formatter.build_report(results)
            formatter.display_report(report)

        return results

    except Exception as e:
        logger.exception(f"分析流程执行失败: {e}")
        return []


async def _prefetch_realtime_quotes_async(stock_codes: list[str]) -> None:
    """批量预取实时行情数据以优化性能 (async)"""
    try:
        data_manager = get_data_manager()
        prefetch_count = await data_manager.prefetch_realtime_quotes(stock_codes)
        if prefetch_count > 0:
            logger.debug(f"已启用批量预取架构：一次拉取全市场数据，{len(stock_codes)} 只股票共享缓存")
    except Exception as e:
        logger.debug(f"批量预取实时行情失败: {e}")


async def _run_dry_mode_async(stock_codes: list[str], max_workers: int) -> list:
    """dry_run模式：仅获取数据，不进行分析 (async)"""
    logger.info("Dry-run模式：仅获取数据")

    data_manager = get_data_manager()

    semaphore = asyncio.Semaphore(max_workers)

    async def fetch_one(code: str):
        async with semaphore:
            try:
                daily_data, source = await data_manager.get_daily_data(code, 90)
                if daily_data is not None and not daily_data.empty:
                    logger.info(f"[{code}] 数据获取成功: {len(daily_data)} 条")
                else:
                    logger.warning(f"[{code}] 数据获取失败")
            except Exception as e:
                logger.error(f"[{code}] 任务执行失败: {e}")

    await asyncio.gather(*[fetch_one(code) for code in stock_codes])

    return []


async def _run_analysis_mode_async(
    stock_codes: list[str],
    config: Config,
    max_workers: int,
    no_notify: bool,
) -> list:
    """正常分析模式 (async)"""
    results = await batch_analyze(
        stock_codes=stock_codes,
        max_workers=max_workers,
    )

    if results and not no_notify:
        await _send_notifications_async(results, config)

    return results


async def _send_notifications_async(results: list, config: Config) -> None:
    """发送分析结果通知 (async)"""
    try:
        notifier = get_notification_service()

        logger.info("生成决策仪表盘日报...")
        report = notifier.generate_dashboard_report(results)
        filepath = notifier.save_report_to_file(report)
        logger.info(f"决策仪表盘日报已保存: {filepath}")

        if not notifier.is_available():
            logger.info("通知渠道未配置，跳过推送")
            return

        await _send_to_channels_async(notifier, report, results)

    except Exception as e:
        logger.error(f"发送通知失败: {e}")


async def _send_to_channels_async(notifier, report: str, results: list) -> None:
    """发送报告到各个通知渠道 (async)"""
    context_success = await notifier.send_to_context(report)

    success = await notifier.send(report) or context_success

    if success:
        logger.info("决策仪表盘推送成功")
    else:
        logger.warning("决策仪表盘推送失败")


def main() -> int:
    """程序主入口"""
    return cli()


if __name__ == "__main__":
    sys.exit(main())
