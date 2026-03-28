"""
LLM 日志模块 - 专门记录 LLM 请求和响应

用于调试 LLM 调用问题，记录：
- 请求内容（prompt, tools, parameters）
- 响应内容（tool_calls, content, usage）
- 错误信息
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# LLM 调试日志专用 logger
_llm_logger: logging.Logger | None = None
_llm_log_file: Path | None = None


def setup_llm_logger(log_dir: str = "./logs") -> logging.Logger:
    """
    初始化 LLM 调试日志记录器。

    创建独立的日志文件，专门记录 LLM 请求和响应详情。

    Args:
        log_dir: 日志文件保存目录

    Returns:
        配置好的 Logger 实例
    """
    global _llm_logger, _llm_log_file

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _llm_log_file = log_path / f"llm_debug_{timestamp}.log"

    # 创建独立的 logger
    _llm_logger = logging.getLogger("llm_debug")
    _llm_logger.setLevel(logging.DEBUG)
    _llm_logger.handlers.clear()
    _llm_logger.propagate = False  # 不传播到父 logger

    # 文件 Handler
    file_handler = logging.FileHandler(_llm_log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _llm_logger.addHandler(file_handler)

    logger.info(f"LLM 调试日志已启用: {_llm_log_file}")
    return _llm_logger


def get_llm_logger() -> logging.Logger | None:
    """获取 LLM 调试日志记录器"""
    return _llm_logger


def get_llm_log_file() -> Path | None:
    """获取 LLM 日志文件路径"""
    return _llm_log_file


def _format_json(obj: Any, indent: int = 2) -> str:
    """格式化 JSON 对象为可读字符串"""
    try:
        return json.dumps(obj, ensure_ascii=False, indent=indent, default=str)
    except (TypeError, ValueError):
        return str(obj)


def log_llm_request(
    agent_name: str,
    model: str,
    prompt: str,
    tool: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    generation_config: dict[str, Any] | None = None,
) -> None:
    """
    记录 LLM 请求内容。

    Args:
        agent_name: 调用的 Agent 名称
        model: 使用的模型名称
        prompt: 用户提示词
        tool: Function call schema（可选）
        system_prompt: 系统提示词（可选）
        generation_config: 生成配置（可选）
    """
    if not _llm_logger:
        return

    _llm_logger.debug("=" * 80)
    _llm_logger.debug(f"[LLM REQUEST] Agent: {agent_name}")
    _llm_logger.debug(f"Model: {model}")
    _llm_logger.debug("-" * 40)

    # 记录系统提示词（截断超长内容）
    if system_prompt:
        sp = system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt
        _llm_logger.debug(f"System Prompt:\n{sp}")
        _llm_logger.debug("-" * 40)

    # 记录用户提示词（截断超长内容）
    p = prompt[:2000] + "..." if len(prompt) > 2000 else prompt
    _llm_logger.debug(f"User Prompt:\n{p}")

    # 记录工具定义
    if tool:
        _llm_logger.debug("-" * 40)
        _llm_logger.debug(f"Tool Schema:\n{_format_json(tool)}")

    # 记录生成配置（隐藏敏感信息）
    if generation_config:
        safe_config = {
            k: v for k, v in generation_config.items()
            if k not in ("api_key", "api_base")
        }
        _llm_logger.debug("-" * 40)
        _llm_logger.debug(f"Generation Config:\n{_format_json(safe_config)}")

    _llm_logger.debug("=" * 80)


def log_llm_response(
    agent_name: str,
    model: str,
    response: Any,
    parsed_result: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> None:
    """
    记录 LLM 响应内容。

    Args:
        agent_name: 调用的 Agent 名称
        model: 使用的模型名称
        response: 原始响应对象
        parsed_result: 解析后的结果（可选）
        duration_ms: 调用耗时（毫秒）
    """
    if not _llm_logger:
        return

    _llm_logger.debug("=" * 80)
    _llm_logger.debug(f"[LLM RESPONSE] Agent: {agent_name}")
    _llm_logger.debug(f"Model: {model}")

    if duration_ms:
        _llm_logger.debug(f"Duration: {duration_ms:.0f}ms")

    _llm_logger.debug("-" * 40)

    # 记录原始响应关键信息
    if response and hasattr(response, "choices") and response.choices:
        choice = response.choices[0]

        # 记录 tool_calls
        if choice.message.tool_calls:
            _llm_logger.debug("Tool Calls:")
            for i, tc in enumerate(choice.message.tool_calls):
                _llm_logger.debug(f"  [{i}] Function: {tc.function.name}")
                _llm_logger.debug(f"      Arguments:\n{tc.function.arguments}")

        # 记录文本内容
        if choice.message.content:
            content_len = len(choice.message.content)
            content = choice.message.content[:1000] + "..." if content_len > 1000 else choice.message.content
            _llm_logger.debug(f"Content:\n{content}")

        # 记录 finish_reason
        if choice.finish_reason:
            _llm_logger.debug(f"Finish Reason: {choice.finish_reason}")

        # 记录 usage
        if hasattr(response, "usage") and response.usage:
            _llm_logger.debug(
                f"Usage: prompt_tokens={response.usage.prompt_tokens}, "
                f"completion_tokens={response.usage.completion_tokens}, "
                f"total_tokens={response.usage.total_tokens}"
            )
    else:
        _llm_logger.debug("Response: (empty or None)")

    # 记录解析后的结果
    if parsed_result:
        _llm_logger.debug("-" * 40)
        _llm_logger.debug(f"Parsed Result:\n{_format_json(parsed_result)}")

    _llm_logger.debug("=" * 80)


def log_llm_error(
    agent_name: str,
    model: str,
    error: Exception,
    request_info: dict[str, Any] | None = None,
) -> None:
    """
    记录 LLM 错误信息。

    Args:
        agent_name: 调用的 Agent 名称
        model: 使用的模型名称
        error: 异常对象
        request_info: 请求相关信息（可选）
    """
    if not _llm_logger:
        return

    _llm_logger.debug("=" * 80)
    _llm_logger.debug(f"[LLM ERROR] Agent: {agent_name}")
    _llm_logger.debug(f"Model: {model}")
    _llm_logger.debug("-" * 40)

    # 记录错误详情
    _llm_logger.debug(f"Error Type: {type(error).__name__}")
    _llm_logger.debug(f"Error Message: {str(error)}")

    # 记录请求信息
    if request_info:
        # 隐藏敏感信息
        safe_info = {
            k: v for k, v in request_info.items()
            if k not in ("api_key", "api_base")
        }
        _llm_logger.debug(f"Request Info:\n{_format_json(safe_info)}")

    _llm_logger.debug("=" * 80)


def log_agent_result(
    agent_name: str,
    stock_code: str,
    signal: str,
    confidence: int,
    reasoning: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    记录 Agent 分析结果。

    Args:
        agent_name: Agent 名称
        stock_code: 股票代码
        signal: 信号类型 (buy/sell/hold)
        confidence: 置信度
        reasoning: 分析理由
        metadata: 额外元数据（可选）
    """
    if not _llm_logger:
        return

    _llm_logger.debug("=" * 80)
    _llm_logger.debug(f"[AGENT RESULT] {agent_name}")
    _llm_logger.debug(f"Stock: {stock_code}")
    _llm_logger.debug(f"Signal: {signal} | Confidence: {confidence}%")
    _llm_logger.debug(f"Reasoning: {reasoning[:500]}..." if len(reasoning) > 500 else f"Reasoning: {reasoning}")

    if metadata:
        _llm_logger.debug(f"Metadata:\n{_format_json(metadata)}")

    _llm_logger.debug("=" * 80)


__all__ = [
    "setup_llm_logger",
    "get_llm_logger",
    "get_llm_log_file",
    "log_llm_request",
    "log_llm_response",
    "log_llm_error",
    "log_agent_result",
]