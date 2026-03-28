"""
通知渠道基类和枚举定义
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """通知渠道类型"""

    EMAIL = "email"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


CHANNEL_NAMES: dict[NotificationChannel, str] = {
    NotificationChannel.EMAIL: "邮件",
    NotificationChannel.TELEGRAM: "Telegram",
    NotificationChannel.DISCORD: "Discord Webhook",
    NotificationChannel.CUSTOM: "自定义Webhook",
    NotificationChannel.UNKNOWN: "未知渠道",
}


def get_channel_name(channel: NotificationChannel) -> str:
    """Get Chinese name for notification channel."""
    return CHANNEL_NAMES.get(channel, "未知渠道")


class NotificationChannelBase(ABC):
    """
    通知渠道抽象基类

    所有具体通知渠道必须继承此类并实现 send 方法
    """

    MAX_MESSAGE_LENGTH = 4096

    def __init__(self, config: dict[str, Any]):
        """
        初始化通知渠道

        Args:
            config: 渠道配置字典
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """验证配置是否完整"""

    @abstractmethod
    def is_available(self) -> bool:
        """检查渠道是否可用（配置完整）"""

    @abstractmethod
    async def send(self, content: str, **kwargs: Any) -> bool:
        """
        发送消息（异步）

        Args:
            content: 消息内容
            **kwargs: 额外参数

        Returns:
            是否发送成功
        """

    @property
    @abstractmethod
    def channel_type(self) -> NotificationChannel:
        """返回渠道类型"""

    @property
    def name(self) -> str:
        """返回渠道名称"""
        return get_channel_name(self.channel_type)

    async def send_chunked(
        self,
        content: str,
        max_length: int | None = None,
        separator: str = "\n---\n",
    ) -> bool:
        """
        Send content in chunks if it exceeds max length.

        Args:
            content: Content to send
            max_length: Maximum length per message (default: MAX_MESSAGE_LENGTH)
            separator: Separator to split content

        Returns:
            True if all chunks sent successfully
        """
        max_length = max_length or self.MAX_MESSAGE_LENGTH

        if len(content) <= max_length:
            return await self.send(content)

        sections = content.split(separator)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for section in sections:
            section_length = len(section) + len(separator)

            if current_length + section_length > max_length:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                current_chunk = [section]
                current_length = section_length
            else:
                current_chunk.append(section)
                current_length += section_length

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        all_success = True
        for chunk in chunks:
            if not await self.send(chunk):
                all_success = False

        return all_success
