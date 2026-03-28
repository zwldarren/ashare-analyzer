"""Tests for notification base module."""

import pytest

from ashare_analyzer.notification.base import (
    NotificationChannel,
    NotificationChannelBase,
    get_channel_name,
)


class TestNotificationChannel:
    """Test NotificationChannel enum."""

    def test_channel_values(self):
        """Test that all channel types have correct values."""
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.TELEGRAM.value == "telegram"
        assert NotificationChannel.DISCORD.value == "discord"
        assert NotificationChannel.CUSTOM.value == "custom"
        assert NotificationChannel.UNKNOWN.value == "unknown"

    def test_all_channels_defined(self):
        """Test that all expected channels are defined."""
        channels = list(NotificationChannel)
        assert len(channels) == 5

    def test_channel_equality(self):
        """Test channel equality comparison."""
        assert NotificationChannel.EMAIL == NotificationChannel.EMAIL
        assert NotificationChannel.EMAIL != NotificationChannel.TELEGRAM


class TestGetChannelName:
    """Test get_channel_name utility function."""

    def test_get_channel_name_email(self):
        """Test getting name for email channel."""
        name = get_channel_name(NotificationChannel.EMAIL)
        assert name == "邮件"

    def test_get_channel_name_telegram(self):
        """Test getting name for telegram channel."""
        name = get_channel_name(NotificationChannel.TELEGRAM)
        assert name == "Telegram"

    def test_get_channel_name_discord(self):
        """Test getting name for discord channel."""
        name = get_channel_name(NotificationChannel.DISCORD)
        assert name == "Discord Webhook"

    def test_get_channel_name_custom(self):
        """Test getting name for custom channel."""
        name = get_channel_name(NotificationChannel.CUSTOM)
        assert name == "自定义Webhook"

    def test_get_channel_name_unknown(self):
        """Test getting name for unknown channel."""
        name = get_channel_name(NotificationChannel.UNKNOWN)
        assert name == "未知渠道"


class ConcreteNotificationChannel(NotificationChannelBase):
    """Concrete implementation for testing."""

    @property
    def channel_type(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    def _validate_config(self) -> None:
        if "required_key" not in self.config:
            raise ValueError("Missing required_key")

    def is_available(self) -> bool:
        return bool(self.config.get("required_key"))

    async def send(self, content: str, **kwargs) -> bool:
        return True


class TestNotificationChannelBase:
    """Test NotificationChannelBase abstract class."""

    def test_init_with_valid_config(self):
        """Test initialization with valid config."""
        config = {"required_key": "value"}
        channel = ConcreteNotificationChannel(config)
        assert channel.config == config

    def test_init_with_missing_required_config(self):
        """Test initialization raises error when validation fails."""
        config = {"wrong_key": "value"}
        with pytest.raises(ValueError, match="Missing required_key"):
            ConcreteNotificationChannel(config)

    def test_name_property(self):
        """Test name property returns correct channel name."""
        config = {"required_key": "value"}
        channel = ConcreteNotificationChannel(config)
        assert channel.name == "邮件"

    def test_is_available_true(self):
        """Test is_available returns True when configured."""
        config = {"required_key": "value"}
        channel = ConcreteNotificationChannel(config)
        assert channel.is_available() is True

    def test_is_available_false(self):
        """Test is_available returns False when not configured."""
        config = {"required_key": ""}
        channel = ConcreteNotificationChannel(config)
        assert channel.is_available() is False

    @pytest.mark.asyncio
    async def test_send_method(self):
        """Test send method works correctly."""
        config = {"required_key": "value"}
        channel = ConcreteNotificationChannel(config)
        result = await channel.send("test message")
        assert result is True


class TestNotificationChannelInheritance:
    """Test NotificationChannelBase inheritance patterns."""

    def test_multiple_channels_can_coexist(self):
        """Test that multiple channel instances can coexist."""

        class EmailChannel(NotificationChannelBase):
            @property
            def channel_type(self) -> NotificationChannel:
                return NotificationChannel.EMAIL

            def _validate_config(self) -> None:
                pass

            def is_available(self) -> bool:
                return "email_config" in self.config

            async def send(self, content: str, **kwargs) -> bool:
                return True

        class TelegramChannel(NotificationChannelBase):
            @property
            def channel_type(self) -> NotificationChannel:
                return NotificationChannel.TELEGRAM

            def _validate_config(self) -> None:
                pass

            def is_available(self) -> bool:
                return "telegram_config" in self.config

            async def send(self, content: str, **kwargs) -> bool:
                return True

        email = EmailChannel({"email_config": "value"})
        telegram = TelegramChannel({"telegram_config": "value"})

        assert email.channel_type == NotificationChannel.EMAIL
        assert telegram.channel_type == NotificationChannel.TELEGRAM
        assert email.is_available() is True
        assert telegram.is_available() is True

    def test_config_is_mutable(self):
        """Test that config can be modified after initialization."""
        config = {"required_key": "initial"}
        channel = ConcreteNotificationChannel(config)

        # Modify config
        channel.config["required_key"] = "modified"
        assert channel.config["required_key"] == "modified"

    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""

        # Attempting to instantiate without implementing abstract methods should fail
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            NotificationChannelBase({})
