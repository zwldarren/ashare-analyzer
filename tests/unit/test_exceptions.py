"""Tests for exception handling module."""

import pytest

from ashare_analyzer.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataFetchError,
    DataSourceUnavailableError,
    NotificationError,
    RateLimitError,
    StockAnalyzerException,
    StorageError,
    ValidationError,
    handle_errors,
    safe_execute,
)


class TestExceptionHierarchy:
    """Test exception hierarchy and inheritance."""

    def test_base_exception_with_code(self):
        """Test base exception with error code."""
        exc = StockAnalyzerException("Test error", code="TEST001")
        assert exc.message == "Test error"
        assert exc.code == "TEST001"
        assert str(exc) == "[TEST001] Test error"

    def test_base_exception_without_code(self):
        """Test base exception without error code."""
        exc = StockAnalyzerException("Test error")
        assert exc.message == "Test error"
        assert exc.code is None
        assert str(exc) == "Test error"

    def test_data_fetch_error_default_message(self):
        """Test DataFetchError with default message."""
        exc = DataFetchError()
        assert exc.message == "数据获取失败"
        assert isinstance(exc, StockAnalyzerException)

    def test_data_fetch_error_custom_message(self):
        """Test DataFetchError with custom message."""
        exc = DataFetchError("Custom fetch error", code="FETCH001")
        assert exc.message == "Custom fetch error"
        assert exc.code == "FETCH001"

    def test_rate_limit_error_default_message(self):
        """Test RateLimitError with default message."""
        exc = RateLimitError()
        assert "频率超限" in exc.message
        assert isinstance(exc, DataFetchError)
        assert isinstance(exc, StockAnalyzerException)

    def test_rate_limit_error_custom_message(self):
        """Test RateLimitError with custom message."""
        exc = RateLimitError("Too many requests", code="RATE001")
        assert exc.message == "Too many requests"
        assert exc.code == "RATE001"

    def test_data_source_unavailable_error(self):
        """Test DataSourceUnavailableError."""
        exc = DataSourceUnavailableError()
        assert "不可用" in exc.message
        assert isinstance(exc, DataFetchError)

    def test_storage_error(self):
        """Test StorageError."""
        exc = StorageError("Database connection failed")
        assert exc.message == "Database connection failed"
        assert isinstance(exc, StockAnalyzerException)

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid data format")
        assert exc.message == "Invalid data format"
        assert isinstance(exc, StockAnalyzerException)

    def test_analysis_error(self):
        """Test AnalysisError."""
        exc = AnalysisError("Analysis failed")
        assert exc.message == "Analysis failed"
        assert isinstance(exc, StockAnalyzerException)

    def test_notification_error(self):
        """Test NotificationError."""
        exc = NotificationError("Send failed")
        assert exc.message == "Send failed"
        assert isinstance(exc, StockAnalyzerException)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Missing config")
        assert exc.message == "Missing config"
        assert isinstance(exc, StockAnalyzerException)


class TestHandleErrorsDecorator:
    """Test handle_errors decorator for sync and async functions."""

    def test_sync_function_success(self):
        """Test decorator with successful sync function."""

        @handle_errors("Error occurred", default_return=None)
        def sync_func():
            return "success"

        result = sync_func()
        assert result == "success"

    def test_sync_function_exception_returns_default(self):
        """Test decorator returns default on exception."""

        @handle_errors("Error occurred", default_return="fallback", raise_on=())
        def sync_func():
            raise ValueError("Test error")

        result = sync_func()
        assert result == "fallback"

    def test_sync_function_reraise_specified_exceptions(self):
        """Test decorator re-raises specified exception types."""

        @handle_errors("Error occurred", default_return="fallback", raise_on=(ValueError,))
        def sync_func():
            raise ValueError("Should re-raise")

        with pytest.raises(ValueError, match="Should re-raise"):
            sync_func()

    def test_sync_function_default_raise_on(self):
        """Test that default raise_on=(Exception,) re-raises all exceptions."""

        @handle_errors("Error occurred", default_return="fallback")
        def sync_func():
            raise ValueError("Test error")

        # Default raise_on=(Exception,) means all exceptions are re-raised
        with pytest.raises(ValueError, match="Test error"):
            sync_func()

    def test_sync_function_log_levels(self, caplog):
        """Test different log levels in decorator."""
        import logging

        @handle_errors("Debug error", default_return=None, log_level="debug", raise_on=())
        def debug_func():
            raise RuntimeError("Debug test")

        @handle_errors("Warning error", default_return=None, log_level="warning", raise_on=())
        def warning_func():
            raise RuntimeError("Warning test")

        with caplog.at_level(logging.DEBUG):
            debug_func()
            assert "Debug error" in caplog.text

        caplog.clear()

        with caplog.at_level(logging.WARNING):
            warning_func()
            assert "Warning error" in caplog.text

    @pytest.mark.asyncio
    async def test_async_function_success(self):
        """Test decorator with successful async function."""

        @handle_errors("Error occurred", default_return=None)
        async def async_func():
            return "async success"

        result = await async_func()
        assert result == "async success"

    @pytest.mark.asyncio
    async def test_async_function_exception_returns_default(self):
        """Test decorator returns default on async exception."""

        @handle_errors("Error occurred", default_return="async fallback", raise_on=())
        async def async_func():
            raise ValueError("Async error")

        result = await async_func()
        assert result == "async fallback"

    @pytest.mark.asyncio
    async def test_async_function_reraise_specified_exceptions(self):
        """Test decorator re-raises specified exception types in async."""

        @handle_errors("Error occurred", default_return="fallback", raise_on=(ValueError,))
        async def async_func():
            raise ValueError("Should re-raise async")

        with pytest.raises(ValueError, match="Should re-raise async"):
            await async_func()

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @handle_errors("Error", default_return=None)
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ is not None
        assert "My docstring" in my_function.__doc__

    def test_decorator_with_args_and_kwargs(self):
        """Test decorator preserves function arguments."""

        @handle_errors("Error", default_return=None)
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = func_with_args("x", "y", c="z")
        assert result == "x-y-z"

    @pytest.mark.asyncio
    async def test_async_decorator_with_args_and_kwargs(self):
        """Test async decorator preserves function arguments."""

        @handle_errors("Error", default_return=None)
        async def async_func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"

        result = await async_func_with_args("x", "y", c="z")
        assert result == "x-y-z"


class TestSafeExecute:
    """Test safe_execute utility function."""

    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""

        def add(a, b):
            return a + b

        result = safe_execute(add, None, 1, 2)
        assert result == 3

    def test_safe_execute_with_exception(self):
        """Test safe_execute returns default on exception."""

        def raise_error():
            raise ValueError("Test error")

        result = safe_execute(raise_error, "default")
        assert result == "default"

    def test_safe_execute_with_kwargs(self):
        """Test safe_execute with keyword arguments."""

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = safe_execute(greet, None, "World", greeting="Hi")
        assert result == "Hi, World!"

    def test_safe_execute_none_default(self):
        """Test safe_execute with None as default."""

        def raise_error():
            raise RuntimeError("Error")

        result = safe_execute(raise_error, None)
        assert result is None

    def test_safe_execute_preserves_exception_info(self, caplog):
        """Test that safe_execute doesn't log (silent execution)."""
        import logging

        def raise_error():
            raise ValueError("Silent error")

        with caplog.at_level(logging.ERROR):
            result = safe_execute(raise_error, "fallback")
            assert result == "fallback"
            # safe_execute doesn't log, it just returns default
            assert "Silent error" not in caplog.text


class TestExceptionChaining:
    """Test exception chaining and context."""

    def test_exception_can_be_chained(self):
        """Test that exceptions can be chained."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DataFetchError("Fetch failed") from e
        except DataFetchError as exc:
            assert exc.message == "Fetch failed"
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ValueError)

    def test_exception_context(self):
        """Test exception context is preserved."""
        try:
            try:
                raise ValueError("Original")
            except ValueError as err:
                raise StorageError("Storage failed") from err
        except StorageError as exc:
            assert exc.message == "Storage failed"
            assert exc.__cause__ is not None
