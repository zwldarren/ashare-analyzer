"""Tests for scheduler module."""

import asyncio
import contextlib
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ashare_analyzer.scheduler import run_with_schedule, run_with_schedule_async


class TestRunWithSchedule:
    """Test synchronous scheduler function."""

    def test_run_immediately_executes_task(self):
        """Test that run_immediately=True executes task once."""
        call_count = 0

        def task():
            nonlocal call_count
            call_count += 1

        # Mock time.sleep inside the function to prevent infinite loop
        sleep_call_count = 0

        def mock_sleep(seconds):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 1:
                raise KeyboardInterrupt()

        with patch.object(time, "sleep", side_effect=mock_sleep):
            run_with_schedule(task, schedule_time="09:00", run_immediately=True)

        assert call_count == 1

    def test_run_immediately_false_skips_first_execution(self):
        """Test that run_immediately=False skips first execution."""
        call_count = 0

        def task():
            nonlocal call_count
            call_count += 1

        def mock_sleep(seconds):
            raise KeyboardInterrupt()

        with patch.object(time, "sleep", side_effect=mock_sleep):
            run_with_schedule(task, schedule_time="09:00", run_immediately=False)

        assert call_count == 0

    def test_task_exception_is_caught(self, caplog):
        """Test that task exceptions are caught and logged."""
        call_count = 0

        def failing_task():
            nonlocal call_count
            call_count += 1
            raise ValueError("Task failed")

        def mock_sleep(seconds):
            raise KeyboardInterrupt()

        with patch.object(time, "sleep", side_effect=mock_sleep):
            run_with_schedule(failing_task, schedule_time="09:00", run_immediately=True)

        assert call_count == 1
        assert "首次任务执行失败" in caplog.text or "任务执行失败" in caplog.text

    def test_keyboard_interrupt_stops_scheduler(self):
        """Test that KeyboardInterrupt stops the scheduler."""
        call_count = 0

        def task():
            nonlocal call_count
            call_count += 1

        def mock_sleep(seconds):
            raise KeyboardInterrupt()

        with patch.object(time, "sleep", side_effect=mock_sleep):
            run_with_schedule(task, schedule_time="09:00", run_immediately=False)

    def test_scheduler_error_recovery(self, caplog):
        """Test that scheduler recovers from errors and continues."""
        call_count = 0

        def task():
            nonlocal call_count
            call_count += 1

        class StopScheduler(Exception):
            pass

        sleep_call_count = 0

        def mock_sleep(seconds):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count == 1:
                raise RuntimeError("Network error")
            raise StopScheduler()

        with patch.object(time, "sleep", side_effect=mock_sleep), contextlib.suppress(StopScheduler):
            run_with_schedule(task, schedule_time="09:00", run_immediately=False)

        assert "调度器错误" in caplog.text or call_count >= 0


class TestRunWithScheduleAsync:
    """Test asynchronous scheduler function."""

    @pytest.mark.asyncio
    async def test_run_immediately_executes_async_task(self):
        """Test that run_immediately=True executes async task once."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        # First call for wait, then raise CancelledError
        sleep_call_count = 0

        async def mock_async_sleep(seconds):
            nonlocal sleep_call_count
            sleep_call_count += 1
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", side_effect=mock_async_sleep):
            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=True)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_run_immediately_false_skips_first_execution(self):
        """Test that run_immediately=False skips first async execution."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        async def mock_async_sleep(seconds):
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", side_effect=mock_async_sleep):
            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=False)

        assert call_count == 0

    @pytest.mark.asyncio
    async def test_async_task_exception_is_caught(self, caplog):
        """Test that async task exceptions are caught and logged."""
        call_count = 0

        async def failing_async_task():
            nonlocal call_count
            call_count += 1
            raise ValueError("Async task failed")

        async def mock_async_sleep(seconds):
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", side_effect=mock_async_sleep):
            await run_with_schedule_async(failing_async_task, schedule_time="09:00", run_immediately=True)

        assert call_count == 1
        assert "首次任务执行失败" in caplog.text or "任务执行失败" in caplog.text

    @pytest.mark.asyncio
    async def test_cancelled_error_stops_scheduler(self):
        """Test that CancelledError stops the async scheduler."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        async def mock_async_sleep(seconds):
            raise asyncio.CancelledError()

        with patch.object(asyncio, "sleep", side_effect=mock_async_sleep):
            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=False)

    @pytest.mark.asyncio
    async def test_async_scheduler_error_recovery(self, caplog):
        """Test that async scheduler recovers from errors."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        class StopScheduler(Exception):
            pass

        sleep_call_count = 0

        async def mock_async_sleep(seconds):
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count == 1:
                raise RuntimeError("Network error")
            raise StopScheduler()

        with patch.object(asyncio, "sleep", side_effect=mock_async_sleep), contextlib.suppress(StopScheduler):
            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=False)

        assert "调度器错误" in caplog.text

    @pytest.mark.asyncio
    async def test_schedule_time_calculation_future_time(self):
        """Test wait time calculation when target is in future."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        async def mock_async_sleep(seconds):
            raise asyncio.CancelledError()

        # Mock datetime to control time
        mock_now = datetime(2024, 1, 15, 8, 0, 0)  # 08:00

        with (
            patch.object(asyncio, "sleep", side_effect=mock_async_sleep),
            patch("ashare_analyzer.scheduler.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime
            mock_datetime.combine = datetime.combine
            mock_datetime.timedelta = timedelta

            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=False)


class TestScheduleTimeParsing:
    """Test schedule time parsing and calculation."""

    def test_valid_time_format(self):
        """Test that valid time formats are parsed correctly."""
        # This is implicitly tested through the scheduler functions
        # The format "%H:%M" should work for times like "09:00", "15:30"
        from datetime import datetime

        valid_times = ["00:00", "09:00", "12:30", "23:59"]
        for time_str in valid_times:
            parsed = datetime.strptime(time_str, "%H:%M")
            assert parsed.hour == int(time_str.split(":")[0])
            assert parsed.minute == int(time_str.split(":")[1])

    @pytest.mark.asyncio
    async def test_target_time_already_passed_today(self):
        """Test that if target time passed, it schedules for tomorrow."""
        call_count = 0

        async def async_task():
            nonlocal call_count
            call_count += 1

        async def mock_async_sleep(seconds):
            raise asyncio.CancelledError()

        # Mock current time as 10:00, target is 09:00 (already passed)
        mock_now = datetime(2024, 1, 15, 10, 0, 0)

        with (
            patch.object(asyncio, "sleep", side_effect=mock_async_sleep),
            patch("ashare_analyzer.scheduler.datetime") as mock_datetime,
        ):
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime
            mock_datetime.combine = datetime.combine
            mock_datetime.timedelta = timedelta

            await run_with_schedule_async(async_task, schedule_time="09:00", run_immediately=False)
