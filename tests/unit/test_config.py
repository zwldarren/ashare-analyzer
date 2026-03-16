"""Unit tests for TOML configuration functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ashare_analyzer.config import (
    LIST_FIELDS,
    _flatten_toml_to_env,
    _get_nested_value,
    _load_toml_config,
    _serialize_toml_value,
    get_config,
)
from ashare_analyzer.exceptions import ConfigurationError


class TestSerializeTomlValue:
    """Tests for _serialize_toml_value function."""

    def test_string_serialization(self):
        """Test string value serialization."""
        assert _serialize_toml_value("hello", "TEST_VAR") == "hello"
        assert _serialize_toml_value("gpt-4o", "LLM_MODEL") == "gpt-4o"

    def test_none_returns_empty_string(self):
        """Test None value returns empty string."""
        assert _serialize_toml_value(None, "TEST_VAR") == ""

    def test_bool_true(self):
        """Test boolean true serialization."""
        assert _serialize_toml_value(True, "DEBUG") == "true"
        assert _serialize_toml_value(True, "SCHEDULE_ENABLED") == "true"

    def test_bool_false(self):
        """Test boolean false serialization."""
        assert _serialize_toml_value(False, "DEBUG") == "false"
        assert _serialize_toml_value(False, "SCHEDULE_ENABLED") == "false"

    def test_integer_serialization(self):
        """Test integer value serialization."""
        assert _serialize_toml_value(42, "MAX_WORKERS") == "42"
        assert _serialize_toml_value(0, "TEST_VAR") == "0"
        assert _serialize_toml_value(-5, "TEST_VAR") == "-5"

    def test_float_serialization(self):
        """Test float value serialization."""
        assert _serialize_toml_value(0.7, "LLM_TEMPERATURE") == "0.7"
        assert _serialize_toml_value(2.0, "LLM_REQUEST_DELAY") == "2.0"
        assert _serialize_toml_value(3.14159, "TEST_VAR") == "3.14159"

    def test_list_comma_joined_for_list_fields(self):
        """Test list serialization as comma-joined for LIST_FIELDS."""
        assert _serialize_toml_value(["key1", "key2", "key3"], "BOCHA_API_KEYS") == "key1,key2,key3"
        assert _serialize_toml_value(["600519", "000858"], "STOCK_LIST") == "600519,000858"
        assert (
            _serialize_toml_value(["email1@test.com", "email2@test.com"], "EMAIL_RECEIVERS")
            == "email1@test.com,email2@test.com"
        )

    def test_list_str_for_non_list_fields(self):
        """Test list serialization as str() for non-LIST_FIELDS."""
        result = _serialize_toml_value(["a", "b"], "SOME_OTHER_VAR")
        assert result == "['a', 'b']"

    def test_list_fields_constant(self):
        """Test LIST_FIELDS contains expected fields."""
        assert "BOCHA_API_KEYS" in LIST_FIELDS
        assert "TAVILY_API_KEYS" in LIST_FIELDS
        assert "BRAVE_API_KEYS" in LIST_FIELDS
        assert "SERPAPI_API_KEYS" in LIST_FIELDS
        assert "EMAIL_RECEIVERS" in LIST_FIELDS
        assert "CUSTOM_WEBHOOK_URLS" in LIST_FIELDS
        assert "STOCK_LIST" in LIST_FIELDS


class TestGetNestedValue:
    """Tests for _get_nested_value function."""

    def test_single_key_access(self):
        """Test accessing value with single key."""
        data = {"name": "test", "value": 42}
        assert _get_nested_value(data, ("name",)) == "test"
        assert _get_nested_value(data, ("value",)) == 42

    def test_nested_key_access(self):
        """Test accessing nested dict values."""
        data = {"ai": {"llm_model": "gpt-4o", "temperature": 0.7}}
        assert _get_nested_value(data, ("ai", "llm_model")) == "gpt-4o"
        assert _get_nested_value(data, ("ai", "temperature")) == 0.7

    def test_deeply_nested_access(self):
        """Test accessing deeply nested values."""
        data = {"notification": {"email": {"sender": "test@test.com", "password": "secret"}}}
        assert _get_nested_value(data, ("notification", "email", "sender")) == "test@test.com"
        assert _get_nested_value(data, ("notification", "email", "password")) == "secret"

    def test_missing_key_returns_none(self):
        """Test missing key returns None."""
        data = {"ai": {"model": "gpt-4o"}}
        assert _get_nested_value(data, ("ai", "missing_key")) is None
        assert _get_nested_value(data, ("missing", "key")) is None

    def test_non_dict_returns_none(self):
        """Test non-dict intermediate value returns None."""
        data = {"ai": "not a dict"}
        assert _get_nested_value(data, ("ai", "model")) is None

    def test_empty_tuple_returns_data(self):
        """Test empty tuple returns the data itself."""
        data = {"key": "value"}
        assert _get_nested_value(data, ()) == data

    def test_none_data_returns_none(self):
        """Test None data returns None."""
        assert _get_nested_value(None, ("key",)) is None  # type: ignore[arg-type]


class TestFlattenTomlToEnv:
    """Tests for _flatten_toml_to_env function."""

    def test_simple_config_flattening(self):
        """Test flattening simple config values."""
        toml_config = {
            "ai": {"llm_model": "gpt-4o", "llm_temperature": 0.7},
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["LLM_MODEL"] == "gpt-4o"
        assert result["LLM_TEMPERATURE"] == "0.7"

    def test_nested_config_fallback(self):
        """Test flattening nested config (ai.fallback)."""
        toml_config = {
            "ai": {
                "fallback": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "fallback-key",
                    "base_url": "https://fallback.api",
                },
            },
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["LLM_FALLBACK_MODEL"] == "gpt-3.5-turbo"
        assert result["LLM_FALLBACK_API_KEY"] == "fallback-key"
        assert result["LLM_FALLBACK_BASE_URL"] == "https://fallback.api"

    def test_list_as_comma_string(self):
        """Test list values are converted to comma-separated strings."""
        toml_config = {
            "search": {"bocha_api_keys": ["key1", "key2", "key3"]},
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["BOCHA_API_KEYS"] == "key1,key2,key3"

    def test_stock_list(self):
        """Test stock list flattening."""
        toml_config = {
            "stock_list": ["600519", "000858", "000001"],
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["STOCK_LIST"] == "600519,000858,000001"

    def test_bool_values(self):
        """Test boolean values are serialized correctly."""
        toml_config = {
            "system": {"debug": True},
            "schedule": {"enabled": False},
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["DEBUG"] == "true"
        assert result["SCHEDULE_ENABLED"] == "false"

    def test_skips_none_values(self):
        """Test None values are skipped."""
        toml_config = {
            "ai": {"llm_model": "gpt-4o", "llm_api_key": None},
        }
        result = _flatten_toml_to_env(toml_config)
        assert "LLM_MODEL" in result
        assert "LLM_API_KEY" not in result

    def test_empty_config(self):
        """Test empty config returns empty dict."""
        result = _flatten_toml_to_env({})
        assert result == {}

    def test_notification_config(self):
        """Test notification configuration flattening."""
        toml_config = {
            "notification": {
                "email": {"sender": "test@test.com", "password": "secret"},
                "telegram": {"bot_token": "token123", "chat_id": "12345"},
            },
        }
        result = _flatten_toml_to_env(toml_config)
        assert result["EMAIL_SENDER"] == "test@test.com"
        assert result["EMAIL_PASSWORD"] == "secret"
        assert result["TELEGRAM_BOT_TOKEN"] == "token123"
        assert result["TELEGRAM_CHAT_ID"] == "12345"


class TestLoadTomlConfig:
    """Tests for _load_toml_config function."""

    def test_nonexistent_file_returns_empty_dict(self):
        """Test nonexistent file returns empty dict."""
        with patch("ashare_analyzer.config.Path") as mock_path:
            mock_home = mock_path.home.return_value
            mock_config_path = mock_home.__truediv__.return_value.__truediv__.return_value
            mock_config_path.exists.return_value = False
            result = _load_toml_config()
            assert result == {}

    def test_valid_toml_file_loads_correctly(self):
        """Test valid TOML file loads correctly."""
        toml_content = """
[ai]
llm_model = "gpt-4o"
llm_temperature = 0.7

[system]
debug = true
max_workers = 5

[stock_list]
items = ["600519", "000858"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            temp_path = Path(f.name)

        try:
            with patch("ashare_analyzer.config.Path") as mock_path:
                mock_home = mock_path.home.return_value
                mock_config_path = mock_home.__truediv__.return_value.__truediv__.return_value
                mock_config_path.exists.return_value = True
                mock_config_path.__fspath__.return_value = str(temp_path)

                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = toml_content.encode()
                    import tomllib

                    expected = tomllib.loads(toml_content)
                    with patch("tomllib.load", return_value=expected):
                        result = _load_toml_config()
                        assert "ai" in result
        finally:
            temp_path.unlink(missing_ok=True)

    def test_malformed_toml_raises_configuration_error(self):
        """Test malformed TOML raises ConfigurationError."""
        with patch("ashare_analyzer.config.Path") as mock_path:
            mock_home = mock_path.home.return_value
            mock_config_path = mock_home.__truediv__.return_value.__truediv__.return_value
            mock_config_path.exists.return_value = True

            import tomllib

            with (
                patch("builtins.open", create=True),
                patch("tomllib.load", side_effect=tomllib.TOMLDecodeError("test error", "", 0)),
                pytest.raises(ConfigurationError) as exc_info,
            ):
                _load_toml_config()
            assert "Failed to parse config file" in str(exc_info.value)


class TestGetConfigMergePriority:
    """Tests for configuration merge priority in get_config."""

    def test_environment_variables_override_toml(self, monkeypatch):
        """Test environment variables take precedence over TOML values."""
        get_config.cache_clear()
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("LLM_API_KEY", "env-api-key")

        toml_config = {
            "ai": {"llm_model": "toml-model", "llm_api_key": "toml-api-key"},
        }

        with patch("ashare_analyzer.config._load_toml_config", return_value=toml_config):
            config = get_config()

        assert config.ai.llm_model == "env-model"
        assert config.ai.llm_api_key == "env-api-key"

    def test_no_toml_uses_env_only(self, monkeypatch):
        """Test environment only when no TOML config exists."""
        get_config.cache_clear()
        monkeypatch.setenv("LLM_MODEL", "env-only-model")
        monkeypatch.setenv("DEBUG", "true")

        with patch("ashare_analyzer.config._load_toml_config", return_value={}):
            config = get_config()

        assert config.ai.llm_model == "env-only-model"

    def test_stock_list_from_toml(self, monkeypatch):
        """Test stock list can be loaded from TOML."""
        get_config.cache_clear()
        monkeypatch.delenv("STOCK_LIST", raising=False)

        toml_config = {
            "stock_list": ["600519", "000858", "000001"],
        }

        with (
            patch("ashare_analyzer.config._load_toml_config", return_value=toml_config),
            patch("ashare_analyzer.config.load_dotenv"),
        ):
            config = get_config()

        assert "600519" in config.stock_list
        assert "000858" in config.stock_list

    def test_cache_clear_and_reload(self, monkeypatch):
        """Test cache clear allows config reload."""
        get_config.cache_clear()
        monkeypatch.setenv("LLM_MODEL", "first-model")

        with patch("ashare_analyzer.config._load_toml_config", return_value={}):
            config1 = get_config()
        assert config1.ai.llm_model == "first-model"

        get_config.cache_clear()
        monkeypatch.setenv("LLM_MODEL", "second-model")

        with patch("ashare_analyzer.config._load_toml_config", return_value={}):
            config2 = get_config()
        assert config2.ai.llm_model == "second-model"

    def test_merge_logic_toml_env_priority(self):
        """Test the merge logic: TOML values are base, env vars override.

        This tests the core merge logic in get_config():
        merged_env = {**toml_env, **current_env}
        """
        toml_config = {
            "ai": {"llm_model": "toml-model", "llm_temperature": 0.5},
            "system": {"max_workers": 10},
        }
        toml_env = _flatten_toml_to_env(toml_config)

        assert toml_env["LLM_MODEL"] == "toml-model"
        assert toml_env["LLM_TEMPERATURE"] == "0.5"
        assert toml_env["MAX_WORKERS"] == "10"

        current_env = {"LLM_MODEL": "env-model"}
        merged = {**toml_env, **current_env}

        assert merged["LLM_MODEL"] == "env-model"
        assert merged["LLM_TEMPERATURE"] == "0.5"
        assert merged["MAX_WORKERS"] == "10"

    def test_env_override_toml_temperature(self, monkeypatch):
        """Test that env var overrides TOML for llm_temperature."""
        get_config.cache_clear()
        monkeypatch.setenv("LLM_TEMPERATURE", "0.9")

        toml_config = {
            "ai": {"llm_temperature": 0.3},
        }

        with patch("ashare_analyzer.config._load_toml_config", return_value=toml_config):
            config = get_config()

        assert config.ai.llm_temperature == 0.9


class TestTomlEnvMappings:
    """Tests for TOML to environment variable mappings."""

    def test_all_mappings_have_valid_keys(self):
        """Test all TOML_TO_ENV_MAPPINGS have valid key tuples."""
        from ashare_analyzer.config import TOML_TO_ENV_MAPPINGS

        for toml_path, env_var in TOML_TO_ENV_MAPPINGS.items():
            assert isinstance(toml_path, tuple), f"TOML path should be tuple: {toml_path}"
            assert len(toml_path) >= 1, f"TOML path should have at least one key: {toml_path}"
            assert all(isinstance(k, str) for k in toml_path), f"All keys should be strings: {toml_path}"
            assert isinstance(env_var, str), f"Env var should be string: {env_var}"
            assert env_var.isupper(), f"Env var should be uppercase: {env_var}"
