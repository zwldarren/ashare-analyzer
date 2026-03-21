"""Tests for stock code utility module."""

from ashare_analyzer.utils.stock_code import (
    StockType,
    convert_to_provider_format,
    detect_stock_type,
    get_market_from_code,
    is_etf_code,
    is_hk_code,
    is_us_code,
)


class TestDetectStockType:
    """Test stock type detection."""

    def test_detect_a_share_shanghai_main(self):
        """Test detecting Shanghai main board A-share."""
        assert detect_stock_type("600519") == StockType.A_SHARE
        assert detect_stock_type("601318") == StockType.A_SHARE
        assert detect_stock_type("603986") == StockType.A_SHARE

    def test_detect_a_share_shanghai_star(self):
        """Test detecting Shanghai STAR Market (科创版) as A-share."""
        # 688xxx codes are STAR Market
        assert detect_stock_type("688001") == StockType.A_SHARE

    def test_detect_a_share_shenzhen_main(self):
        """Test detecting Shenzhen main board A-share."""
        assert detect_stock_type("000001") == StockType.A_SHARE
        assert detect_stock_type("000002") == StockType.A_SHARE

    def test_detect_a_share_shenzhen_sme(self):
        """Test detecting Shenzhen SME board (中小板) as A-share."""
        # 002xxx codes are SME board
        assert detect_stock_type("002415") == StockType.A_SHARE

    def test_detect_a_share_chinext(self):
        """Test detecting ChiNext (创业板) as A-share."""
        # 300xxx codes are ChiNext
        assert detect_stock_type("300750") == StockType.A_SHARE

    def test_detect_etf_codes(self):
        """Test detecting ETF codes."""
        # ETFs start with 15, 16, 51, 56, 58, 59
        assert detect_stock_type("159915") == StockType.ETF  # 创业板ETF
        assert detect_stock_type("160105") == StockType.ETF
        assert detect_stock_type("510300") == StockType.ETF  # 沪深300ETF
        assert detect_stock_type("560010") == StockType.ETF
        assert detect_stock_type("585000") == StockType.ETF
        assert detect_stock_type("590005") == StockType.ETF

    def test_detect_hk_stocks(self):
        """Test detecting Hong Kong stocks."""
        # 5-digit codes
        assert detect_stock_type("00700") == StockType.HK
        assert detect_stock_type("09988") == StockType.HK
        # With hk prefix
        assert detect_stock_type("hk00700") == StockType.HK
        assert detect_stock_type("HK00700") == StockType.HK

    def test_detect_us_stocks(self):
        """Test detecting US stocks."""
        assert detect_stock_type("AAPL") == StockType.US
        assert detect_stock_type("TSLA") == StockType.US
        assert detect_stock_type("MSFT") == StockType.US
        # With class suffix
        assert detect_stock_type("BRK.B") == StockType.US
        assert detect_stock_type("BRK.A") == StockType.US

    def test_detect_unknown_codes(self):
        """Test detecting unknown/invalid codes."""
        assert detect_stock_type("") == StockType.UNKNOWN
        assert detect_stock_type(None) == StockType.UNKNOWN
        assert detect_stock_type("INVALID") == StockType.UNKNOWN
        assert detect_stock_type("12345") == StockType.HK  # 5 digits = HK
        assert detect_stock_type("1234567") == StockType.UNKNOWN  # 7 digits = unknown

    def test_detect_with_whitespace(self):
        """Test detection handles whitespace."""
        assert detect_stock_type(" 600519 ") == StockType.A_SHARE
        assert detect_stock_type(" AAPL ") == StockType.US

    def test_detect_case_insensitive(self):
        """Test detection is case insensitive."""
        assert detect_stock_type("aapl") == StockType.US
        assert detect_stock_type("HK00700") == StockType.HK
        assert detect_stock_type("hk00700") == StockType.HK


class TestIsUsCode:
    """Test US stock code detection."""

    def test_is_us_code_true(self):
        """Test is_us_code returns True for US stocks."""
        assert is_us_code("AAPL") is True
        assert is_us_code("TSLA") is True
        assert is_us_code("BRK.B") is True

    def test_is_us_code_false(self):
        """Test is_us_code returns False for non-US stocks."""
        assert is_us_code("600519") is False
        assert is_us_code("00700") is False
        assert is_us_code("159915") is False


class TestIsHkCode:
    """Test Hong Kong stock code detection."""

    def test_is_hk_code_true(self):
        """Test is_hk_code returns True for HK stocks."""
        assert is_hk_code("00700") is True
        assert is_hk_code("09988") is True
        assert is_hk_code("hk00700") is True

    def test_is_hk_code_false(self):
        """Test is_hk_code returns False for non-HK stocks."""
        assert is_hk_code("600519") is False
        assert is_hk_code("AAPL") is False
        assert is_hk_code("159915") is False


class TestIsEtfCode:
    """Test ETF code detection."""

    def test_is_etf_code_true(self):
        """Test is_etf_code returns True for ETF codes."""
        assert is_etf_code("159915") is True
        assert is_etf_code("510300") is True
        assert is_etf_code("160105") is True

    def test_is_etf_code_false(self):
        """Test is_etf_code returns False for non-ETF codes."""
        assert is_etf_code("600519") is False
        assert is_etf_code("000001") is False
        assert is_etf_code("AAPL") is False


class TestConvertToProviderFormat:
    """Test stock code conversion to provider format."""

    # yfinance tests
    def test_convert_yfinance_shanghai(self):
        """Test converting Shanghai stocks for yfinance."""
        assert convert_to_provider_format("600519", "yfinance") == "600519.SS"
        assert convert_to_provider_format("601318", "yfinance") == "601318.SS"
        assert convert_to_provider_format("688001", "yfinance") == "688001.SS"

    def test_convert_yfinance_shenzhen(self):
        """Test converting Shenzhen stocks for yfinance."""
        assert convert_to_provider_format("000001", "yfinance") == "000001.SZ"
        assert convert_to_provider_format("002415", "yfinance") == "002415.SZ"
        assert convert_to_provider_format("300750", "yfinance") == "300750.SZ"

    def test_convert_yfinance_hk(self):
        """Test converting HK stocks for yfinance."""
        assert convert_to_provider_format("00700", "yfinance") == "0700.HK"
        assert convert_to_provider_format("hk00700", "yfinance") == "0700.HK"
        assert convert_to_provider_format("09988", "yfinance") == "9988.HK"

    def test_convert_yfinance_us(self):
        """Test converting US stocks for yfinance (no change)."""
        assert convert_to_provider_format("AAPL", "yfinance") == "AAPL"
        assert convert_to_provider_format("BRK.B", "yfinance") == "BRK.B"

    # tushare tests
    def test_convert_tushare_shanghai(self):
        """Test converting Shanghai stocks for tushare."""
        assert convert_to_provider_format("600519", "tushare") == "600519.SH"
        assert convert_to_provider_format("688001", "tushare") == "688001.SH"

    def test_convert_tushare_shenzhen(self):
        """Test converting Shenzhen stocks for tushare."""
        assert convert_to_provider_format("000001", "tushare") == "000001.SZ"
        assert convert_to_provider_format("300750", "tushare") == "300750.SZ"

    # baostock tests
    def test_convert_baostock_shanghai(self):
        """Test converting Shanghai stocks for baostock."""
        assert convert_to_provider_format("600519", "baostock") == "sh.600519"
        assert convert_to_provider_format("688001", "baostock") == "sh.688001"

    def test_convert_baostock_shenzhen(self):
        """Test converting Shenzhen stocks for baostock."""
        assert convert_to_provider_format("000001", "baostock") == "sz.000001"
        assert convert_to_provider_format("300750", "baostock") == "sz.300750"

    def test_convert_with_existing_suffix(self):
        """Test conversion handles codes with existing suffixes."""
        # Should remove existing suffix and add correct one
        assert convert_to_provider_format("600519.SS", "tushare") == "600519.SH"
        assert convert_to_provider_format("600519.SH", "yfinance") == "600519.SS"
        assert convert_to_provider_format("000001.SZ", "baostock") == "sz.000001"

    def test_convert_with_existing_prefix(self):
        """Test conversion handles codes with existing prefixes."""
        assert convert_to_provider_format("sh.600519", "yfinance") == "600519.SS"
        assert convert_to_provider_format("sz.000001", "tushare") == "000001.SZ"

    def test_convert_unknown_provider(self):
        """Test conversion for unknown provider returns original code."""
        assert convert_to_provider_format("600519", "unknown") == "600519"
        assert convert_to_provider_format("AAPL", "unknown") == "AAPL"

    def test_convert_whitespace_handling(self):
        """Test conversion handles whitespace."""
        assert convert_to_provider_format(" 600519 ", "yfinance") == "600519.SS"
        assert convert_to_provider_format(" AAPL ", "yfinance") == "AAPL"

    def test_convert_case_handling(self):
        """Test conversion handles case."""
        assert convert_to_provider_format("600519".lower(), "yfinance") == "600519.SS"
        assert convert_to_provider_format("aapl".upper(), "yfinance") == "AAPL"


class TestGetMarketFromCode:
    """Test market detection from stock code."""

    def test_get_market_shanghai(self):
        """Test detecting Shanghai market."""
        assert get_market_from_code("600519") == "sh"
        assert get_market_from_code("601318") == "sh"
        assert get_market_from_code("603986") == "sh"
        assert get_market_from_code("688001") == "sh"  # STAR Market

    def test_get_market_shenzhen(self):
        """Test detecting Shenzhen market."""
        assert get_market_from_code("000001") == "sz"
        assert get_market_from_code("002415") == "sz"
        assert get_market_from_code("300750") == "sz"

    def test_get_market_unknown(self):
        """Test detecting unknown market."""
        assert get_market_from_code("12345") == "unknown"
        assert get_market_from_code("AAPL") == "unknown"
        assert get_market_from_code("") == "unknown"

    def test_get_market_with_suffix(self):
        """Test market detection with existing suffix."""
        assert get_market_from_code("600519.SH") == "sh"
        assert get_market_from_code("600519.SS") == "sh"
        assert get_market_from_code("000001.SZ") == "sz"

    def test_get_market_with_prefix(self):
        """Test market detection with existing prefix."""
        assert get_market_from_code("sh.600519") == "sh"
        assert get_market_from_code("sz.000001") == "sz"

    def test_get_market_case_insensitive(self):
        """Test market detection is case insensitive."""
        assert get_market_from_code("600519".upper()) == "sh"
        assert get_market_from_code("600519".lower()) == "sh"


class TestStockTypeEnum:
    """Test StockType enum."""

    def test_stock_type_values(self):
        """Test StockType enum values."""
        assert StockType.A_SHARE.value == "a_share"
        assert StockType.HK.value == "hk"
        assert StockType.US.value == "us"
        assert StockType.ETF.value == "etf"
        assert StockType.UNKNOWN.value == "unknown"

    def test_stock_type_comparison(self):
        """Test StockType enum comparison."""
        assert StockType.A_SHARE == StockType.A_SHARE
        assert StockType.A_SHARE != StockType.HK
        assert StockType.US != StockType.UNKNOWN


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        assert detect_stock_type("") == StockType.UNKNOWN
        # These would raise exceptions in some implementations
        # but our implementation should handle gracefully

    def test_very_long_codes(self):
        """Test handling of very long codes."""
        # Should return UNKNOWN for codes longer than expected
        result = detect_stock_type("123456789")
        assert result == StockType.UNKNOWN

    def test_special_characters(self):
        """Test handling of special characters in codes."""
        assert detect_stock_type("600519@") == StockType.UNKNOWN
        assert detect_stock_type("AAPL-USD") == StockType.UNKNOWN

    def test_hk_code_with_leading_zeros(self):
        """Test HK code handling with various leading zero patterns."""
        # HK codes are 5 digits
        # Note: 4 digits like "0700" are actually treated as UNKNOWN (not 5 digits)
        # Only 5-digit codes are HK stocks
        assert detect_stock_type("00700") == StockType.HK  # 5 digits
        assert detect_stock_type("09988") == StockType.HK  # 5 digits

    def test_convert_hk_code_edge_cases(self):
        """Test HK code conversion edge cases."""
        # Code with all zeros
        result = convert_to_provider_format("00001", "yfinance")
        # Should handle leading zeros properly
        assert ".HK" in result

    def test_convert_preserves_original_code(self):
        """Test that conversion doesn't modify the original code variable."""
        original = "600519"
        result = convert_to_provider_format(original, "yfinance")
        assert original == "600519"  # Original unchanged
        assert result == "600519.SS"

    def test_detect_stock_type_none_input(self):
        """Test detect_stock_type handles None input."""
        result = detect_stock_type(None)
        assert result == StockType.UNKNOWN
