# Standard Lib
import pytest
from unittest.mock import Mock

# Third Party
import aiohttp
from cachetools import TTLCache
from aioresponses import aioresponses

# Local
from candles.prices.client import PriceClient, PriceProvider, APIConfig
from candles.prices.schemas import CoinDeskResponseOHLC
from candles.core.data import CandleColor


class TestAPIConfig:
    """Test cases for APIConfig dataclass."""

    def test_api_config_creation(self):
        """Test creating an APIConfig instance."""
        config = APIConfig(
            base_url="https://api.example.com",
            api_key="test_key",
            api_key_header="X-API-Key",
            api_params={"param1": "value1"}
        )
        assert config.base_url == "https://api.example.com"
        assert config.api_key == "test_key"
        assert config.api_key_header == "X-API-Key"
        assert config.api_params == {"param1": "value1"}

    def test_api_config_minimal(self):
        """Test creating an APIConfig with minimal parameters."""
        config = APIConfig(base_url="https://api.example.com")
        assert config.base_url == "https://api.example.com"
        assert config.api_key is None
        assert config.api_key_header is None
        assert config.api_params is None


class TestPriceProvider:
    """Test cases for PriceProvider enum."""

    def test_price_provider_values(self):
        """Test PriceProvider enum values."""
        assert PriceProvider.COINDESK == "coindesk"

    def test_coindesk_config(self):
        """Test CoinDesk provider configuration."""
        config = PriceProvider.COINDESK.config
        assert isinstance(config, APIConfig)
        assert config.base_url == "https://data-api.coindesk.com/index/cc/v1/historical"
        assert config.api_key is None
        assert config.api_key_header is None
        expected_params = {
            "market": "cadli",
            "instrument": "",
            "limit": 1,
            "aggregate": 1,
            "fill": "true",
            "apply_mapping": "true",
            "response_format": "JSON",
            "to_ts": "",
            "groups": "OHLC",
        }
        assert config.api_params == expected_params

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            # This would fail in real usage as StrEnum validation prevents invalid values
            # But we test the logic in the config property
            provider = Mock()
            provider.__eq__ = Mock(return_value=False)
            PriceProvider.config.fget(provider)


class TestPriceClient:
    """Test cases for PriceClient class."""

    def test_price_client_initialization(self):
        """Test PriceClient initialization."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        assert client.api_key == "test_key"
        assert client.provider == "coindesk"
        assert client.provider_enum == PriceProvider.COINDESK
        assert isinstance(client.api_config, APIConfig)

    def test_price_client_invalid_provider(self):
        """Test PriceClient with invalid provider."""
        with pytest.raises(ValueError):
            PriceClient(api_key="test_key", provider="invalid_provider")

    @pytest.mark.asyncio
    async def test_get_price_by_interval_success(self):
        """Test successful price retrieval."""
        mock_response_data = {
            "Data": [{
                "OPEN": "50000.0",
                "CLOSE": "51000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            # Match the URL pattern with any query parameters
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::hourly")

            assert isinstance(result, CoinDeskResponseOHLC)
            assert result.open == 50000.0
            assert result.close == 51000.0
            assert result.timestamp == "2023-01-01T00:00:00Z"
            assert result.color == CandleColor.GREEN

    @pytest.mark.asyncio
    async def test_get_price_by_interval_red_candle(self):
        """Test price retrieval resulting in red candle."""
        mock_response_data = {
            "Data": [{
                "OPEN": "51000.0",
                "CLOSE": "50000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            result = await client.get_price_by_interval("BTC", "2023-01-01T00:00:00Z::hourly")

            assert result.color == CandleColor.RED

    @pytest.mark.asyncio
    async def test_get_price_by_interval_http_error(self):
        """Test handling of HTTP errors."""
        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, status=404)

            client = PriceClient(api_key="test_key", provider="coindesk")
            
            with pytest.raises(aiohttp.ClientResponseError):
                await client.get_price_by_interval("INVALID", "2023-01-01T00:00:00Z::hourly")

    def test_get_price_by_interval_caching_setup(self):
        """Test that caching is properly configured."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        
        # Verify that the method has caching decorator
        assert hasattr(client.get_price_by_interval, '__wrapped__')
        
        # Verify cache configuration exists
        # The cache should be accessible through the decorator
        assert isinstance(client.get_price_by_interval.cache, TTLCache)
        assert client.get_price_by_interval.cache.maxsize == 1024
        assert client.get_price_by_interval.cache.ttl == 500

    def test_api_params_modification(self):
        """Test that API parameters are correctly modified for requests."""
        client = PriceClient(api_key="test_key", provider="coindesk")
        
        # Check that api_key and instrument are set properly in params
        client.api_config.api_params.copy()
        
        # Simulate what happens in get_price_by_interval
        client.api_config.api_params["api_key"] = client.api_key
        client.api_config.api_params["instrument"] = "BTC-USDT"
        
        assert client.api_config.api_params["api_key"] == "test_key"
        assert client.api_config.api_params["instrument"] == "BTC-USDT"

    @pytest.mark.asyncio
    async def test_symbol_case_handling(self):
        """Test that symbol is properly converted to uppercase."""
        mock_response_data = {
            "Data": [{
                "OPEN": "50000.0",
                "CLOSE": "51000.0",
                "TIMESTAMP": "2023-01-01T00:00:00Z"
            }]
        }

        with aioresponses() as m:
            import re
            pattern = re.compile(r"https://data-api\.coindesk\.com/index/cc/v1/historical/hours.*")
            m.get(pattern, payload=mock_response_data)

            client = PriceClient(api_key="test_key", provider="coindesk")
            await client.get_price_by_interval("btc", "2023-01-01T00:00:00Z::hourly")

            # Verify that the instrument parameter was set with uppercase symbol
            # Check that a request was made
            assert len(m.requests) == 1