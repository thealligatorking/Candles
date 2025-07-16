# Standard Lib
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

# Third Party
import aiohttp
from cachetools import cached, TTLCache

# Local
from .base import BasePriceClient
from .schemas import CoinDeskResponseOHLC

@dataclass
class APIConfig:
    base_url: str
    api_key: Optional[str] = None
    api_key_header: Optional[str] = None
    api_params: Optional[dict] = None

class PriceProvider(StrEnum):
    COINDESK = "coindesk"

    @property
    def config(self) -> APIConfig:
        if self == PriceProvider.COINDESK:
            return APIConfig(
                base_url="https://data-api.coindesk.com/index/cc/v1/historical",
                api_params={
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
            )
        raise ValueError(f"Unknown provider: {self}")

class PriceClient(BasePriceClient):

    def __init__(self, api_key: str, provider: str, *args, **kwargs):
        super().__init__(api_key, provider, *args, **kwargs)
        self.provider_enum = PriceProvider(provider)
        self.api_config = self.provider_enum.config

    @cached(cache=TTLCache(maxsize=1024, ttl=500))
    async def get_price_by_interval(self, symbol: str, interval_id: str) -> CoinDeskResponseOHLC:
        if self.provider_enum == PriceProvider.COINDESK:
            timestamp, interval = interval_id.split("::")

            match interval:
                case "hourly":
                    coindesk_interval = "hours"
                case "daily":
                    coindesk_interval = "days"
                case _:
                    raise ValueError(f"Unsupported interval: {interval}")

            url = f"{self.api_config.base_url}/{coindesk_interval}"
            self.api_config.api_params["api_key"] = self.api_key
            self.api_config.api_params["instrument"] = f"{symbol}"
            self.api_config.api_params["to_ts"] = timestamp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=self.api_config.api_params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return CoinDeskResponseOHLC.parse_response(data["Data"][0])

