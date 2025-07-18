from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Any, Optional

import bittensor


class BaseStorage(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parser: "ArgumentParser"):
        """Add storage-specific arguments to the parser."""
        pass

    @abstractmethod
    def save_data(self, key: Any, data: Any, prefix: Optional[Any]) -> None:
        """Saves data by key."""
        pass

    @abstractmethod
    def load_data(self, key: Any, prefix: Optional[Any]) -> Any:
        """Loads data by key. Returns None if the key is not found."""
        pass


    def get_config(self):
        """Returns the config object for specific storage based on class implementation."""
        parser = ArgumentParser()
        self.add_args(parser)
        return bittensor.config(parser)

    @staticmethod
    def generate_user_id(config: "bittensor.config") -> str:
        """
        Generate a unique prefix for storage based on the neuron's identity.

        Args:
            config: Bittensor config containing wallet and network info

        Returns:
            str: A unique prefix string for this neuron based on netuid, wallet name, and hotkey name.
        """
        netuid = getattr(config, "netuid", "unknown")
        wallet_name = getattr(config.wallet, "name", "unknown")
        wallet_hotkey = getattr(config.wallet, "hotkey", "unknown")
        return (
            f"{netuid}_{wallet_name}_{wallet_hotkey}".replace(".", "_")
            .replace(" ", "_")
            .lower()
        )
