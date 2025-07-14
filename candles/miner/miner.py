# The MIT License (MIT)
# Copyright © 2024 sportstensor

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import asyncio
from typing import Tuple
import os
import bittensor
from datetime import datetime, timezone
import random
from decimal import Decimal
from pathlib import Path
import glob

from candles.core.synapse import GetCandlePrediction
from candles.core.data import CandlePrediction, CandleColor, TimeInterval
from candles.miner.base import BaseMinerNeuron
from candles.miner.utils import get_file_predictions

# Global miner instance for blacklist function
miner = None

class Miner(BaseMinerNeuron):
    """The Candles Miner."""

    def __init__(self, config=None):
        global miner  # well this is certainly a choice
        super(Miner, self).__init__(config=config)
        miner = self

    async def async_init(self):
        """
        Async initialization for miner. Must be called after __init__.
        """
        # Initialize the base miner async components
        await super().async_init()

    @staticmethod
    def blacklist(synapse: GetCandlePrediction) -> Tuple[bool, str]:
        """
        ** Warning do not use `tuple` or `typing.Tuple` in this function.
        ** Use `Tuple[bool, str]` instead.

        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (GetCandlePrediction): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.
        """
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"

        # Get the miner instance from the synapse

        registered = synapse.dendrite.hotkey in miner.metagraph.hotkeys
        if miner.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            bittensor.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"

        uid = miner.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if miner.config.blacklist.force_validator_permit and not miner.metagraph.validator_permit[uid]:
            bittensor.logging.warning(
                f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Non-validator hotkey"

        stake = miner.metagraph.S[uid].item()
        if (
            miner.config.blacklist.validator_min_stake
            and stake < miner.config.blacklist.validator_min_stake
        ):
            bittensor.logging.warning(
                f"Blacklisting request from {synapse.dendrite.hotkey} [uid={uid}], not enough stake -- {stake}"
            )
            return True, "Stake below minimum"

        bittensor.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: GetCandlePrediction) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (GetCandlePrediction): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bittensor.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority

    def find_prediction_file(self, interval: TimeInterval) -> str:
        """
        Find prediction files based on interval type. Searches in ~/.candles/data/ first,
        then falls back to PREDICTIONS_FILE_PATH env var.

        Args:
            interval: The time interval to find predictions for

        Returns:
            str: Path to the prediction file, or None if not found
        """
        # Map intervals to file prefixes
        interval_prefixes = {
            TimeInterval.HOURLY: "hourly_",
            TimeInterval.DAILY: "daily_",
            TimeInterval.WEEKLY: "weekly_"
        }

        prefix = interval_prefixes.get(interval)
        if not prefix:
            return None

        # Check ~/.candles/data/ directory first
        candles_data_dir = Path.home() / ".candles" / "data"
        if candles_data_dir.exists():
            pattern = str(candles_data_dir / f"{prefix}*.csv")
            if matching_files := glob.glob(pattern):
                # Return the first matching file (sorted for consistency)
                return sorted(matching_files)[0]

        # Fall back to PREDICTIONS_FILE_PATH env var
        predictions_file_path = os.getenv("PREDICTIONS_FILE_PATH")
        if predictions_file_path and os.path.exists(predictions_file_path):
            return predictions_file_path

        return None

    async def make_candle_prediction(self, candle_prediction):
        """
        Makes a prediction for the requested candle.

        Args:
            candle_prediction: The CandlePrediction object containing the request details.

        Returns:
            CandlePrediction: The prediction with color, price, and confidence.
        """
        bittensor.logging.info(f"****************** Making prediction for interval: [blue]{candle_prediction.interval}[/blue] ******************")

        if prediction_file := self.find_prediction_file(
            candle_prediction.interval
        ):
            bittensor.logging.info(f"[orange]Found prediction file[/orange]: [blue]{prediction_file}[/blue]")
            if predictions := get_file_predictions(
                filename=prediction_file,
                interval=candle_prediction.interval,
                miner_uid=self.uid,
                hotkey=self.wallet.hotkey.ss58_address,
            ):
                if prediction := next(
                    (
                        prediction
                        for prediction in predictions
                        if prediction.interval_id == candle_prediction.interval_id
                    ),
                    None,
                ):
                    bittensor.logging.info(f"[orange]Using prediction from file[/orange]: [blue]{prediction}[/blue]")
                    return prediction

        # Generate random prediction directly for better test compatibility
        bittensor.logging.debug(f"Making prediction for interval: {candle_prediction.interval}")

        # Generate a random price between 100 and 1000
        price = Decimal(str(random.uniform(100, 1000)))
        bittensor.logging.debug(f"Generated price: {price}")

        # Randomly choose a color
        color = random.choice([CandleColor.RED, CandleColor.GREEN])
        bittensor.logging.debug(f"Generated color: {color}")

        # Generate a random confidence between 0.5 and 1.0
        confidence = Decimal(str(random.uniform(0.5, 1.0)))
        bittensor.logging.debug(f"Generated confidence: {confidence}")

        # Use the original interval_id if provided
        interval_id = candle_prediction.interval_id if hasattr(candle_prediction, 'interval_id') and candle_prediction.interval_id else f"generated_{candle_prediction.interval}"

        # Preserve the original prediction_id if provided
        prediction_id = candle_prediction.prediction_id if hasattr(candle_prediction, 'prediction_id') and candle_prediction.prediction_id else None

        return CandlePrediction(
            prediction_id=prediction_id,
            price=price,
            color=color,
            confidence=confidence,
            prediction_date=int(datetime.now(timezone.utc).timestamp()),
            interval=candle_prediction.interval,
            interval_id=interval_id,
            miner_uid=self.uid,
            hotkey=self.wallet.hotkey.ss58_address,
        )

    async def get_candle_prediction(self, synapse: GetCandlePrediction) -> GetCandlePrediction:
        bittensor.logging.debug(
            f"Received GetCandlePrediction request in forward() from {synapse.dendrite.hotkey}."
        )

        synapse.candle_prediction = await self.make_candle_prediction(synapse.candle_prediction)
        synapse.version = 1

        bittensor.logging.success(
            f"Returning CandlePrediction to [orange]{synapse.dendrite.hotkey}[/orange]:" +
            f"\n color = [yellow]{synapse.candle_prediction.color}[/yellow]," +
            f"\n price = [blue]{synapse.candle_prediction.price}[/blue]," +
            f"\n confidence = [magenta]{synapse.candle_prediction.confidence}[/magenta]."
        )

        return synapse

    def save_state(self):
        """
        We define this function to avoid printing out the log message in the BaseNeuron class
        that says `save_state() not implemented`.
        """
        pass


async def main():
    """Main async entry point for the miner."""
    miner = Miner()
    try:
        await miner.async_init()
        await miner.run()
    finally:
        if not miner.config.mock:
            await miner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
