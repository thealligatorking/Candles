# Standard Lib
import asyncio
from typing import Any

# Third Party
import bittensor

# Local
from ..data import CandlePrediction
from .base import ScoringResult
from .prediction_scorer import PredictionScorer
from ...prices.client import PriceClient


class PredictionBatchScorer:
    """Main scorer class that coordinates evaluation of multiple predictions."""

    def __init__(self, price_client: PriceClient, symbol: str = "TAO-USD"):
        """Initialize the batch scorer.

        Args:
            price_client: Client for fetching price data
            symbol: Trading symbol to score predictions for
        """
        self.prediction_scorer = PredictionScorer(price_client, symbol)

    async def score_predictions_by_interval(self, predictions_data: dict[str, list[dict[str, Any]]]) -> dict[str, list[ScoringResult]]:
        """Score all predictions grouped by interval_id.

        Args:
            predictions_data: Dictionary with interval_id as key and list of prediction data as value

        Returns:
            dict[str, list[ScoringResult]]: Scoring results grouped by interval_id
        """
        results = {}

        for interval_id, predictions in predictions_data.items():
            # Create list of scoring tasks for this interval
            scoring_tasks = []
            prediction_objects = []

            for prediction_dict in predictions:

                # Convert to CandlePrediction model
                try:
                    prediction = CandlePrediction(**prediction_dict)
                    prediction_objects.append(prediction)
                    # Create async task for scoring
                    # these aren't tasks, but rather coroutines, but that's semantics
                    task = self.prediction_scorer.score_prediction(prediction)
                    scoring_tasks.append(task)
                except Exception as e:
                    bittensor.logging.error(f"Error creating prediction object: {e}")
                    continue

            # Execute all scoring tasks concurrently for this interval
            if scoring_tasks:
                try:
                    interval_results = await asyncio.gather(*scoring_tasks, return_exceptions=True)
                    # Filter out exceptions and keep only successful results
                    successful_results = []
                    for i, result in enumerate(interval_results):
                        if isinstance(result, Exception):
                            bittensor.logging.error(f"Error scoring prediction {prediction_objects[i].prediction_id}: {result}")
                        else:
                            successful_results.append(result)
                    results[interval_id] = successful_results
                except Exception as e:
                    bittensor.logging.error(f"Error processing interval {interval_id}: {e}")
                    results[interval_id] = []
            else:
                results[interval_id] = []

        return results

    def get_miner_scores(self, scoring_results: dict[str, list[ScoringResult]]) -> dict[int, dict[str, float]]:
        """Aggregate scores by miner across all intervals.

        Args:
            scoring_results: Results from score_predictions_by_interval

        Returns:
            dict[int, dict[str, float]]: Miner scores with statistics
        """
        miner_scores = {}
        self._accumulate_miner_stats(scoring_results, miner_scores)
        self._calculate_miner_averages(miner_scores)
        return miner_scores

    def _accumulate_miner_stats(self, scoring_results: dict[str, list[ScoringResult]], miner_scores: dict[int, dict[str, float]]) -> None:
        """Accumulate stats for each miner."""
        for results in scoring_results.values():
            for result in results:
                miner_uid = result.miner_uid

                if miner_uid not in miner_scores:
                    miner_scores[miner_uid] = {
                        'total_score': 0.0,
                        'prediction_count': 0,
                        'color_accuracy': 0.0,
                        'price_accuracy': 0.0,
                        'average_confidence': 0.0
                    }

                stats = miner_scores[miner_uid]
                stats['total_score'] += result.final_score
                stats['prediction_count'] += 1
                stats['color_accuracy'] += result.color_score
                stats['price_accuracy'] += result.price_score
                stats['average_confidence'] += result.confidence_weight

    def _calculate_miner_averages(self, miner_scores: dict[int, dict[str, float]]) -> None:
        """Calculate average statistics for each miner."""
        for stats in miner_scores.values():
            count = stats['prediction_count']
            if count > 0:
                stats['average_score'] = stats['total_score'] / count
                stats['color_accuracy'] = stats['color_accuracy'] / count
                stats['price_accuracy'] = stats['price_accuracy'] / count
                stats['average_confidence'] = stats['average_confidence'] / count

    def get_top_miners(self, miner_scores: dict[int, dict[str, float]], limit: int = 10) -> list[dict[str, Any]]:
        """Get top performing miners sorted by average score.

        Args:
            miner_scores: Results from get_miner_scores
            limit: Maximum number of miners to return

        Returns:
            list[dict[str, Any]]: Top miners with their scores
        """
        sorted_miners = sorted(
            miner_scores.items(),
            key=lambda x: x[1].get('average_score', 0.0),
            reverse=True
        )

        top_miners = []
        top_miners.extend(
            {'miner_uid': miner_uid, **stats}
            for miner_uid, stats in sorted_miners[:limit]
        )
        return top_miners
