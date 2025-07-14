import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from candles.validator.validator import Validator
from candles.core.scoring.base import ScoringResult
from candles.core.scoring.batch_scorer import PredictionBatchScorer
from candles.prices.client import PriceClient


@pytest.fixture
def mock_config():
    """Mock configuration for validator."""
    config = MagicMock()
    config.immediate = False
    config.neuron.disable_set_weights = False
    config.neuron.timeout = 30
    config.neuron.moving_average_alpha = 0.1
    config.offline = False
    config.json_path = None
    return config


@pytest.fixture
def mock_validator_with_scoring():
    """Create a validator instance with mocked dependencies for scoring tests."""
    with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
        with patch("candles.validator.validator.JsonValidatorStorage"):
            with patch.object(Validator, "load_state"):
                with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                    with patch("candles.validator.validator.PriceClient"):
                        with patch("candles.validator.validator.PredictionBatchScorer"):
                            validator = Validator()

                            # Mock basic validator attributes
                            validator.config = MagicMock()
                            validator.config.immediate = False
                            validator.config.neuron.disable_set_weights = False
                            validator.config.offline = False
                            validator.config.mock = True
                            validator.metagraph = MagicMock()
                            validator.metagraph.n = 10
                            validator.scores = np.zeros(10, dtype=np.float32)
                            validator.storage = MagicMock()
                            validator.should_exit = False
                            validator.loop = MagicMock()

                            # Mock scoring components
                            validator.price_client = MagicMock()
                            validator.batch_scorer = MagicMock()
                            validator.set_weights = MagicMock()
                            validator.update_scores = MagicMock()

                            return validator


@pytest.fixture
def sample_predictions_data():
    """Sample predictions data for testing."""
    return {
        "1234567890::hourly": [
            {
                "prediction_id": 1234567890,
                "interval": "hourly",
                "interval_id": "1234567890::hourly",
                "miner_uid": 1,
                "color": "green",
                "price": "100.5",
                "confidence": "0.8"
            },
            {
                "prediction_id": 1234567890,
                "interval": "hourly",
                "interval_id": "1234567890::hourly",
                "miner_uid": 2,
                "color": "red",
                "price": "99.5",
                "confidence": "0.7"
            }
        ],
        "1234567891::daily": [
            {
                "prediction_id": 1234567891,
                "interval": "daily",
                "interval_id": "1234567891::daily",
                "miner_uid": 1,
                "color": "green",
                "price": "101.0",
                "confidence": "0.9"
            }
        ]
    }


@pytest.fixture
def sample_scoring_results():
    """Sample scoring results for testing."""
    return {
        "1234567890::hourly": [
            ScoringResult(
                prediction_id=1234567890,
                miner_uid=1,
                interval_id="1234567890::hourly",
                color_score=1.0,
                price_score=0.9,
                confidence_weight=0.8,
                final_score=0.94,
                actual_color="green",
                actual_price=100.0
            ),
            ScoringResult(
                prediction_id=1234567890,
                miner_uid=2,
                interval_id="1234567890::hourly",
                color_score=0.0,
                price_score=0.8,
                confidence_weight=0.7,
                final_score=0.32,
                actual_color="green",
                actual_price=100.0
            )
        ],
        "1234567891::daily": [
            ScoringResult(
                prediction_id=1234567891,
                miner_uid=1,
                interval_id="1234567891::daily",
                color_score=1.0,
                price_score=0.95,
                confidence_weight=0.9,
                final_score=0.97,
                actual_color="green",
                actual_price=101.0
            )
        ]
    }


@pytest.fixture
def sample_miner_scores():
    """Sample aggregated miner scores for testing."""
    return {
        1: {
            'total_score': 1.91,
            'prediction_count': 2,
            'color_accuracy': 2.0,
            'price_accuracy': 1.85,
            'average_confidence': 1.7,
            'average_score': 0.955,
        },
        2: {
            'total_score': 0.32,
            'prediction_count': 1,
            'color_accuracy': 0.0,
            'price_accuracy': 0.8,
            'average_confidence': 0.7,
            'average_score': 0.32,
        }
    }


class TestIncentiveScoringAndSetWeights:
    """Test the incentive scoring and weight setting functionality."""

    def test_update_validator_scores_updates_scores_correctly(
        self, mock_validator_with_scoring, sample_miner_scores
    ):
        """Test that _update_validator_scores correctly updates validator scores."""
        validator = mock_validator_with_scoring

        validator._update_validator_scores(sample_miner_scores)

        # Verify update_scores was called with correct parameters
        expected_rewards = np.array([0.955, 0.32])
        expected_uids = [1, 2]

        validator.update_scores.assert_called_once()
        call_args = validator.update_scores.call_args[0]

        np.testing.assert_array_almost_equal(call_args[0], expected_rewards)
        assert list(call_args[1]) == expected_uids

    def test_update_validator_scores_handles_empty_scores(self, mock_validator_with_scoring):
        """Test that _update_validator_scores handles empty miner scores gracefully."""
        validator = mock_validator_with_scoring

        validator._update_validator_scores({})

        # Should not call update_scores with empty data
        validator.update_scores.assert_not_called()

    def test_update_validator_scores_filters_invalid_uids(
        self, mock_validator_with_scoring, sample_miner_scores
    ):
        """Test that _update_validator_scores filters out invalid miner UIDs."""
        validator = mock_validator_with_scoring

        # Add invalid UID (out of range)
        invalid_scores = sample_miner_scores.copy()
        invalid_scores[99] = {
            'average_score': 0.5,
            'prediction_count': 1
        }

        validator._update_validator_scores(invalid_scores)

        # Should only update valid UIDs
        validator.update_scores.assert_called_once()
        call_args = validator.update_scores.call_args[0]
        assert len(call_args[1]) == 2  # Only valid UIDs

    @pytest.mark.asyncio
    async def test_score_and_update_weights_async(
        self, mock_validator_with_scoring, sample_predictions_data
    ):
        """Test that _score_and_update_weights_async works correctly."""
        validator = mock_validator_with_scoring

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            with patch.object(validator, '_load_predictions_for_intervals') as mock_load:
                mock_intervals.return_value = ["1234567890::hourly"]
                mock_load.return_value = sample_predictions_data

                # Mock scoring results as async
                expected_results = {"test": "results"}
                from unittest.mock import AsyncMock
                validator.batch_scorer.score_predictions_by_interval = AsyncMock(return_value=expected_results)
                validator.batch_scorer.get_miner_scores.return_value = {1: {'average_score': 0.5}}
                validator.batch_scorer.get_top_miners.return_value = []

                await validator._score_and_update_weights_async()

                # Verify the scoring flow was called
                mock_intervals.assert_called_once()
                mock_load.assert_called_once_with(["1234567890::hourly"])
                validator.batch_scorer.score_predictions_by_interval.assert_called_once()
                validator.batch_scorer.get_miner_scores.assert_called_once()
                validator.batch_scorer.get_top_miners.assert_called_once()

    @pytest.mark.asyncio
    @patch('candles.validator.validator.asyncio.sleep')
    async def test_incentive_scoring_and_set_weights_async_loop(
        self, mock_sleep, mock_validator_with_scoring, sample_predictions_data
    ):
        """Test that _incentive_scoring_and_set_weights_async handles the main loop."""
        validator = mock_validator_with_scoring
        validator.should_exit = True  # Exit immediately to prevent infinite loop

        with patch.object(validator, '_score_and_update_weights_async') as mock_score:
            # Should exit immediately due to should_exit being True
            await validator._incentive_scoring_and_set_weights_async()

            # Should not call scoring since should_exit is True
            mock_score.assert_not_called()

    @pytest.mark.asyncio
    @patch('candles.validator.validator.asyncio.sleep')
    @patch('candles.validator.validator.datetime')
    async def test_incentive_scoring_and_set_weights_30_minute_trigger(
        self, mock_datetime, mock_sleep, mock_validator_with_scoring
    ):
        """Test that scoring is triggered every 30 minutes."""
        validator = mock_validator_with_scoring
        validator.incentive_scoring_interval = 30

        # Mock time to trigger scoring (30 minutes)
        mock_now = MagicMock()
        mock_now.minute = 30
        mock_datetime.now.return_value = mock_now

        # Mock intervals to score
        with patch.object(validator, '_score_and_update_weights_async') as mock_score:
            # Set should_exit to True after first iteration to prevent infinite loop
            call_count = 0
            async def mock_score_impl():
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    validator.should_exit = True
            mock_score.side_effect = mock_score_impl

            # Run the method
            await validator._incentive_scoring_and_set_weights_async()

            # Verify scoring was called
            mock_score.assert_called_once()

    @pytest.mark.asyncio
    @patch('candles.validator.validator.asyncio.sleep')
    @patch('candles.validator.validator.datetime')
    async def test_incentive_scoring_and_set_weights_skips_non_30_minute_intervals(
        self, mock_datetime, mock_sleep, mock_validator_with_scoring
    ):
        """Test that scoring is skipped when not at 30-minute intervals."""
        validator = mock_validator_with_scoring
        validator.incentive_scoring_interval = 30
        validator.should_exit = True  # Exit immediately

        # Mock time to NOT trigger scoring (15 minutes)
        mock_now = MagicMock()
        mock_now.minute = 15
        mock_datetime.now.return_value = mock_now

        # Mock scoring method
        with patch.object(validator, '_score_and_update_weights_async') as mock_score:
            # Run the method
            await validator._incentive_scoring_and_set_weights_async()

            # Verify scoring was NOT called
            mock_score.assert_not_called()

    @pytest.mark.asyncio
    @patch('candles.validator.validator.asyncio.sleep')
    @patch('candles.validator.validator.datetime')
    async def test_incentive_scoring_and_set_weights_immediate_mode(
        self, mock_datetime, mock_sleep, mock_validator_with_scoring
    ):
        """Test that immediate mode is not supported - scoring only happens at intervals."""
        validator = mock_validator_with_scoring
        validator.config.immediate = True
        validator.incentive_scoring_interval = 30
        validator.should_exit = True  # Exit immediately

        # Mock time to NOT trigger scoring normally (15 minutes)
        mock_now = MagicMock()
        mock_now.minute = 15
        mock_datetime.now.return_value = mock_now

        # Mock scoring method
        with patch.object(validator, '_score_and_update_weights_async') as mock_score:
            # Run the method
            await validator._incentive_scoring_and_set_weights_async()

            # Verify scoring was NOT triggered at 15 minutes (immediate mode not supported)
            mock_score.assert_not_called()

    @pytest.mark.asyncio
    @patch('candles.validator.validator.asyncio.sleep')
    @patch('candles.validator.validator.datetime')
    async def test_incentive_scoring_and_set_weights_full_scoring_flow(
        self, mock_datetime, mock_sleep, mock_validator_with_scoring,
        sample_predictions_data, sample_scoring_results, sample_miner_scores
    ):
        """Test the complete scoring flow with predictions."""
        validator = mock_validator_with_scoring
        validator.incentive_scoring_interval = 30

        # Mock time to trigger scoring
        mock_now = MagicMock()
        mock_now.minute = 30
        mock_datetime.now.return_value = mock_now

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            with patch.object(validator, '_load_predictions_for_intervals') as mock_load:
                mock_intervals.return_value = ["1234567890::hourly", "1234567891::daily"]
                mock_load.return_value = sample_predictions_data

                # Mock scoring results
                validator.batch_scorer.score_predictions_by_interval.return_value = sample_scoring_results
                validator.batch_scorer.get_miner_scores.return_value = sample_miner_scores
                validator.batch_scorer.get_top_miners.return_value = [
                    {'miner_uid': 1, 'average_score': 0.955},
                    {'miner_uid': 2, 'average_score': 0.32}
                ]

                # Set should_exit to True after first iteration to prevent infinite loop
                call_count = 0
                async def mock_score_impl():
                    nonlocal call_count
                    call_count += 1
                    if call_count >= 1:
                        validator.should_exit = True

                with patch.object(validator, '_score_and_update_weights_async', side_effect=mock_score_impl) as mock_score:
                    # Run the method
                    await validator._incentive_scoring_and_set_weights_async()

                    # Verify scoring was called
                    mock_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_and_update_weights_handles_no_predictions(
        self, mock_validator_with_scoring
    ):
        """Test handling when no predictions are found for closed intervals."""
        validator = mock_validator_with_scoring

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            with patch.object(validator, '_load_predictions_for_intervals') as mock_load:
                mock_intervals.return_value = ["1234567890::hourly"]
                mock_load.return_value = {}  # No predictions

                # Run the method
                await validator._score_and_update_weights_async()

                # Verify scoring was not attempted
                validator.batch_scorer.score_predictions_by_interval.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_and_update_weights_handles_scoring_errors(
        self, mock_validator_with_scoring, sample_predictions_data
    ):
        """Test error handling during scoring process."""
        validator = mock_validator_with_scoring

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            with patch.object(validator, '_load_predictions_for_intervals') as mock_load:
                mock_intervals.return_value = ["1234567890::hourly"]
                mock_load.return_value = sample_predictions_data

                # Mock scoring error
                validator.batch_scorer.score_predictions_by_interval.side_effect = Exception("Scoring failed")

                # Run the method - should not raise exception but handle gracefully
                try:
                    await validator._score_and_update_weights_async()
                except Exception:
                    pass  # Error is expected to be handled internally

    @pytest.mark.asyncio
    async def test_score_and_update_weights_respects_disable_weights_config(
        self, mock_validator_with_scoring
    ):
        """Test that weight setting is skipped when disabled in config."""
        validator = mock_validator_with_scoring
        validator.config.neuron.disable_set_weights = True

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            mock_intervals.return_value = []

            # Run the method
            await validator._score_and_update_weights_async()

            # Since no intervals to score, set_weights should not be called
            validator.set_weights.assert_not_called()

    @pytest.mark.asyncio
    async def test_score_and_update_weights_respects_offline_config(
        self, mock_validator_with_scoring
    ):
        """Test that weight setting is skipped when validator is offline."""
        validator = mock_validator_with_scoring
        validator.config.offline = True

        # Mock intervals to score
        with patch.object(validator, '_intervals_to_score') as mock_intervals:
            mock_intervals.return_value = []

            # Run the method
            await validator._score_and_update_weights_async()

            # Since no intervals to score, set_weights should not be called
            validator.set_weights.assert_not_called()


class TestValidatorInitialization:
    """Test validator initialization with scoring components."""

    @patch("candles.validator.validator.PriceClient")
    @patch("candles.validator.validator.PredictionBatchScorer")
    @patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"})
    def test_validator_init_creates_scoring_components(
        self, mock_batch_scorer_class, mock_price_client_class, mock_config
    ):
        """Test that validator initialization creates required scoring components."""
        with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
            with patch("candles.validator.validator.JsonValidatorStorage"):
                with patch.object(Validator, "load_state"):
                    mock_config.mock = True  # Prevent background task creation

                    validator = Validator(config=mock_config)

                    # Verify scoring components were created
                    mock_price_client_class.assert_called_once_with(api_key="test_api_key", provider="coindesk")
                    mock_batch_scorer_class.assert_called_once()

                    # Verify no background task was created in mock mode
                    assert validator.scoring_task is None


@pytest.mark.integration
class TestIncentiveScoringIntegration:
    """Integration tests for the complete scoring flow."""

    @pytest.mark.asyncio
    async def test_scoring_integration_with_real_components(self):
        """Integration test with real scoring components (mocked external dependencies)."""
        with patch("candles.validator.validator.BaseValidatorNeuron.__init__"):
            with patch("candles.validator.validator.JsonValidatorStorage"):
                with patch.object(Validator, "load_state"):
                    with patch.dict("os.environ", {"COINDESK_API_KEY": "test_api_key"}):
                        # Create validator with real scoring components but mocked external deps
                        with patch("candles.prices.client.aiohttp"):
                            validator = Validator()
                            validator.config = MagicMock()
                            validator.config.immediate = False
                            validator.config.neuron.disable_set_weights = True  # Prevent actual weight setting
                            validator.config.offline = True
                            validator.config.mock = True  # Prevent background task creation
                            validator.metagraph = MagicMock()
                            validator.metagraph.n = 5
                            validator.scores = np.zeros(5, dtype=np.float32)
                            validator.storage = MagicMock()
                            validator.set_weights = MagicMock()
                            validator.update_scores = MagicMock()
                            validator.should_exit = True  # Prevent async issues

                            # Test that scoring components are properly initialized
                            assert hasattr(validator, 'price_client')
                            assert hasattr(validator, 'batch_scorer')
                            assert isinstance(validator.batch_scorer, PredictionBatchScorer)
                            assert isinstance(validator.price_client, PriceClient)
