"""Factory for creating approach instances."""

from typing import List, Type, Dict

from .shared_tweets_fast import SharedTweetsFastApproach
from .shared_tweets_counting_rt import SharedTweetCountingRTApproach
from .ignoring_tweet_counting_rt import IgnoringTweetCountingRTApproach
from .base import Approach
from .coretweets import CoRetweetsApproach
from .coretweets_numpy import CoRetweetsNumpyApproach
from .coretweets_fast import CoRetweetsFastApproach
from .ignoring_tweet import IgnoringTweetApproach
from .ignoring_tweet_fast import IgnoringTweetFastApproach
from .shared_tweets import SharedTweetsApproach
from .same_tweet_same_time import SameTweetSameTimeApproach
from .lexicographic import LexicographicApproach


class ApproachFactory:
    """Factory for creating and managing approach instances."""

    _approaches: Dict[str, Type[Approach]] = {
        'coretweets': CoRetweetsApproach,
        'coretweets_numpy': CoRetweetsNumpyApproach,
        'coretweets_fast': CoRetweetsFastApproach,
        'ignoring_tweet': IgnoringTweetApproach,
        'ignoring_tweet_fast': IgnoringTweetFastApproach,
        'shared_tweets': SharedTweetsApproach,
        'same_tweet_same_time': SameTweetSameTimeApproach,
        'lexicographic': LexicographicApproach,
        'ignoring_tweet_counting_rt': IgnoringTweetCountingRTApproach,
        'shared_tweet_counting_rt': SharedTweetCountingRTApproach,
        'shared_tweets_fast': SharedTweetsFastApproach,
    }

    @classmethod
    def create(cls, approach_key: str, window_sec: int = 60,
               min_coactions: int = 1) -> Approach:
        """
        Create an approach instance by key.

        Special format for lexicographic approaches:
        - "lexicographic:approach1+approach2+..." creates a LexicographicApproach
          combining the specified sub-approaches in order
        - Example: "lexicographic:ignoring_tweet_fast+shared_tweets"

        Args:
            approach_key: Key identifying the approach
            window_sec: Time window in seconds
            min_coactions: Minimum co-retweets to consider a pair

        Returns:
            Instance of the requested approach

        Raises:
            ValueError: If approach_key is unknown
        """
        # Handle lexicographic approach with special syntax
        if approach_key.startswith('lexicographic:'):
            # Extract sub-approach keys from "lexicographic:key1+key2+..."
            sub_keys_str = approach_key.split(':', 1)[1]
            sub_keys = sub_keys_str.split('+')

            if len(sub_keys) < 2:
                raise ValueError(
                    f"Lexicographic approach requires at least 2 sub-approaches. "
                    f"Format: 'lexicographic:approach1+approach2+...'"
                )

            # Validate all sub-keys exist (excluding 'lexicographic' itself)
            valid_keys = [k for k in cls._approaches.keys() if k != 'lexicographic']
            for key in sub_keys:
                if key not in valid_keys:
                    raise ValueError(
                        f"Unknown sub-approach in lexicographic: {key}. "
                        f"Valid keys: {', '.join(valid_keys)}"
                    )

            return LexicographicApproach(sub_keys, window_sec, min_coactions)

        # Handle regular approaches
        if approach_key not in cls._approaches:
            valid_keys = ', '.join(cls._approaches.keys())
            raise ValueError(f"Unknown approach: {approach_key}. Valid keys: {valid_keys}")

        approach_class = cls._approaches[approach_key]
        return approach_class(window_sec, min_coactions)

    @classmethod
    def get_all_keys(cls) -> List[str]:
        """Get all available approach keys."""
        return list(cls._approaches.keys())

    @classmethod
    def get_all_approaches(cls, window_sec: int = 60,
                          min_coactions: int = 1) -> List[Approach]:
        """
        Create instances of all available approaches.

        Args:
            window_sec: Time window in seconds
            min_coactions: Minimum co-retweets to consider a pair

        Returns:
            List of all approach instances
        """
        return [cls.create(key, window_sec, min_coactions)
                for key in cls.get_all_keys()]

    @classmethod
    def get_approach_names(cls) -> Dict[str, str]:
        """
        Get mapping of approach keys to human-readable names.

        Returns:
            Dictionary mapping keys to names
        """
        # Create temporary instances to get names
        temp_instances = cls.get_all_approaches()
        return {
            approach.get_approach_key(): approach.get_approach_name()
            for approach in temp_instances
        }

    @classmethod
    def validate_approach_key(cls, approach_key: str) -> None:
        """
        Validate an approach key without creating an instance.

        Supports both regular approach keys and lexicographic syntax.

        Args:
            approach_key: Key to validate (e.g., "coretweets" or "lexicographic:a+b")

        Raises:
            ValueError: If the approach key is invalid
        """
        # Handle lexicographic syntax
        if approach_key.startswith('lexicographic:'):
            sub_keys_str = approach_key.split(':', 1)[1]
            sub_keys = sub_keys_str.split('+')

            if len(sub_keys) < 2:
                raise ValueError(
                    f"Invalid lexicographic approach '{approach_key}'. "
                    f"Format: 'lexicographic:approach1+approach2+...'"
                )

            # Validate all sub-keys (excluding 'lexicographic' itself)
            valid_keys = [k for k in cls._approaches.keys() if k != 'lexicographic']
            for key in sub_keys:
                if key not in valid_keys:
                    raise ValueError(
                        f"Invalid sub-approach '{key}' in lexicographic approach. "
                        f"Valid approaches: {', '.join(valid_keys)}"
                    )
        else:
            # Regular approach
            if approach_key not in cls._approaches:
                valid_keys = ', '.join(cls._approaches.keys())
                raise ValueError(
                    f"Invalid approach '{approach_key}'. Valid approaches: {valid_keys}"
                )

    @classmethod
    def register_approach(cls, approach_key: str, approach_class: Type[Approach]):
        """
        Register a new approach class.

        This allows extending the factory with custom approaches.

        Args:
            approach_key: Key to identify the approach
            approach_class: Class implementing the Approach interface

        Raises:
            ValueError: If key already exists or class is not an Approach subclass
        """
        if approach_key in cls._approaches:
            raise ValueError(f"Approach key '{approach_key}' is already registered")

        if not issubclass(approach_class, Approach):
            raise ValueError(f"{approach_class} must be a subclass of Approach")

        cls._approaches[approach_key] = approach_class
