"""Factory for creating approach instances."""

from typing import List, Type, Dict
from .base import Approach
from .coretweets import CoRetweetsApproach
from .coretweets_numpy import CoRetweetsNumpyApproach
from .coretweets_fast import CoRetweetsFastApproach
from .ignoring_tweet import IgnoringTweetApproach
from .ignoring_tweet_fast import IgnoringTweetFastApproach
from .shared_tweets import SharedTweetsApproach
from .same_tweet_same_time import SameTweetSameTimeApproach


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
    }

    @classmethod
    def create(cls, approach_key: str, window_sec: int = 60,
               min_coactions: int = 1) -> Approach:
        """
        Create an approach instance by key.

        Args:
            approach_key: Key identifying the approach
            window_sec: Time window in seconds
            min_coactions: Minimum co-retweets to consider a pair

        Returns:
            Instance of the requested approach

        Raises:
            ValueError: If approach_key is unknown
        """
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
