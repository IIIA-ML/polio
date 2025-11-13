"""Base class for coordinated behavior detection approaches."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any


class Approach(ABC):
    """Abstract base class for all detection approaches."""

    def __init__(self, window_sec: int = 60, min_coactions: int = 1):
        """
        Initialize approach with common parameters.

        Args:
            window_sec: Time window in seconds for synchronous actions
            min_coactions: Minimum co-retweets to consider a pair
        """
        self.window_sec = window_sec
        self.min_coactions = min_coactions

    @abstractmethod
    def compute_pairs_scores(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Compute pair scores for the given retweet data.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            Dictionary mapping user pairs (tuple of sorted user IDs) to their scores
        """
        pass

    @abstractmethod
    def get_approach_name(self) -> str:
        """Return human-readable name of the approach."""
        pass

    @abstractmethod
    def get_approach_key(self) -> str:
        """Return key used for file storage (e.g., 'coretweets')."""
        pass

    def needs_filtered_data(self) -> bool:
        """
        Return whether this approach needs data filtered by coretweets.

        Default is True for most approaches except coretweets itself.
        """
        return self.get_approach_key() != 'coretweets'

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this approach's configuration."""
        return {
            'approach_class': self.__class__.__name__,
            'approach_name': self.get_approach_name(),
            'approach_key': self.get_approach_key(),
            'window_sec': self.window_sec,
            'min_coactions': self.min_coactions,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_sec={self.window_sec}, min_coactions={self.min_coactions})"

    def __str__(self) -> str:
        return self.get_approach_name()
