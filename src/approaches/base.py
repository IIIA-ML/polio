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
    
    @abstractmethod
    def get_suspicious(self, RTs: List[Tuple], **kwargs) -> List[List[int]]:
        """
        Get ordered list of suspicious users grouped by score level.

        Returns a list of lists where each inner list contains users from pairs
        at the same score level, ordered from highest to lowest scores.
        Each user appears only once (in their highest-scoring group).

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            List of lists: [[users_at_highest_score], [users_at_next_score], ...]
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_sec={self.window_sec}, min_coactions={self.min_coactions})"

    def __str__(self) -> str:
        return self.get_approach_name()
    

class PairsApproach(Approach):
    """Abstract base class for approaches that compute pair scores."""

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
    
    def get_suspicious(self, RTs: List[Tuple], **kwargs) -> List[List[int]]:
        """
        Get ordered list of suspicious users grouped by score level.

        Returns a list of lists where each inner list contains users from pairs
        at the same score level, ordered from highest to lowest scores.
        Each user appears only once (in their highest-scoring group).

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            List of lists: [[users_at_highest_score], [users_at_next_score], ...]
        """
        # Compute pair scores using the approach's method
        pairs_scores = self.compute_pairs_scores(RTs, **kwargs)

        if not pairs_scores:
            return []

        # Sort pairs by score in descending order
        sorted_pairs = sorted(pairs_scores.items(), key=lambda x: x[1], reverse=True)

        # Track users already seen to avoid duplicates
        users_seen = set()
        ordered_list_of_suspicious_users = []

        idx = 0
        n = len(sorted_pairs)

        # Process pairs in score blocks (all pairs with same score)
        while idx < n:
            current_score = sorted_pairs[idx][1]
            block_users = set()

            # Collect all users from pairs with the same score
            while idx < n and sorted_pairs[idx][1] == current_score:
                pair, _ = sorted_pairs[idx]
                for user in pair:
                    if user not in users_seen and user not in block_users:
                        block_users.add(user)
                idx += 1

            # Add this score block's users as a new list
            if block_users:
                ordered_list_of_suspicious_users.append(sorted(block_users))
                users_seen.update(block_users)

        return ordered_list_of_suspicious_users

