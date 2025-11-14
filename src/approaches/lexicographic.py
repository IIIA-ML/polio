"""Lexicographic approach: hierarchical refinement using multiple approaches."""

from typing import Dict, Tuple, List, Any
from .base import Approach


class LexicographicApproach(Approach):
    """
    Combines multiple approaches using hierarchical refinement.

    Starting with groups from the first approach, each subsequent approach
    refines those groups by splitting users who are at different rank levels.
    This creates a lexicographic ordering where the first approach is primary,
    and subsequent approaches act as progressive refinements.
    """

    def __init__(self, approach_keys: List[str], window_sec: int = 60, min_coactions: int = 1):
        """
        Initialize LexicographicApproach with a list of sub-approaches.

        Args:
            approach_keys: List of approach keys to combine (e.g., ['ignoring_tweet_fast', 'shared_tweets'])
            window_sec: Time window in seconds for synchronous actions
            min_coactions: Minimum co-retweets to consider a pair

        Raises:
            ValueError: If fewer than 2 approaches are provided
        """
        super().__init__(window_sec, min_coactions)

        if len(approach_keys) < 2:
            raise ValueError("LexicographicApproach requires at least 2 approaches")

        self.approach_keys = approach_keys

        # Instantiate sub-approaches using factory
        from .factory import ApproachFactory
        self.approaches = [
            ApproachFactory.create(key, window_sec, min_coactions)
            for key in approach_keys
        ]

    def get_approach_name(self) -> str:
        """Return human-readable name."""
        approach_names = [a.get_approach_name() for a in self.approaches]
        return f"Lexicographic ({', '.join(approach_names)})"

    def get_approach_key(self) -> str:
        """Return storage key."""
        return f"lexicographic_{'_'.join(self.approach_keys)}"

    def get_metadata(self) -> Dict[str, Any]:
        """Include sub-approach information in metadata."""
        metadata = super().get_metadata()
        metadata['sub_approaches'] = self.approach_keys
        return metadata

    def needs_filtered_data(self) -> bool:
        """Depends on whether first approach needs filtered data."""
        return self.approaches[0].needs_filtered_data()

    def get_suspicious(self, RTs: List[Tuple], **kwargs) -> List[List[int]]:
        """
        Hierarchically refine suspicious users using multiple approaches.

        Process:
        1. Start with groups from first approach
        2. For each subsequent approach:
           - Get its suspicious user grouping
           - Convert to ranks using Approach.get_ranks()
           - Split current groups based on rank differences
        3. Return final refined groups

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            List of lists: progressively refined user groups
        """
        # Get initial grouping from first approach
        current_groups = self.approaches[0].get_suspicious(RTs, **kwargs)

        if not current_groups:
            return []

        # Refine with each subsequent approach
        for approach in self.approaches[1:]:
            # Get this approach's grouping and convert to ranks
            approach_suspicious = approach.get_suspicious(RTs, **kwargs)

            if not approach_suspicious:
                # If approach returns nothing, skip refinement
                continue

            # Convert to ranks: {user_id: rank}
            ranks = Approach.get_ranks(approach_suspicious)

            # Refine current groups using these ranks
            current_groups = self._refine_groups(current_groups, ranks)

        return current_groups

    def _refine_groups(self, current_groups: List[List[int]], ranks: Dict[int, int]) -> List[List[int]]:
        """
        Split each group based on rank differences.

        Users in the same group who have different ranks in the refining approach
        will be split into separate groups, maintaining rank order (lower rank = more suspicious).

        Args:
            current_groups: Current list of user groups
            ranks: Mapping from user_id to rank in the refining approach (0 = most suspicious)

        Returns:
            Refined groups where users are split by their ranks
        """
        refined_groups = []

        for group in current_groups:
            if len(group) <= 1:
                # Single user or empty group, no need to split
                refined_groups.append(group)
                continue

            # Group users by their rank in the refining approach
            # Users not in ranks dict get None (appear last)
            rank_to_users = {}
            unranked_users = []

            for user in group:
                if user in ranks:
                    rank = ranks[user]
                    if rank not in rank_to_users:
                        rank_to_users[rank] = []
                    rank_to_users[rank].append(user)
                else:
                    # User not found in refining approach
                    unranked_users.append(user)

            # Add subgroups ordered by rank (0 = most suspicious first)
            for rank in sorted(rank_to_users.keys()):
                refined_groups.append(sorted(rank_to_users[rank]))

            # Add unranked users last if any
            if unranked_users:
                refined_groups.append(sorted(unranked_users))

        return refined_groups
