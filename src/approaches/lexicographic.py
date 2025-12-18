"""Lexicographic approach: hierarchical refinement using multiple approaches."""

from typing import Dict, Tuple, List, Any
from .base import Approach
from synchronous_repeated_detection import filter_RTs


class LexicographicApproach(Approach):
    """
    Combines multiple approaches using hierarchical refinement.

    Starting with groups from the first approach, each subsequent approach
    refines those groups by splitting users who are at different rank levels.
    This creates a lexicographic ordering where the first approach is primary,
    and subsequent approaches act as progressive refinements.
    """

    def __init__(self, approach_keys: List[str], window_sec: int = 60, min_coactions = None, ranking_mode: str = 'Linf', sub_modes: List[str] = None, sub_need_filtering: List[bool] = None):
        """
        Initialize LexicographicApproach with a list of sub-approaches.

        Args:
            approach_keys: List of approach keys to combine (e.g., ['coretweets', 'ignoring_tweet_fast'])
            window_sec: Time window in seconds for synchronous actions
            min_coactions: Minimum co-retweets to consider a pair. Can be:
                - int: applied to all sub-approaches
                - List[int]: one value per sub-approach (must match length of approach_keys)
            ranking_mode: Default ranking mode (unused in new format, but kept for compatibility)
            sub_modes: List of ranking modes per sub-approach (must match length of approach_keys)

        Raises:
            ValueError: If fewer than 2 approaches are provided or sub_modes is None/wrong length
        """
        # Handle min_coactions conversion for parent class
        if isinstance(min_coactions, list):
            min_coactions_for_parent = min_coactions[0] if min_coactions else 1
        else:
            min_coactions_for_parent = min_coactions if min_coactions is not None else 1

        super().__init__(window_sec, min_coactions_for_parent, ranking_mode)

        if len(approach_keys) < 2:
            raise ValueError("LexicographicApproach requires at least 2 approaches")

        if sub_modes is None:
            raise ValueError(
                "LexicographicApproach now requires explicit ranking modes for each sub-approach. "
                "Use format: 'lexicographic:approach1[mode1]+approach2[mode2]+...'"
            )

        if len(sub_modes) != len(approach_keys):
            raise ValueError(
                f"Length of sub_modes ({len(sub_modes)}) must match "
                f"length of approach_keys ({len(approach_keys)})"
            )

        self.approach_keys = approach_keys
        self.sub_modes = sub_modes

        # Convert min_coactions to list if needed
        if isinstance(min_coactions, list):
            sub_min_coactions = min_coactions
        elif isinstance(min_coactions, int):
            sub_min_coactions = [min_coactions] * len(approach_keys)
        else:
            sub_min_coactions = [1] * len(approach_keys)

        # Store sub_min_coactions for use in display names
        self.sub_min_coactions = sub_min_coactions

        # Normalize need_filtering flags per sub-approach
        if isinstance(sub_need_filtering, list):
            self.sub_need_filtering = sub_need_filtering
        elif isinstance(sub_need_filtering, bool):
            self.sub_need_filtering = [sub_need_filtering] * len(approach_keys)
        else:
            # default: True for non-coretweets, False for coretweets
            self.sub_need_filtering = [False if k == 'coretweets' else True for k in approach_keys]

        # Instantiate sub-approaches using factory with their specific modes and min_coactions
        from .factory import ApproachFactory
        self.approaches = [
            ApproachFactory.create(key, window_sec, min_coact, mode)
            for key, min_coact, mode in zip(approach_keys, sub_min_coactions, sub_modes)
        ]
        # Attach need_filtering to each created sub-approach
        for appr, key, nf in zip(self.approaches, approach_keys, self.sub_need_filtering):
            try:
                setattr(appr, 'need_filtering', False if key == 'coretweets' else bool(nf))
            except Exception:
                pass

    def get_approach_name(self) -> str:
        """Return human-readable name using each sub-approach's full name with min_coactions and modes."""
        # Use each sub-approach's full name (which includes min_coactions if > 1)
        # and append the ranking mode in brackets
        parts = []
        for approach, mode in zip(self.approaches, self.sub_modes):
            # Get base name (without any mode suffix)
            base_name = approach.get_approach_name()
            
            # Add min_coactions suffix if > 1
            if approach.min_coactions > 1:
                base_name = f"{base_name} min{approach.min_coactions}"
            
            # Add NoFilter suffix if filtering disabled (but never show for coretweets)
            key = approach.get_approach_key() if hasattr(approach, 'get_approach_key') else ''
            nf = getattr(approach, 'need_filtering', None)
            if nf is None:
                try:
                    nf = approach._get_need_filtering()
                except Exception:
                    nf = True
            if (nf is False) and (key != 'coretweets'):
                base_name = f"{base_name} NoFilter"
            
            # Always add ranking mode in brackets
            base_name = f"{base_name} [{mode}]"
            parts.append(base_name)
        
        return f"Lexicographic: {'+'.join(parts)}"

    def get_approach_key(self) -> str:
        """Return storage key including min_coactions, modes, and nofilter per sub-approach."""
        parts = []
        for idx, (key, min_coact, mode) in enumerate(zip(self.approach_keys, self.sub_min_coactions, self.sub_modes)):
            # Base part with min and mode
            if min_coact > 1:
                part = f"{key}_min{min_coact}_{mode}"
            else:
                part = f"{key}_{mode}"
            # Append _nofilter when filtering disabled for this sub-approach (except coretweets)
            add_nofilter = False
            # Prefer explicit flag on sub approach instance; fallback to provided list; then to _get_need_filtering
            try:
                sub_appr = self.approaches[idx]
            except Exception:
                sub_appr = None
            if sub_appr is not None and hasattr(sub_appr, 'need_filtering'):
                add_nofilter = (getattr(sub_appr, 'need_filtering') is False)
            elif isinstance(getattr(self, 'sub_need_filtering', None), list) and idx < len(self.sub_need_filtering):
                add_nofilter = (self.sub_need_filtering[idx] is False)
            else:
                try:
                    add_nofilter = (sub_appr is not None and not sub_appr._get_need_filtering())
                except Exception:
                    add_nofilter = False
            if add_nofilter:
                part = part + "_nofilter"
            parts.append(part)
        return f"lexicographic_{'_'.join(parts)}"

    def get_full_approach_key(self) -> str:
        """
        Return full key for storage.
        
        For lexicographic, the key already includes all mode information,
        so we don't append the default ranking_mode like the base class does.
        """
        return self.get_approach_key()

    def get_full_approach_name(self) -> str:
        """
        Return display name.
        
        For lexicographic, the name already includes all min_coactions and mode information
        from each sub-approach, so we don't need to add anything extra.
        """
        return self.get_approach_name()

    def get_metadata(self) -> Dict[str, Any]:
        """Include sub-approach information in metadata."""
        metadata = super().get_metadata()
        metadata['sub_approaches'] = self.approach_keys
        metadata['sub_modes'] = self.sub_modes
        metadata['sub_min_coactions'] = self.sub_min_coactions
        metadata['sub_approach_modes'] = dict(zip(self.approach_keys, self.sub_modes))
        metadata['sub_approach_min_coactions'] = dict(zip(self.approach_keys, self.sub_min_coactions))
        return metadata

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

        Caching is handled transparently both for sub-approaches and for the
        lexicographic approach itself when output_dir and dataset are provided.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters including:
                - output_dir: Optional experiment output directory for caching
                - dataset: Dataset name (needed when caching)
                - data_dir: Optional data directory
                - io_users: Optional inauthentic users (passed to sub-approaches)

        Returns:
            List of lists: progressively refined user groups
        """
        # Check cache first if output_dir and dataset provided
        output_dir = kwargs.get('output_dir')
        dataset = kwargs.get('dataset')
        io_users = kwargs.get('io_users')

        if output_dir and dataset and self.is_result_cached(output_dir, dataset):
            try:
                cached_results = self.load_cached_results(output_dir, dataset)
                return cached_results.get('suspicious_users', [])
            except Exception:
                # If loading fails, fall through to computation
                pass

        # First approach - caching handled transparently inside get_suspicious
        current_groups = self.approaches[0].get_suspicious(RTs, **kwargs)

        if not current_groups:
            return []

        # Refine with each subsequent approach
        for approach in self.approaches[1:]:
            # Each approach handles its own caching transparently
            approach_suspicious = approach.get_suspicious(RTs, **kwargs)

            if not approach_suspicious:
                # If approach returns nothing, skip refinement
                continue

            # Convert to ranks: {user_id: rank}
            ranks = Approach.get_ranks(approach_suspicious)

            # Refine current groups using these ranks
            current_groups = self._refine_groups(current_groups, ranks)

        # Cache the lexicographic results for next time
        if output_dir and dataset and current_groups:
            try:
                results = {
                    'dataset': dataset,
                    'suspicious_users': current_groups,
                    'io_users': io_users,
                    'num_suspicious_groups': len(current_groups),
                    'num_suspicious_users': sum(len(group) for group in current_groups),
                    **self.get_metadata()
                }
                self.save_results(output_dir, dataset, results)
            except Exception:
                # Don't fail if we can't save cache
                pass

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
