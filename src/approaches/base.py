"""Base class for coordinated behavior detection approaches."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
from pathlib import Path
import pickle


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

    @classmethod
    def get_ranks(cls, suspicious: List[List[int]]) -> Dict[int, int]:
        """
        Convert ordered list of suspicious users into a rank mapping.

        Args:
            suspicious: List of lists of user IDs ordered by suspicion level

        Returns:
            Dictionary mapping user_id to their rank (0 = most suspicious)
        """
        ranks = {}
        for rank, users in enumerate(suspicious):
            for user in users:
                ranks[user] = rank
        return ranks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(window_sec={self.window_sec}, min_coactions={self.min_coactions})"

    def __str__(self) -> str:
        return self.get_approach_name()
    

class PairsApproach(Approach):
    """Abstract base class for approaches that compute pair scores with automatic caching."""

    def _get_cache_path(self, processed_dir: Path) -> Path:
        """
        Get the cache file path for this approach.

        Args:
            processed_dir: Path to processed dataset directory

        Returns:
            Path to the cache file
        """
        processed_dir = Path(processed_dir)
        approach_key = self.get_approach_key()
        cache_filename = f"pairs_{approach_key}_w{self.window_sec}_m{self.min_coactions}.pkl"
        return processed_dir / cache_filename

    def _is_cache_valid(self, processed_dir: Path) -> bool:
        """
        Check if pairs cache exists and is up-to-date.

        Args:
            processed_dir: Path to processed dataset directory

        Returns:
            True if cache exists and is newer than source data files
        """
        processed_dir = Path(processed_dir)
        cache_file = self._get_cache_path(processed_dir)

        if not cache_file.exists():
            return False

        # Check if cache is newer than source files
        cache_time = cache_file.stat().st_mtime

        # Source files that pairs depend on
        source_files = [
            processed_dir / "RTs.txt",
            processed_dir / "data_cache.pkl"  # If data_cache exists, it's the effective source
        ]

        # Use whichever source exists (prefer data_cache.pkl if available)
        existing_sources = [f for f in source_files if f.exists()]
        if not existing_sources:
            return False

        # Cache is valid if it's newer than the newest source file
        newest_source = max(f.stat().st_mtime for f in existing_sources)
        return cache_time >= newest_source

    def _load_cache(self, processed_dir: Path) -> Dict[Tuple, float]:
        """
        Load pairs from pickle cache file.

        Args:
            processed_dir: Path to processed dataset directory

        Returns:
            Dictionary mapping user pairs to their scores

        Raises:
            FileNotFoundError: If cache file doesn't exist
            pickle.UnpicklingError: If cache is corrupted
        """
        cache_file = self._get_cache_path(processed_dir)

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['pairs']

    def _save_cache(self, processed_dir: Path, pairs: Dict[Tuple, float]) -> None:
        """
        Save pairs to pickle cache file.

        Args:
            processed_dir: Path to processed dataset directory
            pairs: Dictionary mapping user pairs to their scores

        Raises:
            IOError: If cache file cannot be written
        """
        cache_file = self._get_cache_path(processed_dir)

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'pairs': pairs,
                'approach_key': self.get_approach_key(),
                'window_sec': self.window_sec,
                'min_coactions': self.min_coactions
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def _compute_pairs_scores_impl(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Internal implementation of pair score computation.

        Subclasses must implement this method to perform the actual computation.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            Dictionary mapping user pairs (tuple of sorted user IDs) to their scores
        """
        pass

    def compute_pairs_scores(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Compute pair scores with transparent caching support.

        This method supports automatic caching for performance. When 'processed_dir'
        is provided in kwargs, it will automatically use/create a pickle cache.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Optional parameters:
                - processed_dir: Directory path for caching (enables automatic caching)
                - use_cache: If False, bypass cache (default: True)
                - force_reload: If True, ignore existing cache and recompute (default: False)
                - Additional approach-specific parameters

        Returns:
            Dictionary mapping user pairs to their scores

        Example:
            >>> # Without caching (standard usage)
            >>> approach = CoRetweetsApproach(window_sec=60, min_coactions=1)
            >>> pairs = approach.compute_pairs_scores(RTs)
            >>>
            >>> # With automatic caching (faster for repeated calls)
            >>> pairs = approach.compute_pairs_scores(RTs, processed_dir="../data/Armenia/Processed/")
        """
        processed_dir = kwargs.get('processed_dir')
        use_cache = kwargs.get('use_cache', True)
        force_reload = kwargs.get('force_reload', False)

        # Try loading from cache if enabled and processed_dir provided
        if processed_dir and use_cache and not force_reload:
            processed_dir = Path(processed_dir)
            if self._is_cache_valid(processed_dir):
                try:
                    return self._load_cache(processed_dir)
                except Exception:
                    # If cache loading fails, fall through to computation
                    pass

        # Compute pairs using subclass implementation
        pairs = self._compute_pairs_scores_impl(RTs, **kwargs)

        # Save cache for next time if processed_dir provided
        if processed_dir and use_cache:
            try:
                self._save_cache(Path(processed_dir), pairs)
            except Exception as e:
                # Don't fail if we can't write cache
                print(f"Warning: Could not write pairs cache file: {e}")

        return pairs
    
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

