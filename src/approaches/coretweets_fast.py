"""Co-retweet counting approach."""

from typing import Counter, Dict, Tuple, List, Optional
from collections import defaultdict, deque
from pathlib import Path
import pickle
from .base import Approach


class CoRetweetsApproach(Approach):
    """
    Co-retweet counting approach.

    Detects pairs of users who retweeted the same tweet within a time window.
    This is the baseline approach that other methods build upon.
    """

    def get_approach_name(self) -> str:
        return "Fast Co-retweets"

    def get_approach_key(self) -> str:
        return "coretweets_fast"

    def needs_filtered_data(self) -> bool:
        """Co-retweets doesn't need pre-filtered data."""
        return False

    @staticmethod
    def _group_RTs_by_tweet(RTs):
        """Group retweets by tweet, sorted by timestamp."""
        tweet_data = defaultdict(list)
        for user, tid, ts in RTs:
            tweet_data[tid].append((user, ts))
        # Sort each tweet's retweets chronologically
        for tid in tweet_data:
            tweet_data[tid].sort(key=lambda x: x[1])
        return tweet_data

    def _get_cache_path(self, processed_dir: Path) -> Path:
        """
        Get the cache file path for pairs_coretweets.

        Args:
            processed_dir: Path to processed dataset directory

        Returns:
            Path to the cache file
        """
        processed_dir = Path(processed_dir)
        cache_filename = f"pairs_coretweets_w{self.window_sec}_m{self.min_coactions}.pkl"
        return processed_dir / cache_filename

    def _is_cache_valid(self, processed_dir: Path) -> bool:
        """
        Check if pairs_coretweets cache exists and is up-to-date.

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

        # Source files that pairs_coretweets depends on
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

    def _load_cache(self, processed_dir: Path) -> Dict[Tuple, int]:
        """
        Load pairs_coretweets from pickle cache file.

        Args:
            processed_dir: Path to processed dataset directory

        Returns:
            Dictionary mapping user pairs to their co-retweet counts

        Raises:
            FileNotFoundError: If cache file doesn't exist
            pickle.UnpicklingError: If cache is corrupted
        """
        cache_file = self._get_cache_path(processed_dir)

        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['pairs_coretweets']

    def _save_cache(self, processed_dir: Path, pairs_coretweets: Dict[Tuple, int]) -> None:
        """
        Save pairs_coretweets to pickle cache file.

        Args:
            processed_dir: Path to processed dataset directory
            pairs_coretweets: Dictionary mapping user pairs to their co-retweet counts

        Raises:
            IOError: If cache file cannot be written
        """
        cache_file = self._get_cache_path(processed_dir)

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'pairs_coretweets': pairs_coretweets,
                'window_sec': self.window_sec,
                'min_coactions': self.min_coactions
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _compute_pairs_scores_impl(self, RTs: List[Tuple]) -> Dict[Tuple, float]:
        """
        Internal implementation of co-retweet counting.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples

        Returns:
            Dictionary mapping user pairs to their co-retweet counts
        """
        by_tweet = self._group_RTs_by_tweet(RTs)
        y = Counter()

        # For each tweet, find pairs of users who retweeted within the time window
        # Using sliding window approach: O(n*k) where k = avg retweets in window
        for acc_times in by_tweet.values():
            active = deque()  # Deque of (user_id, timestamp) within current window

            for acc_curr, ts_curr in acc_times:
                # Remove retweets outside the time window from the front
                while active and (ts_curr - active[0][1]).total_seconds() > self.window_sec:
                    active.popleft()

                # Count pairs with all users in the active window
                for acc_prev, _ in active:
                    if acc_prev != acc_curr:
                        pair = tuple(sorted((acc_prev, acc_curr)))
                        y[pair] += 1

                # Add current retweet to active window
                active.append((acc_curr, ts_curr))
        # Filter pairs with at least min_coactions co-retweets
        return {k: v for k, v in y.items() if v >= self.min_coactions}

    def compute_pairs_scores(self, RTs: List[Tuple], **kwargs) -> Dict[Tuple, float]:
        """
        Count pairs who retweeted same tweet within time window.

        This method supports transparent caching for performance. When 'processed_dir'
        is provided in kwargs, it will automatically use/create a pickle cache.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Optional parameters:
                - processed_dir: Directory path for caching (enables automatic caching)
                - use_cache: If False, bypass cache (default: True)
                - force_reload: If True, ignore existing cache and recompute (default: False)

        Returns:
            Dictionary mapping user pairs to their co-retweet counts

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

        # Compute pairs_coretweets
        pairs_coretweets = self._compute_pairs_scores_impl(RTs)

        # Save cache for next time if processed_dir provided
        if processed_dir and use_cache:
            try:
                self._save_cache(Path(processed_dir), pairs_coretweets)
            except Exception as e:
                # Don't fail if we can't write cache
                print(f"Warning: Could not write pairs_coretweets cache file: {e}")

        return pairs_coretweets
