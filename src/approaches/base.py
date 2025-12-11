"""Base class for coordinated behavior detection approaches."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
from pathlib import Path
import pickle
from scipy.optimize import minimize
from statistics import mean
from statistics import median
import math
from synchronous_repeated_detection import filter_RTs


class Approach(ABC):
    """Abstract base class for all detection approaches."""

    def __init__(self, window_sec: int = 60, min_coactions: int = 1, ranking_mode: str = 'L2'):
        """
        Initialize approach with common parameters.

        Args:
            window_sec: Time window in seconds for synchronous actions
            min_coactions: Minimum co-retweets to consider a pair
            ranking_mode: How to aggregate pair scores for user ranking.
                         'max': rank by maximum of all pair scores
                         'L1': rank by median of all pairs scores
                         'L2': rank by mean of all pair scores
                         'Linf': rank by midrange of all pair scores
                         'LX' (X>=3): rank using Lp-norm minimization with p=X
        """
        if not self._is_valid_ranking_mode(ranking_mode):
            raise ValueError(f"ranking_mode must be 'max', 'L1', 'L2', 'Linf', or 'LX' (X>=3), got '{ranking_mode}'")
        
        self.window_sec = window_sec
        self.min_coactions = min_coactions
        self.ranking_mode = ranking_mode

    @staticmethod
    def _is_valid_ranking_mode(ranking_mode: str) -> bool:
        """
        Validate if a ranking mode is valid.
        
        Args:
            ranking_mode: The ranking mode string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if ranking_mode in ('L1', 'L2', 'Linf', 'max'):
            return True
        # Check for LX where X is a number >= 3
        if ranking_mode.startswith('L') and len(ranking_mode) > 1:
            try:
                norm_value = int(ranking_mode[1:])
                return norm_value >= 3
            except ValueError:
                return False
        return False

    @staticmethod
    def _minimize_norm(norm: float, data: List[float]) -> float:
        def sum_norm(v):
            return sum(abs(d - v[0]) ** norm for d in data)

        # Initial guess: mean of data
        res = minimize(sum_norm, x0=[mean(data)])
        return round(res.x[0], 2)

    @abstractmethod
    def get_approach_name(self) -> str:
        """Return human-readable name of the approach."""
        pass

    @abstractmethod
    def get_approach_key(self) -> str:
        """Return key used for file storage (e.g., 'coretweets')."""
        pass

    def get_full_approach_key(self) -> str:
        """
        Return full key including ranking mode if not default.
        
        This is used for storing results separately for different ranking modes.
        """
        base_key = self.get_approach_key()
        return f"{base_key}_{self.ranking_mode}"
    
    def get_full_approach_name(self) -> str:
        """
        Return display name including ranking mode if not default.
        
        This is used for displaying approaches with different ranking modes
        as separate entries in analysis and plots.
        """
        base_name = self.get_approach_name()
        return f"{base_name} ({self.ranking_mode})"

    def needs_filtered_data(self) -> bool:
        """
        Return whether this approach needs data filtered by coretweets.

        Default is True for most approaches except coretweets itself.
        """
        return self.get_approach_key() != 'coretweets' # and self.get_approach_key() != 'ignoring_tweet_fast' and self.get_approach_key() != 'shared_tweets' and self.get_approach_key() != 'same_tweet_same_time' and self.get_approach_key() != 'ignoring_tweet_counting_rt'

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this approach's configuration."""
        return {
            'approach_class': self.__class__.__name__,
            'approach_name': self.get_approach_name(),
            'approach_key': self.get_approach_key(),
            'window_sec': self.window_sec,
            'min_coactions': self.min_coactions,
            'ranking_mode': self.ranking_mode,
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
        return f"{self.__class__.__name__}(window_sec={self.window_sec}, min_coactions={self.min_coactions}, ranking_mode='{self.ranking_mode}')"

    def __str__(self) -> str:
        return self.get_approach_name()
    
    @staticmethod
    def compute_user_scores_from_pairs(pairs_scores: Dict[Tuple, float], ranking_mode: str) -> Dict[int, float]:
        """
        Compute user scores from pair scores based on ranking mode.

        Args:
            pairs_scores: Dictionary mapping user pairs to their scores
            ranking_mode: How to aggregate pair scores for each user:
                         'max': maximum of all pair scores for the user
                         'L1': median of all pair scores for the user
                         'L2': mean of all pair scores for the user
                         'Linf': midrange (average of min and max) of all pair scores for the user
                         'LX' (X>=3): minimize Lp-norm with p=X over all pair scores

        Returns:
            Dictionary mapping user_id to their aggregated score
        """
        
        if not Approach._is_valid_ranking_mode(ranking_mode):
            raise ValueError(f"ranking_mode must be 'max', 'L1', 'L2', 'Linf', or 'LX' (X>=3), got '{ranking_mode}'")
        
        # Collect all scores for each user
        user_pair_scores = {}  # {user_id: [list of scores from pairs involving this user]}
        
        for pair, score in pairs_scores.items():
            if score != 0:
                for user in pair:
                    if user not in user_pair_scores:
                        user_pair_scores[user] = []
                    user_pair_scores[user].append(score)
            
        # Compute aggregated score for each user based on ranking mode
        user_scores = {}
        for user, scores in user_pair_scores.items():
            if len(scores) == 1:
                # Only one pair score, use it directly
                user_scores[user] = scores[0]
                continue
            if ranking_mode == 'L1':
                # L1: median of all pair scores
                user_scores[user] = median(scores)
            elif ranking_mode == 'L2':
                # L2: mean of all pair scores
                user_scores[user] = sum(scores) / len(scores)
            elif ranking_mode == 'Linf':
                # Linf: midrange (average of min and max)
                user_scores[user] = (min(scores) + max(scores)) / 2
            elif ranking_mode == 'max':
                # Special case for L-infinity norm
                user_scores[user] = max(scores)
            else:
                # LX where X >= 3: minimize Lp-norm
                norm_value = int(ranking_mode[1:])
                user_scores[user] = Approach._minimize_norm(norm_value, scores)
        
        return user_scores

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

        The ranking mode determines how user scores are computed:
        - 'max': each user's score is the maximum score among all their pairs
        - 'L1': each user's score is the median score among all their pairs
        - 'L2': each user's score is the mean of scores from all their pairs (default)
        - 'Linf': each user's score is the midrange of scores from all their pairs
        - 'LX' (X>=3): each user's score minimizes the Lp-norm with p=X

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            List of lists: [[users_at_highest_score], [users_at_next_score], ...]
        """
        # Compute pair scores using the approach's method

        if self.needs_filtered_data():
            RTs_to_use = filter_RTs(RTs, self.window_sec, self.min_coactions)
        else:
            RTs_to_use = RTs

        pairs_scores = self.compute_pairs_scores(RTs_to_use, **kwargs)

        if not pairs_scores:
            return []

        # Compute user scores based on ranking mode
        user_scores = Approach.compute_user_scores_from_pairs(pairs_scores, self.ranking_mode)

        # Sort users by score (descending)
        sorted_users = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

        # Group users with the same score
        ordered_list_of_suspicious_users = []
        idx = 0
        n = len(sorted_users)

        while idx < n:
            current_score = sorted_users[idx][1]
            block_users = []

            # Collect all users with the same score
            while idx < n and sorted_users[idx][1] == current_score:
                block_users.append(sorted_users[idx][0])
                idx += 1

            # Add this score block's users as a new list (sorted for consistency)
            ordered_list_of_suspicious_users.append(sorted(block_users))

        return ordered_list_of_suspicious_users

