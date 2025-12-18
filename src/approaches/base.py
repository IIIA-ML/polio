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

    def __init__(self, window_sec: int = 60, min_coactions: int = 1, ranking_mode: str = 'Linf'):
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
        # need_filtering can be set by factory or lexicographic; if not set, defaults to _get_need_filtering()

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
        Return full key including min_coactions/ranking mode if not default, and filtering flag.
        
        This is used for storing results separately for different configurations.
        Format examples:
        - coretweets
        - coretweets_min2
        - consistency_min2_L2
        - consistency_min2_Linf_nofilter (when filtering disabled)
        """
        base_key = self.get_approach_key()
        
        # Add min_coactions to key if > 1
        if self.min_coactions > 1:
            base_key = f"{base_key}_min{self.min_coactions}"
        
        # Add ranking mode to key
        base_key = f"{base_key}_{self.ranking_mode}"
        
        # Add nofilter suffix when filtering is disabled
        if not self._get_need_filtering():
            base_key = f"{base_key}_nofilter"
        
        return base_key
    
    def get_full_approach_name(self) -> str:
        """
        Return display name including min_coactions and ranking mode if not default.
        
        This is used for displaying approaches with different min_coactions and ranking modes
        as separate entries in analysis and plots.
        
        Examples:
        - "Vol. [Linf]" (default: min_coactions=1, ranking_mode=Linf)
        - "Vol. min2 [Linf]" (min_coactions=2, ranking_mode=Linf)
        - "Vol. [L2]" (min_coactions=1, ranking_mode=L2)
        - "Vol. min2 [L2]" (min_coactions=2, ranking_mode=L2)
        """
        base_name = self.get_approach_name()
        
        # Add min_coactions suffix if > 1
        if self.min_coactions > 1:
            base_name = f"{base_name} min{self.min_coactions}"
        
        # Always add ranking mode in brackets
        base_name = f"{base_name} [{self.ranking_mode}]"
        
        # Add NoFilter suffix if filtering disabled
        if not self._get_need_filtering():
            base_name = f"{base_name} NoFilter"
        
        return base_name

    def needs_filtered_data(self) -> bool:
        """
        Return whether this approach needs data filtered by coretweets.

        Default is True for most approaches except coretweets itself.
        """
        return self.get_approach_key() != 'coretweets'
    
    def _get_need_filtering(self) -> bool:
        """
        Get the need_filtering flag for this approach.
        
        Checks the need_filtering attribute if set explicitly (by factory or lexicographic),
        otherwise defaults to needs_filtered_data().
        
        Returns:
            True if data should be filtered, False otherwise
        """
        if hasattr(self, 'need_filtering'):
            return self.need_filtering
        return self.get_approach_key() != 'coretweets'
    
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
    
    # Result caching methods for use in run_experiments and lexicographic
    
    def _get_result_cache_path(self, output_dir: str, dataset: str) -> Path:
        """
        Get the cache file path for this approach's results on a dataset.

        Args:
            output_dir: Experiment output directory (results are stored in {output_dir}/results)
            dataset: Dataset name

        Returns:
            Path to the cache file for this approach+dataset combination
        """
        output_dir = Path(output_dir)
        approach_key = self.get_full_approach_key()
        dataset_dir = output_dir / "results" / dataset
        return dataset_dir / f"{approach_key}.pkl"

    def is_result_cached(self, output_dir: str, dataset: str) -> bool:
        """
        Check if results for this approach on a dataset are already cached.

        Args:
            output_dir: Experiment output directory
            dataset: Dataset name

        Returns:
            True if cache exists, False otherwise
        """
        cache_path = self._get_result_cache_path(output_dir, dataset)
        return cache_path.exists()

    def load_cached_results(self, output_dir: str, dataset: str) -> Dict[str, Any]:
        """
        Load previously computed results for this approach on a dataset.

        Args:
            output_dir: Experiment output directory
            dataset: Dataset name

        Returns:
            Dictionary containing the cached results

        Raises:
            FileNotFoundError: If cache doesn't exist
            pickle.UnpicklingError: If cache is corrupted
        """
        cache_path = self._get_result_cache_path(output_dir, dataset)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    def save_results(self, output_dir: str, dataset: str, results: Dict[str, Any]) -> None:
        """
        Save computed results for this approach on a dataset.

        Args:
            output_dir: Experiment output directory
            dataset: Dataset name
            results: Dictionary containing the results to cache

        Raises:
            IOError: If results cannot be written
        """
        cache_path = self._get_result_cache_path(output_dir, dataset)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    
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
    """Abstract base class for approaches that compute pair scores."""

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
        Compute pair scores.

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters

        Returns:
            Dictionary mapping user pairs to their scores
        """
        return self._compute_pairs_scores_impl(RTs, **kwargs)
    
    def get_suspicious(self, RTs: List[Tuple], **kwargs) -> List[List[int]]:
        """
        Get ordered list of suspicious users grouped by score level.

        Returns a list of lists where each inner list contains users from pairs
        at the same score level, ordered from highest to lowest scores.
        Each user appears only once (in their highest-scoring group).

        Supports transparent caching when output_dir and dataset are provided.

        The ranking mode determines how user scores are computed:
        - 'max': each user's score is the maximum score among all their pairs
        - 'L1': each user's score is the median score among all their pairs
        - 'L2': each user's score is the mean of scores from all their pairs (default)
        - 'Linf': each user's score is the midrange of scores from all their pairs
        - 'LX' (X>=3): each user's score minimizes the Lp-norm with p=X

        Args:
            RTs: List of (user_id, tweet_id, timestamp) tuples
            **kwargs: Additional approach-specific parameters including:
                - output_dir: Experiment output directory (enables result caching)
                - dataset: Dataset name (required for caching)
                - io_users: Known inauthentic users (included in cached results)
                - filter_min_coactions: Override min_coactions for filtering
                - Other approach-specific parameters

        Returns:
            List of lists: [[users_at_highest_score], [users_at_next_score], ...]
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

        # Filter RTs if needed before computing pair scores
        if self._get_need_filtering():
            # Allow overriding the min_coactions used for the coretweets filter via kwargs
            # Default to 1 if not provided or if None is passed
            filter_min_coactions = kwargs.get('filter_min_coactions', 1)
            if filter_min_coactions is None:
                filter_min_coactions = 1
            RTs_to_use = filter_RTs(RTs, self.window_sec, filter_min_coactions)
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

        # Cache results for next time if output_dir and dataset provided
        if output_dir and dataset and ordered_list_of_suspicious_users:
            try:
                results = {
                    'dataset': dataset,
                    'suspicious_users': ordered_list_of_suspicious_users,
                    'io_users': io_users,
                    'num_suspicious_groups': len(ordered_list_of_suspicious_users),
                    'num_suspicious_users': sum(len(group) for group in ordered_list_of_suspicious_users),
                    **self.get_metadata() # Includes all approach configuration
                }
                self.save_results(output_dir, dataset, results)
            except Exception:
                # Don't fail if we can't save cache
                pass

        return ordered_list_of_suspicious_users

