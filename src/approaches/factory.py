"""Factory for creating approach instances."""

from typing import List, Type, Dict, Optional

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
               min_coactions: int = 1, ranking_mode: str = 'L2') -> Approach:
        """
        Create an approach instance by key.

        Supported formats:
        - Regular: "approach" or "approach[mode]"
        - Lexicographic: "lexicographic:approach1[mode1]+approach2[mode2]+..."
          Example: "lexicographic:coretweets[Linf]+ignoring_tweet_fast[L1]"
        """

        default_ranking_mode = ranking_mode

        # 1) Lexicographic handling
        if approach_key.startswith('lexicographic:'):
            spec = approach_key.split(':', 1)[1]

            # Parse: approach1[mode1]+approach2[mode2]+...
            tokens = spec.split('+')
            if len(tokens) < 2:
                raise ValueError(
                    f"Invalid lexicographic approach '{approach_key}'. Format: 'lexicographic:approach1[mode1]+approach2[mode2]+...'"
                )

            sub_keys: List[str] = []
            sub_modes: List[str] = []
            valid_keys = [k for k in cls._approaches.keys() if k != 'lexicographic']

            for tok in tokens:
                tok = tok.strip()
                if not '[' in tok or not tok.endswith(']'):
                    raise ValueError(
                        f"Invalid lexicographic token '{tok}'. Format: 'approach[mode]' (each sub-approach must have a ranking mode)"
                    )

                base, mode_part = tok.rsplit('[', 1)
                base = base.strip()
                mode = mode_part[:-1].strip()

                # Validate approach key
                if base not in valid_keys:
                    raise ValueError(
                        f"Unknown sub-approach '{base}' in lexicographic. Valid approaches: {', '.join(valid_keys)}"
                    )

                # Validate ranking mode
                if not cls._is_valid_ranking_mode(mode):
                    raise ValueError(f"Invalid ranking mode '{mode}' in '{approach_key}'")

                sub_keys.append(base)
                sub_modes.append(mode)

            return LexicographicApproach(sub_keys, window_sec, min_coactions, default_ranking_mode, sub_modes)

        # 2) Regular approaches with optional [mode]
        if '[' in approach_key and approach_key.endswith(']'):
            base_key, mode_part = approach_key.rsplit('[', 1)
            mode = mode_part.rstrip(']')
            if not cls._is_valid_ranking_mode(mode):
                raise ValueError(f"Invalid ranking mode '{mode}' in '{approach_key}'")
            approach_key = base_key
            ranking_mode = mode

        # 3) Regular approach creation
        if approach_key not in cls._approaches:
            valid_keys = ', '.join(cls._approaches.keys())
            raise ValueError(f"Unknown approach: {approach_key}. Valid keys: {valid_keys}")

        approach_class = cls._approaches[approach_key]
        return approach_class(window_sec, min_coactions, ranking_mode)

    @classmethod
    def get_all_keys(cls) -> List[str]:
        """Get all available approach keys."""
        return list(cls._approaches.keys())

    @classmethod
    def get_all_approaches(cls, window_sec: int = 60,
                          min_coactions: int = 1, ranking_mode: str = 'L2') -> List[Approach]:
        """
        Create instances of all available approaches.

        Args:
            window_sec: Time window in seconds
            min_coactions: Minimum co-retweets to consider a pair
            ranking_mode: How to aggregate pair scores for user ranking

        Returns:
            List of all approach instances
        """
        return [cls.create(key, window_sec, min_coactions, ranking_mode)
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

        Supports regular approach keys and lexicographic syntax.

        Args:
            approach_key: Key to validate (e.g., "coretweets", "coretweets[L2]", 
                         or "lexicographic:coretweets[L1]+ignoring_tweet_fast[L2]")

        Raises:
            ValueError: If the approach key is invalid
        """
        # Lexicographic validation
        if approach_key.startswith('lexicographic:'):
            spec = approach_key.split(':', 1)[1]

            # Parse: approach1[mode1]+approach2[mode2]+...
            tokens = spec.split('+')
            if len(tokens) < 2:
                raise ValueError(
                    f"Invalid lexicographic approach '{approach_key}'. Format: 'lexicographic:approach1[mode1]+approach2[mode2]+...'"
                )

            valid_keys = [k for k in cls._approaches.keys() if k != 'lexicographic']
            for tok in tokens:
                tok = tok.strip()
                if not '[' in tok or not tok.endswith(']'):
                    raise ValueError(
                        f"Invalid lexicographic token '{tok}'. Format: 'approach[mode]' (each sub-approach must have a ranking mode)"
                    )

                base, mode_part = tok.rsplit('[', 1)
                base = base.strip()
                mode = mode_part[:-1].strip()

                if base not in valid_keys:
                    raise ValueError(
                        f"Invalid sub-approach '{base}' in lexicographic approach. Valid approaches: {', '.join(valid_keys)}"
                    )

                if not cls._is_valid_ranking_mode(mode):
                    raise ValueError(f"Invalid ranking mode '{mode}' in '{approach_key}'")
            return

        # Regular approach with optional [mode]
        base_key = approach_key
        if '[' in approach_key and approach_key.endswith(']'):
            base_key, mode_part = approach_key.rsplit('[', 1)
            mode_part = mode_part.rstrip(']')
            if not cls._is_valid_ranking_mode(mode_part):
                raise ValueError(f"Invalid ranking mode '{mode_part}' in '{approach_key}'")

        if base_key not in cls._approaches:
            valid_keys = ', '.join(cls._approaches.keys())
            raise ValueError(
                f"Invalid approach '{base_key}'. Valid approaches: {valid_keys}"
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
