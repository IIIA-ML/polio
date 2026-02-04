"""Factory for creating approach instances."""

from typing import List, Type, Dict
from .base import Approach
from .coretweets import CoRetweetsApproach
from .lexicographic import LexicographicApproach
from .coretweets_weighted_1day import CoretweetsWeighted1DayApproach
from .coretweets_weighted_2days import CoretweetsWeighted2DaysApproach
from .coretweets_weighted_12h import CoretweetsWeighted12hApproach


class ApproachFactory:
    """Factory for creating and managing approach instances.
    
    To add new approaches, create an approach class inheriting from Approach
    and add it to the _approaches dictionary below with a descriptive key.
    """

    _approaches: Dict[str, Type[Approach]] = {
        'coretweets': CoRetweetsApproach,
        'lexicographic': LexicographicApproach,
        'coretweets_weighted_1day': CoretweetsWeighted1DayApproach,
        'coretweets_weighted_2days': CoretweetsWeighted2DaysApproach,
        'coretweets_weighted_12h': CoretweetsWeighted12hApproach,
    }

    @staticmethod
    def _is_valid_ranking_mode(ranking_mode: str) -> bool:
        """Validate if a ranking mode is valid."""
        if ranking_mode in ('L1', 'L2', 'Linf', 'max'):
            return True
        if ranking_mode.startswith('L') and len(ranking_mode) > 1:
            try:
                return int(ranking_mode[1:]) >= 3
            except ValueError:
                return False
        return False

    @classmethod
    def _validate_sub_approaches(cls, sub_approaches: List[str]) -> None:
        """Validate a list of sub-approach keys."""
        valid_keys = [k for k in cls._approaches.keys() if k != 'lexicographic']
        for sub_key in sub_approaches:
            if sub_key not in valid_keys:
                raise ValueError(f"Unknown sub-approach '{sub_key}'. Valid: {', '.join(valid_keys)}")

    @classmethod
    def _validate_ranking_modes(cls, modes: List[str]) -> None:
        """Validate a list of ranking modes."""
        for mode in modes:
            if not cls._is_valid_ranking_mode(mode):
                raise ValueError(f"Invalid ranking mode '{mode}'")

    @classmethod
    def _create_lexicographic_from_dict(cls, spec: dict, window_sec: int, 
                                       min_coactions: int, ranking_mode: str) -> LexicographicApproach:
        """Create lexicographic approach from dictionary specification."""
        sub_approaches = spec.get('approaches')
        if not sub_approaches or not isinstance(sub_approaches, list):
            raise ValueError("Lexicographic approach must include 'approaches' as a list")
        if len(sub_approaches) < 2:
            raise ValueError("Lexicographic approach requires at least 2 sub-approaches")
        
        # Validate sub-approaches
        cls._validate_sub_approaches(sub_approaches)
        
        # Get ranking modes (can be single value or list)
        sub_modes_spec = spec.get('ranking_modes', ranking_mode)
        if isinstance(sub_modes_spec, str):
            sub_modes = [sub_modes_spec] * len(sub_approaches)
        elif isinstance(sub_modes_spec, list):
            if len(sub_modes_spec) != len(sub_approaches):
                raise ValueError(f"ranking_modes length ({len(sub_modes_spec)}) must match approaches length ({len(sub_approaches)})")
            sub_modes = sub_modes_spec
        else:
            raise ValueError("ranking_modes must be a string or list")
        
        cls._validate_ranking_modes(sub_modes)
        
        # Get min_coactions (can be single value or list)
        sub_min_coactions = spec.get('min_coactions', min_coactions)
        if isinstance(sub_min_coactions, int):
            sub_min_coactions_list = [sub_min_coactions] * len(sub_approaches)
        elif isinstance(sub_min_coactions, list):
            if len(sub_min_coactions) != len(sub_approaches):
                raise ValueError(f"min_coactions length ({len(sub_min_coactions)}) must match approaches length ({len(sub_approaches)})")
            sub_min_coactions_list = sub_min_coactions
        else:
            raise ValueError("min_coactions must be an integer or list")
        
        # Get need_filtering (can be single value or list)
        sub_need_filtering = spec.get('need_filtering', True)
        if isinstance(sub_need_filtering, bool):
            sub_need_filtering_list = [sub_need_filtering] * len(sub_approaches)
        elif isinstance(sub_need_filtering, list):
            if len(sub_need_filtering) != len(sub_approaches):
                raise ValueError(f"need_filtering length ({len(sub_need_filtering)}) must match approaches length ({len(sub_approaches)})")
            # validate booleans
            for nf in sub_need_filtering:
                if not isinstance(nf, bool):
                    raise ValueError(f"need_filtering values must be booleans, got {nf}")
            sub_need_filtering_list = sub_need_filtering
        else:
            raise ValueError("need_filtering must be a boolean or list")
        
        return LexicographicApproach(sub_approaches, window_sec, sub_min_coactions_list, ranking_mode, sub_modes, sub_need_filtering_list)

    @classmethod
    def create(cls, approach_spec, window_sec: int = 60,
               min_coactions: int = 1, ranking_mode: str = 'Linf') -> Approach:
        """
        Create an approach instance from specification.

        Supported formats:
        - String: "approach" or "approach[mode]"
        - Dictionary: {"name": "approach", "min_coactions": 2, "ranking_mode": "Linf"}
        - Lexicographic dict: {
            "name": "lexicographic",
            "approaches": ["app1", "app2"],
            "ranking_modes": ["Linf", "L1"],  # optional
            "min_coactions": 2  # optional
          }
        """
        # Parse dictionary format
        if isinstance(approach_spec, dict):
            approach_key = approach_spec.get('name')
            if not approach_key:
                raise ValueError("Approach dictionary must include 'name' key")
            
            # Handle lexicographic format
            if approach_key == 'lexicographic':
                return cls._create_lexicographic_from_dict(approach_spec, window_sec, min_coactions, ranking_mode)
            
            # Regular approach - override defaults
            min_coactions = approach_spec.get('min_coactions', min_coactions)
            ranking_mode = approach_spec.get('ranking_mode', ranking_mode)
            need_filtering_spec = approach_spec.get('need_filtering', None)
        else:
            approach_key = approach_spec
            need_filtering_spec = None

        # Handle regular approaches with optional [mode] suffix
        if '[' in approach_key and approach_key.endswith(']'):
            base_key, mode_part = approach_key.rsplit('[', 1)
            mode = mode_part.rstrip(']')
            if not cls._is_valid_ranking_mode(mode):
                raise ValueError(f"Invalid ranking mode '{mode}' in '{approach_key}'")
            approach_key = base_key
            ranking_mode = mode

        # Create regular approach
        if approach_key not in cls._approaches:
            raise ValueError(f"Unknown approach: {approach_key}. Valid keys: {', '.join(cls._approaches.keys())}")

        approach_class = cls._approaches[approach_key]
        instance = approach_class(window_sec, min_coactions, ranking_mode)

        # Attach need_filtering attribute when provided or when coretweets
        try:
            if approach_key == 'coretweets':
                setattr(instance, 'need_filtering', False)
            elif need_filtering_spec is not None:
                # Only accept booleans; ignore otherwise
                if isinstance(need_filtering_spec, bool):
                    setattr(instance, 'need_filtering', need_filtering_spec)
        except Exception:
            # Non-fatal: if the instance doesn't allow attribute, ignore
            pass

        return instance

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
    def validate_approach_key(cls, approach_spec) -> None:
        """
        Validate an approach specification without creating an instance.

        Args:
            approach_spec: String key or dictionary to validate

        Raises:
            ValueError: If the approach specification is invalid
        """
        # Handle dictionary format
        if isinstance(approach_spec, dict):
            if 'name' not in approach_spec:
                raise ValueError("Approach dictionary must include 'name' key")
            approach_key = approach_spec['name']
            
            # Validate lexicographic dictionary format
            if approach_key == 'lexicographic':
                if 'approaches' not in approach_spec:
                    raise ValueError("Lexicographic approach must include 'approaches' list")
                
                sub_approaches = approach_spec['approaches']
                if not isinstance(sub_approaches, list) or len(sub_approaches) < 2:
                    raise ValueError("Lexicographic requires at least 2 sub-approaches as a list")
                
                cls._validate_sub_approaches(sub_approaches)
                
                # Validate ranking modes if provided
                if 'ranking_modes' in approach_spec:
                    ranking_modes = approach_spec['ranking_modes']
                    if isinstance(ranking_modes, list):
                        if len(ranking_modes) != len(sub_approaches):
                            raise ValueError(f"ranking_modes length must match approaches length")
                        cls._validate_ranking_modes(ranking_modes)
                    elif isinstance(ranking_modes, str):
                        if not cls._is_valid_ranking_mode(ranking_modes):
                            raise ValueError(f"Invalid ranking mode: {ranking_modes}")
                    else:
                        raise ValueError("ranking_modes must be a string or list")
                
                # Validate min_coactions if provided
                if 'min_coactions' in approach_spec:
                    min_coactions = approach_spec['min_coactions']
                    if isinstance(min_coactions, list):
                        if len(min_coactions) != len(sub_approaches):
                            raise ValueError(f"min_coactions length must match approaches length")
                        for mc in min_coactions:
                            if not isinstance(mc, int) or mc < 1:
                                raise ValueError(f"min_coactions values must be positive integers, got {mc}")
                    elif isinstance(min_coactions, int):
                        if min_coactions < 1:
                            raise ValueError(f"min_coactions must be a positive integer, got {min_coactions}")
                    else:
                        raise ValueError(f"min_coactions must be an integer or list, got {min_coactions}")
                
                return
            
            # Validate regular approach dictionary
            if 'min_coactions' in approach_spec:
                mc = approach_spec['min_coactions']
                if not isinstance(mc, int) or mc < 1:
                    raise ValueError(f"min_coactions must be a positive integer, got {mc}")
            
            if 'ranking_mode' in approach_spec:
                rm = approach_spec['ranking_mode']
                if not isinstance(rm, str) or not cls._is_valid_ranking_mode(rm):
                    raise ValueError(f"Invalid ranking_mode: {rm}")
            
            if 'need_filtering' in approach_spec:
                nf = approach_spec['need_filtering']
                if not isinstance(nf, bool):
                    raise ValueError(f"need_filtering must be a boolean, got {nf}")
        else:
            approach_key = approach_spec
        
        # Validate regular approach with optional [mode] suffix
        base_key = approach_key
        if '[' in approach_key and approach_key.endswith(']'):
            base_key, mode_part = approach_key.rsplit('[', 1)
            mode = mode_part.rstrip(']')
            if not cls._is_valid_ranking_mode(mode):
                raise ValueError(f"Invalid ranking mode '{mode}' in '{approach_key}'")

        if base_key not in cls._approaches:
            raise ValueError(f"Invalid approach '{base_key}'. Valid: {', '.join(cls._approaches.keys())}")

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
