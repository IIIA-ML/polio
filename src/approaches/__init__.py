"""
Approaches module for coordinated behavior detection.

This module provides a uniform interface for different detection approaches
through the Approach base class and its concrete implementations.
"""

from .base import Approach
from .coretweets import CoRetweetsApproach
from .coretweets_numpy import CoRetweetsNumpyApproach
from .coretweets_fast import CoRetweetsFastApproach
from .ignoring_tweet import IgnoringTweetApproach
from .ignoring_tweet_fast import IgnoringTweetFastApproach
from .shared_tweets import SharedTweetsApproach
from .same_tweet_same_time import SameTweetSameTimeApproach
from .factory import ApproachFactory

__all__ = [
    'Approach',
    'CoRetweetsApproach',
    'CoRetweetsNumpyApproach',
    'CoRetweetsFastApproach',
    'IgnoringTweetApproach',
    'IgnoringTweetFastApproach',
    'SharedTweetsApproach',
    'SameTweetSameTimeApproach',
    'ApproachFactory',
]
