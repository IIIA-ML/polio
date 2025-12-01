"""
Data loading utilities with pickle caching support.

This module provides shared functions for loading datasets with automatic
pickle caching for improved performance.
"""

import ast
import pickle
import datetime
from pathlib import Path
from typing import Tuple, List


def load_from_text_files(processed_dir: Path) -> Tuple[List[Tuple], List[int], dict]:
    """
    Load dataset from text files (original format).

    Args:
        processed_dir: Path to processed dataset directory

    Returns:
        Tuple of (RTs, io_users, index_to_accountid):
        - RTs: List of (user_id, tweet_id, timestamp) tuples
        - io_users: List of inauthentic operation user IDs
        - index_to_accountid: Mapping of user indices to account IDs

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If file parsing fails
    """
    processed_dir = Path(processed_dir)

    # Load user ID mappings
    data_file = processed_dir / "index_to_accountid.txt"
    with open(data_file, "r") as f:
        content = f.readlines()
        dict_str = content[0].split('=', 1)[1]
        index_to_accountid = ast.literal_eval(dict_str)
        accountid_to_index = {v: k for k, v in index_to_accountid.items()}

    # Load inauthentic operation (IO) users list
    data_file = processed_dir / "io_users.txt"
    with open(data_file, "r") as f:
        content = f.readlines()
        io_users = [line.strip() for line in content]
        io_users = [accountid_to_index[acc] for acc in io_users]

    # Load retweet data: (user_id, tweet_id, timestamp)
    data_file = processed_dir / "RTs.txt"
    with open(data_file, "r") as f:
        content = f.readlines()
        dict_str = content[0].split('=', 1)[1].strip()
        RTs = eval(dict_str, {"datetime": datetime})

    return RTs, io_users, index_to_accountid


def save_to_pickle_cache(processed_dir: Path, RTs: List[Tuple], io_users: List[int],
                         index_to_accountid: dict) -> None:
    """
    Save dataset to pickle cache file.

    Args:
        processed_dir: Path to processed dataset directory
        RTs: List of (user_id, tweet_id, timestamp) tuples
        io_users: List of inauthentic operation user IDs
        index_to_accountid: Mapping of user indices to account IDs

    Raises:
        IOError: If cache file cannot be written
    """
    processed_dir = Path(processed_dir)
    pickle_file = processed_dir / "data_cache.pkl"

    with open(pickle_file, 'wb') as f:
        pickle.dump({
            'RTs': RTs,
            'io_users': io_users,
            'index_to_accountid': index_to_accountid
        }, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle_cache(processed_dir: Path) -> Tuple[List[Tuple], List[int]]:
    """
    Load dataset from pickle cache file.

    Args:
        processed_dir: Path to processed dataset directory

    Returns:
        Tuple of (RTs, io_users)

    Raises:
        FileNotFoundError: If cache file doesn't exist
        pickle.UnpicklingError: If cache is corrupted
    """
    processed_dir = Path(processed_dir)
    pickle_file = processed_dir / "data_cache.pkl"

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        return data['RTs'], data['io_users']


def is_cache_valid(processed_dir: Path) -> bool:
    """
    Check if pickle cache exists and is up-to-date.

    Args:
        processed_dir: Path to processed dataset directory

    Returns:
        True if cache exists and is newer than source files
    """
    processed_dir = Path(processed_dir)
    pickle_file = processed_dir / "data_cache.pkl"

    if not pickle_file.exists():
        return False

    # Check if cache is newer than source files
    cache_time = pickle_file.stat().st_mtime
    source_files = [
        processed_dir / "index_to_accountid.txt",
        processed_dir / "io_users.txt",
        processed_dir / "RTs.txt"
    ]

    # Verify all source files exist
    if not all(f.exists() for f in source_files):
        return False

    # Cache is valid if it's newer than all source files
    newest_source = max(f.stat().st_mtime for f in source_files)
    return cache_time >= newest_source


def import_data(processed_dir, use_cache=True, force_reload=False) -> Tuple[List[Tuple], List[int]]:
    """
    Load dataset with automatic pickle caching.

    This is the main entry point for loading datasets. It automatically
    handles caching for optimal performance.

    Args:
        processed_dir: Directory containing processed dataset files
        use_cache: If True, use/create pickle cache (default: True)
        force_reload: If True, ignore existing cache and reload from text (default: False)

    Returns:
        Tuple of (RTs, io_users):
        - RTs: List of (user_id, tweet_id, timestamp) tuples
        - io_users: List of inauthentic operation user IDs

    Performance:
        - First load (text): ~slow (uses eval())
        - Cached load (pickle): ~10-100x faster

    Example:
        >>> RTs, io_users = import_data("../data/Armenia/Processed/")
        >>> print(f"Loaded {len(RTs)} retweets")
    """
    processed_dir = Path(processed_dir)

    # Try loading from cache if enabled
    if use_cache and not force_reload and is_cache_valid(processed_dir):
        try:
            return load_from_pickle_cache(processed_dir)
        except Exception:
            # If cache loading fails, fall through to text loading
            pass

    # Load from text files
    RTs, io_users, index_to_accountid = load_from_text_files(processed_dir)

    # Save cache for next time
    if use_cache:
        try:
            save_to_pickle_cache(processed_dir, RTs, io_users, index_to_accountid)
        except Exception as e:
            # Don't fail if we can't write cache
            print(f"Warning: Could not write cache file: {e}")

    return RTs, io_users
