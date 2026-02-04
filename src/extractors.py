"""Extractors for interaction datasets.

Strategies to extract (account_id, object_id, timestamp) triples from various data formats (retweets, tweets, hashtags, etc.).
"""

import ast
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

# Type aliases
RawEvent = Tuple[Any, Any, Any]  # (account, object, timestamp) triple


class BaseExtractor(ABC):
    """Strategy to extract zero or more (account_id, object_id, timestamp) triples."""

    name = "base"

    @abstractmethod
    def extract(self, dataset_type: str, row: Dict[str, Any]) -> Iterable[RawEvent]:
        """Extract events from a single row.
        
        Args:
            dataset_type: Type of dataset ('twitter' or 'anonymized')
            row: A single row/record from the dataset
            
        Returns:
            Iterable of (account_id, object_id, timestamp) tuples
        """
        raise NotImplementedError


class RetweetExtractor(BaseExtractor):
    """Extracts retweet interactions from datasets."""

    name = "RTs"

    def _parse_ts_twitter(self, ts_str: str) -> datetime:
        """Parse Twitter API timestamp format (e.g., 2020-04-01T12:34:56.123Z)."""
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _parse_ts_cimaio(self, ts_millis: int) -> datetime:
        """Parse CimaIO timestamp format (milliseconds since epoch)."""
        return datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)

    def extract(self, dataset_type: str, row: Dict[str, Any]) -> Iterable[RawEvent]:
        """Extract retweet events from a row."""
        if dataset_type == "anonymized":
            if not row.get("is_repost"):
                return []
            return [(row["accountid"], row["reposted_postid"], row["post_time"])]

        if dataset_type == "CimaIO":
            if row["retweet_tweetid"] is None:
                return []

            ts = self._parse_ts_cimaio(row["tweet_time"])
            return [(row["userid"], row["retweet_tweetid"], ts)]

        if dataset_type == "twitter":
            ref = row.get("referenced_tweets")
            if not ref:
                return []
            first = ref[0]
            if first.get("type") != "retweeted":
                return []
            post_id = first.get("id")
            ts = row.get("created_at")
            ts = self._parse_ts_twitter(ts) if isinstance(ts, str) else ts
            return [(row.get("author_id"), post_id, ts)]

        raise ValueError("Unsupported dataset_type. Use 'twitter', 'anonymized', or 'CimaIO'.")


class HashtagExtractor(BaseExtractor):
    """Extracts hashtag interactions from datasets."""

    name = "hashtag"

    def _parse_ts_twitter(self, ts_str: str) -> datetime:
        """Parse Twitter API timestamp format (e.g., 2020-04-01T12:34:56.123Z)."""
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _parse_ts_cimaio(self, ts_millis: int) -> datetime:
        """Parse CimaIO timestamp format (milliseconds since epoch)."""
        return datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)

    def extract(self, dataset_type: str, row: Dict[str, Any]) -> Iterable[RawEvent]:
        """Extract hashtag events from a row."""
        events: List[RawEvent] = []

        if dataset_type == "anonymized":
            if row.get("is_repost"):
                return events
            hashtags = row.get("hashtags") or []
            for ht in hashtags:
                if ht:
                    events.append((row["accountid"], ht, row["post_time"]))
            return events

        if dataset_type == "CimaIO":
            if row.get("retweet_tweetid"):
                return events
            hashtags = row.get("hashtags") or []
            ts = self._parse_ts_cimaio(row["tweet_time"])
            for ht in hashtags:
                if ht:
                    events.append((row["userid"], ht, ts))
            return events

        if dataset_type == "twitter":
            ref = row.get("referenced_tweets")
            if not ref:
                return events
            first = ref[0]
            if first.get("type") == "retweeted":
                return events

            raw_hashtags = first.get("hashtags") or []
            ts = row.get("created_at")
            ts = self._parse_ts_twitter(ts) if isinstance(ts, str) else ts
            author_id = row.get("author_id")

            normalized_hashtags: List[Any] = []
            if isinstance(raw_hashtags, list):
                for ht in raw_hashtags:
                    if isinstance(ht, dict):
                        tag = ht.get("tag") or ht.get("text") or ht.get("name")
                        if tag:
                            normalized_hashtags.append(tag)
                    elif ht:
                        normalized_hashtags.append(ht)
            elif raw_hashtags:
                normalized_hashtags.append(raw_hashtags)

            for ht in normalized_hashtags:
                events.append((author_id, ht, ts))
            return events

        raise ValueError("Unsupported dataset_type. Use 'twitter', 'anonymized', or 'CimaIO'.")


class SameContentExtractor(BaseExtractor):
    """Extracts same content interactions by processing tweet text."""

    name = "same_content"

    def _parse_ts_twitter(self, ts_str: str) -> datetime:
        """Parse Twitter API timestamp format (e.g., 2020-04-01T12:34:56.123Z)."""
        return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S.%fZ")

    def _parse_ts_cimaio(self, ts_millis: int) -> datetime:
        """Parse CimaIO timestamp format (milliseconds since epoch)."""
        return datetime.fromtimestamp(ts_millis / 1000.0, tz=timezone.utc)

    def _process_text(self, text: str) -> str:
        """Process text by lowercasing, removing @mentions, and standardizing whitespaces.
        
        Args:
            text: Raw text to process
            
        Returns:
            Processed text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        
        # Standardize whitespaces (replace multiple spaces with single space and strip)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def extract(self, dataset_type: str, row: Dict[str, Any]) -> Iterable[RawEvent]:
        """Extract same content events from a row."""
        events: List[RawEvent] = []

        if dataset_type == "anonymized":
            # Skip retweets
            if row.get("is_repost"):
                return events
            
            # Get and process the post text
            post_text = row.get("post_text")
            if not post_text:
                return events
            
            processed_text = self._process_text(post_text)
            if processed_text:
                events.append((row["accountid"], processed_text, row["post_time"]))
            return events

        if dataset_type == "CimaIO":
            # Skip retweets
            if row.get("retweet_tweetid"):
                return events
            
            # Get and process the tweet text
            text = row.get("tweet_text")
            if not text:
                return events
            
            processed_text = self._process_text(text)
            if processed_text:
                ts = self._parse_ts_cimaio(row["tweet_time"])
                events.append((row["userid"], processed_text, ts))
            return events

        if dataset_type == "twitter":
            # Check if it's a retweet and skip it
            ref = row.get("referenced_tweets")
            if ref:
                first = ref[0]
                if first.get("type") == "retweeted":
                    return events
            
            # Get and process the tweet text
            text = row.get("text")
            if not text:
                return events
            
            processed_text = self._process_text(text)
            if processed_text:
                ts = row.get("created_at")
                ts = self._parse_ts_twitter(ts) if isinstance(ts, str) else ts
                events.append((row.get("author_id"), processed_text, ts))
            return events

        raise ValueError("Unsupported dataset_type. Use 'twitter', 'anonymized', or 'CimaIO'.")
