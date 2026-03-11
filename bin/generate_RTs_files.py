#!/usr/bin/env python3
"""Extract interactions from social media datasets and build indexed event files.

This script processes parquet and JSON datasets using pluggable extractors
(retweets, hashtags, same content) and generates normalized event files with
index mappings and user metadata. Supports anonymized (Seckin datasets), Twitter, and CimaIO formats.

Run: ./bin/generate_RTs_files.py (configure datasets in __main__ section)
"""
import json
from datetime import datetime
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Dict, Any
import pyarrow.parquet as pq
from collections import Counter
import os
import glob
import sys

# Add src directory to path to import extractors
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.extractors import BaseExtractor, RetweetExtractor, HashtagExtractor, SameContentExtractor


Row = Dict[str, Any]
Event = Tuple[int, int, Any]
# Raw (account, object, timestamp) triple emitted by extractors
RawEvent = Tuple[Any, Any, Any]


def _sorted_ids(st: Iterable[Any]) -> List[Any]:
    return sorted(set(st), key=lambda x: str(x))



class RowReader:
    """Yields raw rows from parquet (.gzip.parquet) or NDJSON files.
    
    Supports dataset types: 'anonymized' (Seckin datasets) (parquet), 'twitter' (NDJSON), 'CimaIO' (NDJSON).
    """

    def __init__(self, dataset_type: str, filepaths: Sequence[str]):
        self.dataset_type = dataset_type
        self.filepaths = list(filepaths)

    def _json_row_iterator(self, filename: str) -> Iterator[Row]:
        with open(filename, "r") as infile:
            for line in infile:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _parquet_row_iterator(self, filename: str, columns: Optional[List[str]] = None) -> Iterator[Row]:
        parquet_file = pq.ParquetFile(filename)
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=columns)
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    yield row

    def iter_rows(self) -> Iterator[Row]:
        if self.dataset_type == "twitter":
            for fp in self.filepaths:
                yield from self._json_row_iterator(fp)
        elif self.dataset_type == "anonymized":
            for fp in self.filepaths:
                yield from self._parquet_row_iterator(fp)
        elif self.dataset_type == "CimaIO":
            for fp in self.filepaths:
                yield from self._json_row_iterator(fp)
        else:
            raise ValueError("Unsupported dataset_type. Use 'twitter', 'anonymized', or 'CimaIO'.")


class InteractionDataset:
    """General interaction dataset with pluggable extraction strategy.

    Builds deterministic index mappings by sorting ids by str(id).
    """

    def __init__(self, dataset_type: str, filepaths: Sequence[str], extractor: BaseExtractor):
        self.dataset_type = dataset_type
        self.filepaths = list(filepaths)
        self.extractor = extractor

        self.index_to_accountid: Dict[int, Any] = {}
        self.index_to_objectid: Dict[int, Any] = {}
        self.accountid_to_index: Dict[Any, int] = {}
        self.objectid_to_index: Dict[Any, int] = {}

        self.events: List[Event] = []
        self.user_info: Dict[Any, Any] = {}
        self._raw_events: List[RawEvent] = []

    def _collect_unique_ids(self) -> Tuple[List[Any], List[Any]]:
        """Collect and sort unique IDs from all input files.
        
        IMPORTANT: All files are processed together and IDs are sorted once.
        This ensures the same ID gets the same index regardless of which file it appears in.
        For RetweetExtractor, deduplicates to keep only the earliest event per (account, object) pair.
        """
        reader = RowReader(self.dataset_type, self.filepaths)
        raw_events: List[RawEvent] = []
        
        for row in reader.iter_rows():
            events = self.extractor.extract(self.dataset_type, row) or []
            for acc, obj, ts in events:
                raw_events.append((acc, obj, ts))

                if self.dataset_type == "anonymized":
                    if acc not in self.user_info:
                        # Support both "is_control" (1 = non-IO) and "good" (0 = IO) keys
                        if "is_control" in row:
                            self.user_info[acc] = row["is_control"]
                elif self.dataset_type == "CimaIO":
                    if acc not in self.user_info and "good" in row:
                        # "good": 0 = IO user, "good": 1 = non-IO user
                        self.user_info[acc] = row["good"]
                elif self.dataset_type == "twitter":
                    if acc not in self.user_info and isinstance(row.get("user"), dict):
                        username = row["user"].get("username")
                        if username is not None:
                            self.user_info[acc] = username

        # For RetweetExtractor, keep only the earliest event for each (account, object) pair
        if isinstance(self.extractor, RetweetExtractor):
            acc_obj_earliest: Dict[Tuple[Any, Any], RawEvent] = {}
            for acc, obj, ts in raw_events:
                key = (acc, obj)
                if key not in acc_obj_earliest or ts < acc_obj_earliest[key][2]:
                    acc_obj_earliest[key] = (acc, obj, ts)
            raw_events = list(acc_obj_earliest.values())
        
        # Store deduplicated raw events for use in _collect_events
        self._raw_events = raw_events
        
        # Extract unique accounts and objects from deduplicated events
        accounts = [acc for acc, _, _ in raw_events]
        objects = [obj for _, obj, _ in raw_events]

        # Sort IDs together from ALL files to ensure consistent indexing
        sorted_accounts = _sorted_ids(accounts)
        sorted_objects = _sorted_ids(objects)
        return sorted_accounts, sorted_objects

    def _build_index_mappings(self, accounts: List[Any], objects: List[Any]) -> None:
        self.index_to_accountid = {i: acc for i, acc in enumerate(accounts)}
        self.index_to_objectid = {i: obj for i, obj in enumerate(objects)}
        self.accountid_to_index = {acc: i for i, acc in self.index_to_accountid.items()}
        self.objectid_to_index = {obj: i for i, obj in self.index_to_objectid.items()}

    def _collect_events(self) -> None:
        # Convert stored raw events to indexed events
        evs: List[Event] = []
        for acc, obj, ts in self._raw_events:
            aidx = self.accountid_to_index[acc]
            oidx = self.objectid_to_index[obj]
            evs.append((aidx, oidx, ts))
        
        self.events = evs

    def load(self) -> None:
        accounts, objects = self._collect_unique_ids()
        self._build_index_mappings(accounts, objects)
        self._collect_events()

        unique_accounts = len(self.index_to_accountid)
        unique_objects = len(self.index_to_objectid)
        print(
            f"In the dataset there are:\n  {unique_objects} objects\n  {unique_accounts} accounts\n  {len(self.events)} events"
        )

    def filter_min_participation(self, min_participation: int = 2) -> List[Event]:
        usr_activity = Counter()
        for usr, _, _ in self.events:
            usr_activity[usr] += 1

        filtered: List[Event] = []
        for usr, obj, ts in self.events:
            if usr_activity[usr] >= min_participation:
                filtered.append((usr, obj, ts))
        return filtered

    def save_index_mappings(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "index_to_accountid.txt"), "w") as f:
            f.write(f"index_to_accountid={str(self.index_to_accountid)}\n")
        with open(os.path.join(out_dir, "index_to_objectid.txt"), "w") as f:
            f.write(f"index_to_objectid={str(self.index_to_objectid)}\n")

    def save_user_info(self, out_path: str) -> None:
        with open(out_path, "w") as f:
            for user, is_not_io in self.user_info.items():
                if not is_not_io:
                    # Convert to string to ensure consistent type when reloading
                    f.write(f"{str(user)}\n")

    def save_events(self, out_path: str, label: str = "events") -> None:
        with open(out_path, "w") as f:
            f.write(f"{label}={str(self.events)}")


def process_anonymized_datasets() -> None:
    """Process anonymized parquet datasets (e.g., polio surveillance data).
    
    Available datasets:
    - 'Armenia', 'Bangladesh', 'Catalonia', 'China_1', 'China_2', 'Cuba', 'Ecuador',
    - 'Egypt_UAE', 'Ghana_Nigeria', 'Iran_1'-'Iran_6', 'Qatar', 'Russia_1'-'Russia_5',
    - 'Spain', 'Thailand', 'UAE', 'Venezuela_1', 'Venezuela_2'
    """
    dataset_names = [
        "Catalonia"
        # Uncomment or add other datasets as needed
    ]

    for dataset_name in dataset_names:
        base_dir = f"datasets/{dataset_name}/"
        dataset_files = glob.glob(os.path.join(base_dir, "*.gzip.parquet"))
        dataset_files.sort()

        dataset_type = "anonymized"
        extractor = HashtagExtractor()  # Options: RetweetExtractor(), HashtagExtractor(), SameContentExtractor()
        
        dataset = InteractionDataset(dataset_type, dataset_files, extractor)
        dataset.load()

        processed_dir = f"datasets/{dataset_name}/Processed/{extractor.name}/Processed/"
        os.makedirs(processed_dir, exist_ok=True)

        # Save processed data
        dataset.save_index_mappings(processed_dir)
        dataset.save_user_info(os.path.join(processed_dir, "io_users.txt"))
        dataset.save_events(os.path.join(processed_dir, f"{extractor.name}.txt"), label=extractor.name)
        print(f"Processed {dataset_name}")


def process_cimiao_dataset(region: str, data_type: str = "RTs_Tweets", 
                           extractor: Optional[BaseExtractor] = None) -> None:
    """Process CimaIO dataset for a specific region.
    
    Args:
        region: Region name (e.g., 'Honduras', 'UAE')
        data_type: Data type label (e.g., 'RTs_Tweets')
        extractor: BaseExtractor instance (defaults to RetweetExtractor)
    """
    if extractor is None:
        extractor = RetweetExtractor()
    
    # Define file paths for the region
    files = [
        f"datasets/CimaIO/{region}/{region.lower()}-bad-anonymized.jsonl",
        f"datasets/CimaIO/{region}/{region.lower()}-good-anonymized.jsonl"
    ]
    
    dataset_type = "CimaIO"
    dataset = InteractionDataset(dataset_type, files, extractor)
    dataset.load()

    # Create output directory
    processed_dir = f"datasets/CimaIO/{data_type}/{region}/Processed/"
    os.makedirs(processed_dir, exist_ok=True)

    # Save processed data
    dataset.save_index_mappings(processed_dir)
    dataset.save_user_info(os.path.join(processed_dir, "io_users.txt"))
    dataset.save_events(os.path.join(processed_dir, f"{extractor.name}.txt"), label=extractor.name)
    print(f"Processed CimaIO {region} dataset")


if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION: Select which dataset(s) to process
    # ========================================================================
    
    # Uncomment the function call(s) below to process the desired dataset(s)
    
    # Process anonymized datasets (parquet format)
    # process_anonymized_datasets()
    
    # Process CimaIO datasets
    process_cimiao_dataset(region="Honduras", data_type="RTs", extractor=RetweetExtractor())
    process_cimiao_dataset(region="UAE", data_type="RTs", extractor=RetweetExtractor())
    
    # ========================================================================
    # To add new datasets or extractors:
    # 1. Anonymized: Add dataset name to dataset_names list in process_anonymized_datasets()
    # 2. CimaIO: Call process_cimiao_dataset() with new region and/or extractor
    # 3. Extractor options: RetweetExtractor(), HashtagExtractor(), SameContentExtractor()
    # ========================================================================"