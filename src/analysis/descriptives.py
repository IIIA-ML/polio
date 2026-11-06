"""
Descriptive statistics of the co-retweet graph: pairs formed by a single co-retweet.
"""

from collections import Counter

import numpy as np
import pandas as pd

from approaches.coretweets import CoRetweetsApproach
from data_loader import import_data


def coretweet_pair_statistics(cases):
    """
    Statistics of the co-retweet counts c >= 1 of the pairs (min_coactions = 1, so every pair is included), one
    row per distinct (data_dir, dataset, window_sec) of `cases`, in order:
        pairs_{g}        number of nonzero pairs, for g = all, io (>= 1 IO endpoint)
        c1_share_{g}     share of them with c = 1, for g = all, io, control (no IO endpoint)
        ties_{k}_median  median number of nonzero pairs per user, for k = io, control
        single_{k}_share share of the users whose pairs all have c = 1 (tied at the bottom by the count), for
                         k = io, control
        io_total         number of IO accounts of the dataset
    """
    rows, seen = [], set()
    for data_dir, dataset, window in cases:
        if (data_dir, dataset, window) in seen:
            continue
        seen.add((data_dir, dataset, window))
        print(f"Ties: {dataset} (window {window} s)")
        RTs, io_users = import_data(f"{data_dir}/{dataset}/Processed/")
        io = set(io_users)
        pairs = CoRetweetsApproach(window, 1).compute_pairs_scores(RTs)
        pairs = {pair: c for pair, c in pairs.items() if c != 0}

        c = np.fromiter(pairs.values(), dtype=float, count=len(pairs))
        n_io = np.fromiter(((a in io) + (b in io) for a, b in pairs), dtype=int, count=len(pairs))
        row = {'dataset': dataset, 'window_sec': window}
        for group, mask in (('all', np.ones(len(c), dtype=bool)), ('io', n_io >= 1), ('control', n_io == 0)):
            row[f'pairs_{group}'] = int(mask.sum())
            row[f'c1_share_{group}'] = float((c[mask] == 1).mean()) if mask.any() else np.nan

        ties = Counter(u for pair in pairs for u in pair)
        max_count = {}
        for pair, count in pairs.items():
            for u in pair:
                if count > max_count.get(u, 0):
                    max_count[u] = count
        for kind, users in (('io', [u for u in ties if u in io]), ('control', [u for u in ties if u not in io])):
            row[f'ties_{kind}_median'] = np.median([ties[u] for u in users]) if users else np.nan
            row[f'single_{kind}_share'] = float(np.mean([max_count[u] == 1 for u in users])) if users else np.nan
        row['io_total'] = len(io)
        rows.append(row)
    return pd.DataFrame(rows)
