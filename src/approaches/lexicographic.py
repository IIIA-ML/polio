"""Lexicographic approach: users ranked by a first approach, ties broken by the next ones."""

from .base import Approach, get_ranks


class LexicographicApproach(Approach):
    """
    Ranking by the tie groups of the first sub-approach, each group split by the ranks of the next sub-approach,
    and so on (the dense rank of the tuple of sub-approach ranks).

    Its own min_coactions is that of the first sub-approach and its ranking_mode the default; the sub-approach
    rankings are stored (and reused) next to the lexicographic one.
    """

    def __init__(self, approaches, window_sec=60):
        super().__init__(window_sec, approaches[0].min_coactions)
        self.approaches = approaches
        self.approach_keys = [a.get_approach_key() for a in approaches]
        self.sub_modes = [a.ranking_mode for a in approaches]
        self.sub_min_coactions = [a.min_coactions for a in approaches]

    def get_approach_name(self):
        parts = []
        for approach in self.approaches:
            name = approach.get_approach_name()
            if approach.min_coactions > 1:
                name = f"{name} min{approach.min_coactions}"
            if not approach.need_filtering and approach.get_approach_key() != 'coretweets':
                name = f"{name} NoFilter"
            parts.append(f"{name} [{approach.ranking_mode}]")
        return f"Lexicographic: {'+'.join(parts)}"

    def get_approach_key(self):
        return f"lexicographic_{'_'.join(a.get_full_approach_key() for a in self.approaches)}"

    def get_full_approach_key(self):
        return self.get_approach_key()        # the key already holds every sub-approach setting

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['sub_approaches'] = self.approach_keys
        metadata['sub_modes'] = self.sub_modes
        metadata['sub_min_coactions'] = self.sub_min_coactions
        metadata['sub_approach_modes'] = dict(zip(self.approach_keys, self.sub_modes))
        metadata['sub_approach_min_coactions'] = dict(zip(self.approach_keys, self.sub_min_coactions))
        return metadata

    def _compute_suspicious(self, RTs, **storage):
        groups = self.approaches[0].get_suspicious(RTs, **storage)
        if not groups:
            return []
        for approach in self.approaches[1:]:
            suspicious = approach.get_suspicious(RTs, **storage)
            if suspicious:
                groups = _refine_groups(groups, get_ranks(suspicious))
        return groups


def _refine_groups(groups, ranks):
    """Split every group by the ranks of its users (lowest first); users without a rank go last, together."""
    refined = []
    for group in groups:
        if len(group) <= 1:
            refined.append(group)
            continue
        by_rank, unranked = {}, []
        for user in group:
            if user in ranks:
                by_rank.setdefault(ranks[user], []).append(user)
            else:
                unranked.append(user)
        refined += [sorted(by_rank[rank]) for rank in sorted(by_rank)]
        if unranked:
            refined.append(sorted(unranked))
    return refined
