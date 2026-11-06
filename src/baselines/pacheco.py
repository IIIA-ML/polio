"""
Co-retweet TF-IDF cosine similarity of Pacheco et al. (2021), "Uncovering Coordinated Networks on
Social Media: Methods and Case Studies", ICWSM, Case Study 4 (co-retweets).

The implementation follows the released code (github.com/osome-iu/coordination-detection,
CS4/co_retweet.ipynb), which differs from the text of the paper in two details noted below:
    1. Trace: for every account, the IDs of the tweets it retweeted, with their counts. No time
       window and no exclusion of self-retweets (neither is applied in the released code).
    2. Support filter: accounts with MORE than `support_threshold` = 10 retweets are kept (the
       code uses `> sup_thresh`; the paper says accounts with fewer than 10 are removed), and only
       accounts with more than one distinct retweeted tweet form a document (`len(grp) > 1`).
    3. Bipartite weight: scikit-learn TfidfVectorizer with sublinear_tf=True (tf = 1 + ln count),
       smooth idf (idf = ln((1 + D) / (1 + df)) + 1, D = number of documents), l2 normalisation and
       min_df = 3 (tweets retweeted by fewer than 3 kept accounts are dropped; min_df = 1 if that
       leaves no tweet).
    4. Projection weight: cosine similarity of the TF-IDF vectors. The code keeps, for every
       account, its `top_n` = 100 most similar accounts (sparse_dot_topn, the account itself
       included) and lists the pairs (u1, u2) with u1 before u2 in the document order, taken from
       the list of u1.
    5. Edge filter: edges with similarity >= the `edge_quantile` = 0.995 quantile of the listed
       similarities; the coordinated accounts are the nodes of the remaining edges.

Account score for ranking: the largest similarity of the account over its listed edges. An account
is flagged by the edge filter if and only if this score reaches the threshold, so ranking by it is
the ranking implied by the method for any edge quantile. Accounts without a listed edge get no
score.

Everything is computed with numpy and scipy (no sparse_dot_topn); top_neighbours keeps the top_n
values of every row exactly, with ties at the cut broken by column index.
"""

from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

SUPPORT_THRESHOLD = 10
MIN_DF = 3
TOP_N = 100
EDGE_QUANTILE = 0.995


def account_documents(RTs, support_threshold=SUPPORT_THRESHOLD):
    """
    Documents of the kept accounts: {account: Counter(retweeted tweet id -> count)}, for the
    accounts with more than support_threshold retweets and more than one distinct retweeted tweet.

    Args:
        RTs: iterable of (account, retweeted tweet id, timestamp)
    """
    docs = {}
    for account, tweet, _ in RTs:
        docs.setdefault(account, Counter())[tweet] += 1
    return {a: c for a, c in docs.items() if sum(c.values()) > support_threshold and len(c) > 1}


def tfidf_matrix(docs, min_df=MIN_DF):
    """
    TF-IDF matrix of the documents as TfidfVectorizer(sublinear_tf=True, smooth_idf=True,
    norm='l2', min_df=min_df) would build it.

    Returns:
        (accounts in row order, csr matrix accounts x kept tweets, rows l2-normalised)
    """
    accounts = sorted(docs)
    df = Counter(t for a in accounts for t in docs[a])
    kept = sorted(t for t, d in df.items() if d >= min_df)
    if not kept:                    # as the released code: retry with min_df = 1
        kept = sorted(df)
    col = {t: j for j, t in enumerate(kept)}
    rows, cols, vals = [], [], []
    for i, a in enumerate(accounts):
        for t, count in docs[a].items():
            j = col.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
                vals.append(1.0 + np.log(count))
    n_docs = len(accounts)
    idf = np.log((1.0 + n_docs) / (1.0 + np.array([df[t] for t in kept], dtype=float))) + 1.0
    X = csr_matrix((np.array(vals) * idf[np.array(cols, dtype=np.int64)], (rows, cols)),
                   shape=(n_docs, len(kept)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    X = csr_matrix(X.multiply(1.0 / norms[:, None]))
    return accounts, X


def top_neighbours(X, top_n=TOP_N, chunk=2048):
    """
    For every row i of X X^T, its top_n largest entries (the diagonal included, as in the
    released code), as (rows, cols, values) arrays. Ties at the cut are broken by column index.
    """
    XT = X.T.tocsc()
    out_r, out_c, out_v = [], [], []
    for start in range(0, X.shape[0], chunk):
        S = (X[start:start + chunk] @ XT).tocsr()
        for k in range(S.shape[0]):
            lo, hi = S.indptr[k], S.indptr[k + 1]
            cols, vals = S.indices[lo:hi], S.data[lo:hi]
            if len(vals) > top_n:
                order = np.lexsort((cols, -vals))[:top_n]
                cols, vals = cols[order], vals[order]
            out_r.append(np.full(len(cols), start + k, dtype=np.int64))
            out_c.append(cols.astype(np.int64))
            out_v.append(vals)
    return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_v)


def interactions(RTs, support_threshold=SUPPORT_THRESHOLD, min_df=MIN_DF, top_n=TOP_N):
    """
    Edge list of the method: (accounts, u1 index, u2 index, similarity) with u1 < u2 in row
    order, taken from the top_n list of u1.
    """
    accounts, X = tfidf_matrix(account_documents(RTs, support_threshold), min_df)
    if not accounts:
        return accounts, np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0)
    r, c, v = top_neighbours(X, top_n)
    keep = r < c
    return accounts, r[keep], c[keep], v[keep]


def account_scores(accounts, u1, u2, sim):
    """{account: largest similarity over its listed edges}, for the accounts with an edge."""
    best = np.full(len(accounts), -np.inf)
    np.maximum.at(best, u1, sim)
    np.maximum.at(best, u2, sim)
    return {accounts[i]: float(s) for i, s in enumerate(best) if np.isfinite(s)}


def flagged_accounts(accounts, u1, u2, sim, edge_quantile=EDGE_QUANTILE):
    """
    Accounts of the edges with similarity >= the edge_quantile quantile of the listed
    similarities (the coordinated accounts of the method), and the threshold.
    """
    if len(sim) == 0:
        return set(), np.nan
    threshold = float(np.quantile(sim, edge_quantile))   # pandas Series.quantile: linear, as numpy
    keep = sim >= threshold
    return {accounts[i] for i in np.concatenate([u1[keep], u2[keep]])}, threshold
