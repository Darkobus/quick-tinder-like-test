"""Pair people by how well each one matches what the other is looking for.

Each person has three attributes (age range, work type, department), the values they
want in a partner, and a preference order such as ``work type > age range > department``.

Compatibility of A and B is mutual: A's wishes are checked against B's attributes and
B's wishes against A's, each attribute weighted by its rank in that person's preference
order (3 / 2 / 1). The maximum is 12.
"""
from __future__ import annotations

import pandas as pd

ATTRIBUTES = ["age range", "work type", "department"]
WEIGHTS = (3, 2, 1)
MAX_SCORE = 2 * sum(WEIGHTS)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and values; accept the 'departament' spelling."""
    df = df.copy()
    df.columns = [c.strip().lower().replace("departament", "department") for c in df.columns]
    missing = {"name", "preference", *ATTRIBUTES, *(f"desired {a}" for a in ATTRIBUTES)} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    for col in df.columns:
        if col != "name" and df[col].dtype == object:
            df[col] = (df[col].fillna("").astype(str).str.strip().str.lower()
                       .str.replace("departament", "department"))
    return df.reset_index(drop=True)


def _weights(preference: str) -> dict[str, int]:
    order = [p.strip() for p in preference.split(">") if p.strip() in ATTRIBUTES]
    order += [a for a in ATTRIBUTES if a not in order]  # missing ranks go last
    return dict(zip(order, WEIGHTS))


def one_way(seeker: pd.Series, other: pd.Series) -> int:
    w = _weights(seeker["preference"])
    return sum(w[a] for a in ATTRIBUTES if seeker[f"desired {a}"] == other[a])


def compatibility(a: pd.Series, b: pd.Series) -> int:
    return one_way(a, b) + one_way(b, a)


def score_matrix(df: pd.DataFrame) -> list[list[int]]:
    rows = [df.iloc[i] for i in range(len(df))]
    return [[compatibility(p, q) if i != j else -1 for j, q in enumerate(rows)]
            for i, p in enumerate(rows)]


# --- algorithms -------------------------------------------------------------------
# Each takes a normalized DataFrame and returns a list of (index, index) pairs.


def greedy(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Walk through people in order, give each the best partner still free."""
    s = score_matrix(df)
    free, pairs = set(range(len(df))), []
    for i in range(len(df)):
        if i not in free:
            continue
        free.discard(i)
        if not free:
            break
        j = max(free, key=lambda k: s[i][k])
        free.discard(j)
        pairs.append((i, j))
    return pairs


def max_weight(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Maximum-weight matching on the full graph: best total compatibility."""
    import networkx as nx

    s = score_matrix(df)
    g = nx.Graph()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            g.add_edge(i, j, weight=s[i][j])
    return sorted(tuple(sorted(p)) for p in nx.max_weight_matching(g, maxcardinality=True))


def hungarian(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Assignment (each person -> one partner) via the Hungarian algorithm.

    Assignment is bipartite, so i -> j does not force j -> i; pairs are read off in
    order and conflicts dropped. An approximation here, kept for comparison.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    s = np.array(score_matrix(df), dtype=float)
    np.fill_diagonal(s, -1e6)
    rows, cols = linear_sum_assignment(-s)
    used, pairs = set(), []
    for i, j in zip(rows, cols):
        if i != j and i not in used and j not in used:
            pairs.append((int(i), int(j)))
            used |= {i, j}
    return pairs


def stable_split(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Gale–Shapley after splitting the list into two halves (proposers / receivers)."""
    s = score_matrix(df)
    n = len(df) // 2
    proposers, receivers = list(range(n)), list(range(n, 2 * n))
    prefs = {p: sorted(receivers, key=lambda r: -s[p][r]) for p in proposers}
    rank = {r: {p: -s[r][p] for p in proposers} for r in receivers}
    engaged: dict[int, int] = {}
    nxt = {p: 0 for p in proposers}
    free = list(proposers)
    while free:
        p = free.pop(0)
        r = prefs[p][nxt[p]]
        nxt[p] += 1
        if r not in engaged:
            engaged[r] = p
        elif rank[r][p] < rank[r][engaged[r]]:
            free.append(engaged[r])
            engaged[r] = p
        else:
            free.append(p)
    return sorted((p, r) for r, p in engaged.items())


def kmeans_then_greedy(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Cluster people by what they want (one-hot), then pair greedily inside clusters."""
    from sklearn.cluster import KMeans

    x = pd.get_dummies(df[[f"desired {a}" for a in ATTRIBUTES]]).to_numpy(dtype=float)
    k = max(1, len(df) // 4)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)
    s = score_matrix(df)
    order = sorted(range(len(df)), key=lambda i: labels[i])
    free, pairs = set(order), []
    for i in order:
        if i not in free:
            continue
        free.discard(i)
        if not free:
            break
        same = [j for j in free if labels[j] == labels[i]] or list(free)
        j = max(same, key=lambda m: s[i][m])
        free.discard(j)
        pairs.append((i, j))
    return pairs


ALGORITHMS = {
    "Maximum Weight Matching": (max_weight, "Best total compatibility over all pairs (graph matching). Recommended."),
    "Basic Greedy": (greedy, "Each person, in list order, takes the best partner still free. Fast, not optimal."),
    "Stable Matching (two halves)": (stable_split, "Gale–Shapley: no two people would both rather be with each other. Splits the list into two groups."),
    "Hungarian Algorithm": (hungarian, "Optimal one-way assignment, turned into pairs. An approximation for this problem."),
    "K-Means + Greedy": (kmeans_then_greedy, "Groups people with similar wishes, then pairs inside each group."),
}


def run(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    df = normalize(df)
    func, _ = ALGORITHMS[algorithm]
    s = score_matrix(df)
    rows = [
        {"Person A": df.at[i, "name"], "Person B": df.at[j, "name"],
         "Compatibility %": round(100 * s[i][j] / MAX_SCORE)}
        for i, j in func(df)
    ]
    return pd.DataFrame(rows, columns=["Person A", "Person B", "Compatibility %"])
