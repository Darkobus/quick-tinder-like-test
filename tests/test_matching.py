from pathlib import Path

import pandas as pd
import pytest

import matching as m

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _person(name, attrs, wants, pref="work type > age range > department"):
    return {"name": name, "preference": pref,
            "age range": attrs[0], "work type": attrs[1], "department": attrs[2],
            "desired age range": wants[0], "desired work type": wants[1], "desired department": wants[2]}


def test_compatibility_is_mutual_and_weighted():
    a = pd.Series(_person("A", ("gen z", "intern", "it"), ("boomer", "senior", "hr")))
    b = pd.Series(_person("B", ("boomer", "senior", "lab"), ("gen z", "x", "x")))
    # A wants: age (2) + work type (3) match B, department (1) does not -> 5
    # B wants: only age (2) matches A -> 2
    assert m.one_way(a, b) == 5
    assert m.one_way(b, a) == 2
    assert m.compatibility(a, b) == m.compatibility(b, a) == 7


def test_normalize_accepts_original_spelling():
    df = pd.read_excel(EXAMPLES / "people_10.xlsx")  # uses "departament"
    assert "department" in m.normalize(df).columns


def test_missing_column_is_reported():
    with pytest.raises(ValueError, match="missing columns"):
        m.normalize(pd.DataFrame({"Name": ["x"]}))


@pytest.mark.parametrize("algorithm", list(m.ALGORITHMS))
def test_every_algorithm_returns_disjoint_pairs(algorithm):
    df = m.normalize(pd.read_excel(EXAMPLES / "people_26.xlsx"))
    pairs = m.ALGORITHMS[algorithm][0](df)
    people = [p for pair in pairs for p in pair]
    assert len(people) == len(set(people))
    assert len(pairs) >= len(df) // 2 - 1


def test_max_weight_is_never_worse_than_greedy():
    df = m.normalize(pd.read_excel(EXAMPLES / "people_26.xlsx"))
    s = m.score_matrix(df)
    total = lambda pairs: sum(s[i][j] for i, j in pairs)  # noqa: E731
    assert total(m.max_weight(df)) >= total(m.greedy(df))
