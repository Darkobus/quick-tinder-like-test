# Pair matching tool

Pairs people up by how well each one fits what the other is looking for — e.g. for a
mentoring programme or job-shadowing pairs — and compares five matching algorithms on the
same data. Desktop app (CustomTkinter) plus a plain Python module.

## The model

Every person has three attributes (**age range**, **work type**, **department**), the
values they want in a partner, and a preference order such as
`work type > age range > department`.

Compatibility of A and B is **mutual**: A's wishes are checked against B's attributes and
B's wishes against A's, each attribute weighted 3 / 2 / 1 by its rank in that person's
preference order (max 12 → 100%).

## Algorithms

On `examples/people_26.xlsx` (13 pairs), total compatibility score:

| Algorithm | Total | Note |
|---|---|---|
| Maximum weight matching (NetworkX) | **99** | optimal for this problem |
| Hungarian algorithm (SciPy) | 99 | one-way assignment turned into pairs; matched the optimum here, not guaranteed |
| Greedy | 97 | each person in list order takes the best free partner |
| Stable matching (Gale–Shapley on two halves) | 90 | no two people would both rather swap; optimises stability, not total |
| K-means + greedy | 61 | clustering first throws away cross-cluster pairs |

## Usage

```bash
pip install -r requirements.txt
python app.py
```

Pick an `.xlsx` with columns `Name, Age Range, Work Type, Department, Preference,
Desired Age Range, Desired Work Type, Desired Department` (see `examples/`), an output
folder and an algorithm. Results go to `matched_pairs.xlsx`.

From code:

```python
import pandas as pd, matching
pairs = matching.run(pd.read_excel("examples/people_26.xlsx"), "Maximum Weight Matching")
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
