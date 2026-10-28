import csv

import numpy as np


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    """Returns a list of dicts with numeric fields converted to float."""
    def conv(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(path) as f:
        return [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]


def select(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def col(rows, key):
    return np.array([r[key] for r in rows])
