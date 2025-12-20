# eval_common.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def iter_queries_from_jsonl(path: str, id_key: str, qs_key: str) -> Iterable[Tuple[str, str]]:
    """
    Yields (query, target_id_as_str)
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                raise ValueError(f"[{path}] line {line_no}: invalid json: {e}")

            if id_key not in item:
                raise KeyError(f"[{path}] line {line_no}: missing id_key '{id_key}'")
            if qs_key not in item:
                raise KeyError(f"[{path}] line {line_no}: missing qs_key '{qs_key}'")

            target_id = str(item[id_key])
            qs = item[qs_key]
            if not isinstance(qs, list):
                raise TypeError(f"[{path}] line {line_no}: '{qs_key}' must be a list")

            for q in qs:
                if not isinstance(q, str):
                    continue
                q = q.strip()
                if not q:
                    continue
                yield (q, target_id)


@dataclass
class AccStats:
    n: int = 0
    hit: int = 0
    mrr_sum: float = 0.0

    def add_one(self, is_hit: bool, rr: float):
        self.n += 1
        if is_hit:
            self.hit += 1
        self.mrr_sum += rr

    def summarize(self) -> Dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "hit": 0.0, "mrr": 0.0}
        return {
            "n": self.n,
            "hit": self.hit / self.n,
            "mrr": self.mrr_sum / self.n,
        }


def percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    n = len(sorted_values)
    r = (p / 100.0) * (n - 1)
    lo = int(r)
    hi = min(lo + 1, n - 1)
    frac = r - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass
class LatStats:
    lat_ms: List[float]

    def __init__(self):
        self.lat_ms = []

    def add(self, ms: float):
        self.lat_ms.append(ms)

    def summarize(self) -> Dict[str, Any]:
        if not self.lat_ms:
            return {"n": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "mean": 0.0}
        l = sorted(self.lat_ms)
        mean = sum(l) / len(l)
        return {
            "n": len(l),
            "p50": percentile(l, 50),
            "p90": percentile(l, 90),
            "p95": percentile(l, 95),
            "mean": mean,
        }
