# test.py
from __future__ import annotations

import os
import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from search import search



# ----------------------------
# Metrics helpers
# ----------------------------

def percentile(sorted_values: List[float], p: float) -> float:
    """
    p in [0,100]. Uses linear interpolation between closest ranks.
    """
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]

    n = len(sorted_values)
    # rank in [0, n-1]
    r = (p / 100.0) * (n - 1)
    lo = int(r)
    hi = min(lo + 1, n - 1)
    frac = r - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass
class EvalStats:
    n: int = 0
    hit: int = 0
    mrr: float = 0.0
    lat_ms: List[float] = None

    def __post_init__(self):
        if self.lat_ms is None:
            self.lat_ms = []

    def add_one(self, is_hit: bool, rr: float, lat_ms: float):
        self.n += 1
        self.hit += 1 if is_hit else 0
        self.mrr += rr
        self.lat_ms.append(lat_ms)

    def summarize(self) -> Dict[str, Any]:
        if self.n == 0:
            return {
                "n": 0,
                "hit": 0,
                "mrr": 0.0,
                "lat_p50_ms": 0.0,
                "lat_p90_ms": 0.0,
                "lat_p95_ms": 0.0,
                "lat_mean_ms": 0.0,
            }
        l = sorted(self.lat_ms)
        lat_mean = sum(l) / len(l)
        return {
            "n": self.n,
            "hit": self.hit / self.n,
            "mrr": self.mrr / self.n,
            "lat_p50_ms": percentile(l, 50),
            "lat_p90_ms": percentile(l, 90),
            "lat_p95_ms": percentile(l, 95),
            "lat_mean_ms": lat_mean,
        }


# ----------------------------
# Data reading
# ----------------------------

def iter_queries_from_jsonl(path: str, id_key: str, qs_key: str) -> Iterable[Tuple[str, str]]:
    """
    Yields (query, target_id_as_str)
    Supports:
      - generated_questions.jsonl: {"skuid": 115024, "questions": [...]}
      - queries1/2.jsonl: {"product_id": "...", "queries": [...]}
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


# ----------------------------
# Core evaluation
# ----------------------------
def eval_one_file(
    path: str,
    id_key: str,
    qs_key: str,
    k_list: List[int],
    filters_json: str = "{}",
    quiet: bool = False,
    progress_every: int = 50,     # ✅ 新增：每多少条打印一次
    progress_secs: float = 5.0,   # ✅ 新增：至少每多少秒打印一次
) -> Dict[int, EvalStats]:
    """
    Returns stats per K.
    """
    filters: Dict[str, Any] = json.loads(filters_json) if filters_json else {}
    stats_map: Dict[int, EvalStats] = {k: EvalStats() for k in k_list}

    max_k = max(k_list)
    start_t = time.perf_counter()
    last_print_t = start_t

    for q, target_id in iter_queries_from_jsonl(path, id_key, qs_key):
        t0 = time.perf_counter()
        hits = search(q, filters=filters, topn=max_k)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        ids = []
        for h in hits or []:
            if isinstance(h, dict) and "_id" in h:
                ids.append(str(h["_id"]))
            else:
                ids.append(str(h))

        try:
            rank = ids.index(target_id) + 1
        except ValueError:
            rank = None

        for k in k_list:
            if rank is not None and rank <= k:
                stats_map[k].add_one(True, 1.0 / rank, dt_ms)
            else:
                stats_map[k].add_one(False, 0.0, dt_ms)

        # ✅ 进度打印：按条数 or 按时间
        if not quiet:
            n = stats_map[max_k].n
            now = time.perf_counter()
            need_print = (n % progress_every == 0) or ((now - last_print_t) >= progress_secs)
            if need_print:
                elapsed = now - start_t
                qps = n / elapsed if elapsed > 0 else 0.0

                # 简单用当前平均速度估 ETA（需要知道总数才精确；这里先不给总数也能看速度）
                mean_lat = (sum(stats_map[max_k].lat_ms) / len(stats_map[max_k].lat_ms)) if stats_map[max_k].lat_ms else 0.0

                print(
                    f"[{os.path.basename(path)}] n={n} elapsed={elapsed:.1f}s "
                    f"qps={qps:.2f} mean_lat={mean_lat:.1f}ms",
                    flush=True,  # ✅ 强制立刻显示
                )
                last_print_t = now

    return stats_map



def merge_stats(a: EvalStats, b: EvalStats) -> EvalStats:
    out = EvalStats()
    out.n = a.n + b.n
    out.hit = a.hit + b.hit
    out.mrr = a.mrr + b.mrr
    out.lat_ms = (a.lat_ms or []) + (b.lat_ms or [])
    return out


def print_report(title: str, stats_map: Dict[int, EvalStats]):
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    for k in sorted(stats_map.keys()):
        s = stats_map[k].summarize()
        print(
            f"K={k:>3} | n={s['n']:>6} | Hit@K={s['hit']:.3f} | MRR@K={s['mrr']:.3f} "
            f"| lat(ms) p50={s['lat_p50_ms']:.1f} p90={s['lat_p90_ms']:.1f} p95={s['lat_p95_ms']:.1f} mean={s['lat_mean_ms']:.1f}"
        )
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=os.path.join(PROJECT_ROOT, "output", "generated_questions.jsonl"), help="path to generated_questions.jsonl")
    ap.add_argument("--queries1", default=os.path.join(PROJECT_ROOT, "output", "queries1.jsonl"), help="path to queries1.jsonl")
    ap.add_argument("--queries2", default=os.path.join(PROJECT_ROOT, "output", "queries2.jsonl"), help="path to queries2.jsonl")

    ap.add_argument("--k", default="1,5,10", help="comma-separated K list, e.g. 1,5,10")
    ap.add_argument("--filters", default="{}", help='JSON string filters passed to search(), e.g. \'{"brand":"xxx"}\'')
    ap.add_argument("--quiet", action="store_true", help="no progress printing")

    ap.add_argument("--progress_every", type=int, default=50, help="print progress every N queries")
    ap.add_argument("--progress_secs", type=float, default=5.0, help="print progress at least every N seconds")


    args = ap.parse_args()
    k_list = [int(x) for x in args.k.split(",") if x.strip()]

    # per-file eval
    all_stats_by_k: Dict[int, EvalStats] = {k: EvalStats() for k in k_list}

    if args.generated:
        stats = eval_one_file(
            args.generated, "skuid", "questions", k_list, 
            filters_json=args.filters, 
            quiet=args.quiet,
            progress_every=args.progress_every,
            progress_secs=args.progress_secs,
        )
        print_report(f"FILE: {args.generated} (id_key=skuid qs_key=questions)", stats)
        for k in k_list:
            all_stats_by_k[k] = merge_stats(all_stats_by_k[k], stats[k])

    # if args.queries1:
    #     stats = eval_one_file(
    #         args.queries1, "product_id", "queries", k_list, 
    #         filters_json=args.filters, 
    #         quiet=args.quiet,
    #         progress_every=args.progress_every,
    #         progress_secs=args.progress_secs,
    #     )
    #     print_report(f"FILE: {args.queries1} (id_key=product_id qs_key=queries)", stats)
    #     for k in k_list:
    #         all_stats_by_k[k] = merge_stats(all_stats_by_k[k], stats[k])

    # if args.queries2:
    #     stats = eval_one_file(
    #         args.queries2, "product_id", "queries", k_list, 
    #         filters_json=args.filters, 
    #         quiet=args.quiet,
    #         progress_every=args.progress_every,
    #         progress_secs=args.progress_secs,
    #     )
    #     print_report(f"FILE: {args.queries2} (id_key=product_id qs_key=queries)", stats)
    #     for k in k_list:
    #         all_stats_by_k[k] = merge_stats(all_stats_by_k[k], stats[k])

    # overall
    print_report("OVERALL (merged)", all_stats_by_k)


if __name__ == "__main__":
    main()
