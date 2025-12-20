# test_latency.py
from __future__ import annotations

import os
import argparse
import json
import time
from typing import Any, Dict, Iterable, List, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from search import search
from eval_common import iter_queries_from_jsonl, LatStats


def iter_only_queries(path: str, id_key: str, qs_key: str) -> Iterable[str]:
    for q, _ in iter_queries_from_jsonl(path, id_key, qs_key):
        yield q


def measure_latency(
    queries: List[str],
    filters: Dict[str, Any],
    topn: int,
    warmup: int,
    repeat: int,
    quiet: bool,
    progress_every: int,
    progress_secs: float,
) -> LatStats:
    lat = LatStats()

    # warmup
    for i in range(min(warmup, len(queries))):
        _ = search(queries[i], filters=filters, topn=topn)

    start_t = time.perf_counter()
    last_print_t = start_t
    n_total = 0

    for r in range(repeat):
        for q in queries:
            t0 = time.perf_counter()
            _ = search(q, filters=filters, topn=topn)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            lat.add(dt_ms)

            n_total += 1
            if not quiet:
                now = time.perf_counter()
                need_print = (n_total % progress_every == 0) or ((now - last_print_t) >= progress_secs)
                if need_print:
                    elapsed = now - start_t
                    qps = n_total / elapsed if elapsed > 0 else 0.0
                    snap = lat.summarize()
                    print(
                        f"[LAT] n={n_total} elapsed={elapsed:.1f}s qps={qps:.2f} "
                        f"p50={snap['p50']:.1f}ms p90={snap['p90']:.1f}ms p95={snap['p95']:.1f}ms mean={snap['mean']:.1f}ms",
                        flush=True,
                    )
                    last_print_t = now

    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=os.path.join(PROJECT_ROOT, "output", "generated_questions.jsonl"))
    ap.add_argument("--id_key", default="skuid")
    ap.add_argument("--qs_key", default="questions")

    ap.add_argument("--filters", default="{}")
    ap.add_argument("--topn", type=int, default=10)

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--dedup", action="store_true", help="deduplicate queries before timing")

    ap.add_argument("--limit", type=int, default=0, help="0 means no limit; otherwise use first N queries")

    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--progress_secs", type=float, default=5.0)

    args = ap.parse_args()
    filters: Dict[str, Any] = json.loads(args.filters) if args.filters else {}

    queries = list(iter_only_queries(args.generated, args.id_key, args.qs_key))
    if args.dedup:
        # 保持相对顺序去重
        seen = set()
        deduped = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                deduped.append(q)
        queries = deduped

    if args.limit and args.limit > 0:
        queries = queries[: args.limit]

    if not queries:
        print("No queries found.")
        return

    lat = measure_latency(
        queries=queries,
        filters=filters,
        topn=args.topn,
        warmup=args.warmup,
        repeat=args.repeat,
        quiet=args.quiet,
        progress_every=args.progress_every,
        progress_secs=args.progress_secs,
    )

    s = lat.summarize()
    print("\n" + "=" * 80)
    print(f"LATENCY REPORT: {os.path.basename(args.generated)}")
    print("-" * 80)
    print(
        f"n={s['n']} | p50={s['p50']:.1f}ms | p90={s['p90']:.1f}ms | "
        f"p95={s['p95']:.1f}ms | mean={s['mean']:.1f}ms"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
