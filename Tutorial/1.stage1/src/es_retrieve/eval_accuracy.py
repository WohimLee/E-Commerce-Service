# test_accuracy.py
from __future__ import annotations

import os
import argparse
import json
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from search import search
from eval_common import iter_queries_from_jsonl, AccStats


def eval_accuracy_one_file(
    path: str,
    id_key: str,
    qs_key: str,
    k_list: List[int],
    filters_json: str = "{}",
    quiet: bool = False,
    progress_every: int = 200,
    progress_secs: float = 10.0,
) -> Dict[int, AccStats]:
    filters: Dict[str, Any] = json.loads(filters_json) if filters_json else {}
    stats_map: Dict[int, AccStats] = {k: AccStats() for k in k_list}

    max_k = max(k_list)
    start_t = time.perf_counter()
    last_print_t = start_t

    for q, target_id in iter_queries_from_jsonl(path, id_key, qs_key):
        hits = search(q, filters=filters, topn=max_k)

        ids: List[str] = []
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
                stats_map[k].add_one(True, 1.0 / rank)
            else:
                stats_map[k].add_one(False, 0.0)

        if not quiet:
            n = stats_map[max_k].n
            now = time.perf_counter()
            need_print = (n % progress_every == 0) or ((now - last_print_t) >= progress_secs)
            if need_print:
                elapsed = now - start_t
                qps = n / elapsed if elapsed > 0 else 0.0
                # 打印 max_k 的当前精度快照
                snap = stats_map[max_k].summarize()
                print(
                    f"[ACC {os.path.basename(path)}] n={n} elapsed={elapsed:.1f}s "
                    f"qps={qps:.2f} Hit@{max_k}={snap['hit']:.3f} MRR@{max_k}={snap['mrr']:.3f}",
                    flush=True,
                )
                last_print_t = now

    return stats_map


def print_report(title: str, stats_map: Dict[int, AccStats]):
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    for k in sorted(stats_map.keys()):
        s = stats_map[k].summarize()
        print(f"K={k:>3} | n={s['n']:>6} | Hit@K={s['hit']:.3f} | MRR@K={s['mrr']:.3f}")
    print("=" * 80)


def merge_stats(a: AccStats, b: AccStats) -> AccStats:
    out = AccStats()
    out.n = a.n + b.n
    out.hit = a.hit + b.hit
    out.mrr_sum = a.mrr_sum + b.mrr_sum
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=os.path.join(PROJECT_ROOT, "output", "generated_questions.jsonl"))
    ap.add_argument("--queries1", default=os.path.join(PROJECT_ROOT, "output", "queries1.jsonl"))
    ap.add_argument("--queries2", default=os.path.join(PROJECT_ROOT, "output", "queries2.jsonl"))

    ap.add_argument("--k", default="1,5,10")
    ap.add_argument("--filters", default="{}")
    ap.add_argument("--quiet", action="store_true")

    ap.add_argument("--progress_every", type=int, default=200)
    ap.add_argument("--progress_secs", type=float, default=10.0)

    # 你现在只关心 generated，就把 queries1/2 默认关掉也可以；这里留参数方便你随时打开
    ap.add_argument("--enable_queries12", action="store_true", help="also eval queries1/queries2")

    args = ap.parse_args()
    k_list = [int(x) for x in args.k.split(",") if x.strip()]

    all_stats: Dict[int, AccStats] = {k: AccStats() for k in k_list}

    if args.generated:
        stats = eval_accuracy_one_file(
            args.generated, "skuid", "questions", k_list,
            filters_json=args.filters,
            quiet=args.quiet,
            progress_every=args.progress_every,
            progress_secs=args.progress_secs,
        )
        print_report(f"FILE: {args.generated} (id_key=skuid qs_key=questions)", stats)
        for k in k_list:
            all_stats[k] = merge_stats(all_stats[k], stats[k])

    if args.enable_queries12 and args.queries1:
        stats = eval_accuracy_one_file(
            args.queries1, "product_id", "queries", k_list,
            filters_json=args.filters,
            quiet=args.quiet,
            progress_every=args.progress_every,
            progress_secs=args.progress_secs,
        )
        print_report(f"FILE: {args.queries1} (id_key=product_id qs_key=queries)", stats)
        for k in k_list:
            all_stats[k] = merge_stats(all_stats[k], stats[k])

    if args.enable_queries12 and args.queries2:
        stats = eval_accuracy_one_file(
            args.queries2, "product_id", "queries", k_list,
            filters_json=args.filters,
            quiet=args.quiet,
            progress_every=args.progress_every,
            progress_secs=args.progress_secs,
        )
        print_report(f"FILE: {args.queries2} (id_key=product_id qs_key=queries)", stats)
        for k in k_list:
            all_stats[k] = merge_stats(all_stats[k], stats[k])

    print_report("OVERALL (merged)", all_stats)


if __name__ == "__main__":
    main()
