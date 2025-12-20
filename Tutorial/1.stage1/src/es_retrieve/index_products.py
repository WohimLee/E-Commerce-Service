# index_products.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Iterable, Tuple, Optional

from es_client import get_es, create_index_if_not_exists, bulk_index
from schemas import PRODUCT_MAPPING
from embedder import get_embedder


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at {path}:{lineno}: {e}") from e


def batched(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description="Create ES index and bulk index merged product docs with embeddings.")
    parser.add_argument("--es", default=os.getenv("ES_URL", "http://localhost:9200"), help="Elasticsearch URL")
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "products_sku_v1"), help="Index name")
    parser.add_argument(
        "--input",
        default="output/merged_products.jsonl",
        help="Input merged jsonl (from build_dataset.py)",
    )
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "500")), help="Embedding/bulk batch size")
    parser.add_argument("--refresh", default="false", choices=["false", "true", "wait_for"], help="ES refresh policy after bulk")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate index (DANGEROUS)")
    parser.add_argument("--dims", type=int, default=int(os.getenv("EMBEDDER_DIMS", "1024")), help="Embedding dims (must match mapping)")
    parser.add_argument("--id_field", default="skuid", help="Doc id field (default skuid)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    es = get_es(es_url=args.es)
    embedder = get_embedder()

    # Optional recreate
    if args.recreate:
        from es_client import delete_index
        delete_index(es, args.index, ignore_missing=True)

    # Create index if needed
    # Ensure mapping dims match (basic safety)
    mapping = PRODUCT_MAPPING
    try:
        emb_field = mapping["mappings"]["properties"]["emb_text"]
        emb_dims = int(emb_field.get("dims", args.dims))
        if emb_dims != args.dims:
            print(f"[WARN] Mapping emb_text.dims={emb_dims} but --dims={args.dims}. "
                  f"Indexing may fail unless they match. Using mapping dims.")
    except Exception:
        pass

    create_index_if_not_exists(es, args.index, mapping=mapping)

    # Refresh policy
    refresh: Any
    if args.refresh == "false":
        refresh = False
    elif args.refresh == "true":
        refresh = True
    else:
        refresh = "wait_for"

    total_docs = 0
    total_success = 0
    total_fail = 0
    first_failure: Optional[Dict[str, Any]] = None

    # Stream, batch, embed, bulk index
    for batch_no, docs in enumerate(batched(read_jsonl(args.input), args.batch_size), start=1):
        total_docs += len(docs)

        # 1) collect search_text
        texts = []
        for d in docs:
            st = d.get("search_text")
            if not st or not str(st).strip():
                # fallback: at least give product_name
                st = str(d.get("product_name", "") or "")
            texts.append(str(st))

        # 2) embed
        vectors = embedder.embed(texts)
        if len(vectors) != len(docs):
            raise RuntimeError(f"Embedding count mismatch: got {len(vectors)} expected {len(docs)}")

        # 3) attach emb_text
        for d, v in zip(docs, vectors):
            d["emb_text"] = v

        # 4) bulk index (each doc _id = skuid via es_client.bulk_index id_field)
        success, fail, failures = bulk_index(
            es,
            index_name=args.index,
            docs=docs,
            id_field=args.id_field,
            chunk_size=min(args.batch_size, 1000),
            refresh=False,  # do refresh once at end unless user wants per-batch refresh
            raise_on_error=False,
        )
        total_success += success
        total_fail += fail
        if failures and first_failure is None:
            first_failure = failures[0]

        print(f"[batch {batch_no}] docs={len(docs)} success={success} fail={fail} total={total_docs}")

    # Final refresh (optional)
    if refresh:
        es.indices.refresh(index=args.index)
        if refresh == "wait_for":
            # "wait_for" is mostly meaningful per request; here we just refresh.
            pass

    print("\nDone indexing.")
    print(f"  index:         {args.index}")
    print(f"  input:         {args.input}")
    print(f"  docs seen:     {total_docs}")
    print(f"  bulk success:  {total_success}")
    print(f"  bulk failed:   {total_fail}")
    if first_failure:
        print("  first failure (sample):")
        print(json.dumps(first_failure, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
