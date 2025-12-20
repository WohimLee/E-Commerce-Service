# search.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from es_client import get_es
from embedder import get_embedder


# 你给导购/RAG 用的字段（按需增减）
DEFAULT_SOURCE_FIELDS = [
    "skuid", "spuid",
    "brand_name", "group_name", "product_name",
    "price", "sales",
    "season", "scene", "material", "style",
    "people_gender", "age_range", "color", "size"
]


# ----------------------------
# Filter builder
# ----------------------------

def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def build_es_filters(filters: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert a simple filter dict into ES filter clauses for:
      - bool.filter (standard retriever)
      - knn.filter (knn retriever)

    Supported keys:
      - terms filters: season/scene/material/style/people_gender/age_range/color/size/brand_name_kw/group_name_kw/spuid/skuid
      - range filters: price (price_lte/price_gte/price_lt/price_gt), sales (sales_lte/...)

    Examples:
      filters = {
        "season": ["冬季"],
        "people_gender": ["女"],
        "color": ["黑色", "经典黑"],
        "price_lte": 1500,
        "price_gte": 300
      }
    """
    bool_filters: List[Dict[str, Any]] = []
    knn_filters: List[Dict[str, Any]] = []

    # Terms-like filters (keyword fields)
    term_fields = [
        "season", "scene", "material", "style", "people_gender", "age_range", "color", "size",
        "skuid", "spuid",
        # 下面两个如果你想精确匹配品牌/品类，可用 brand_name.kw / group_name.kw
        "brand_name.kw", "group_name.kw",
    ]

    for f in term_fields:
        key = f.replace(".kw", "_kw") if f.endswith(".kw") else f
        if key in filters and filters[key] is not None:
            vals = _as_list(filters[key])
            vals = [str(x) for x in vals if str(x).strip() != ""]
            if vals:
                clause = {"terms": {f: vals}}
                bool_filters.append(clause)
                knn_filters.append(clause)

    # Range filters helper
    def add_range(field: str):
        ops = {}
        for op in ["lte", "gte", "lt", "gt"]:
            k = f"{field}_{op}"
            if k in filters and filters[k] is not None:
                ops[op] = filters[k]
        if ops:
            clause = {"range": {field: ops}}
            bool_filters.append(clause)
            knn_filters.append(clause)

    add_range("price")
    add_range("sales")

    return bool_filters, knn_filters


# ----------------------------
# Query builder
# ----------------------------

def build_search_body(
    query: str,
    query_vector: List[float],
    filters: Dict[str, Any],
    topn: int = 50,
    *,
    bm25_window: int = 200,
    knn_k: int = 200,
    knn_num_candidates: int = 1500,
    rrf_rank_constant: int = 60,
    rrf_rank_window_size: int = 200,
    source_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bool_filters, knn_filters = build_es_filters(filters)

    # BM25 multi_match fields (加权可按效果再调)
    mm_fields = [
        "product_name^4",
        "brand_name^2",
        "group_name^2",
        "marketing_attributes^1.5",
        "product_description",
        "search_text^2",
    ]

    body: Dict[str, Any] = {
        "size": topn,
        "track_total_hits": False,
        "_source": source_fields or DEFAULT_SOURCE_FIELDS,
        "retriever": {
            "rrf": {
                "rank_constant": rrf_rank_constant,
                "rank_window_size": rrf_rank_window_size,
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "bool": {
                                    "filter": bool_filters,
                                    "must": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "fields": mm_fields,
                                                "type": "best_fields"
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": "emb_text",
                            "query_vector": query_vector,
                            "k": knn_k,
                            "num_candidates": knn_num_candidates,
                            "filter": knn_filters
                        }
                    }
                ]
            }
        }
    }

    return body


# ----------------------------
# Public API: search()
# ----------------------------

def search(
    query: str,
    filters: Dict[str, Any],
    topn: int = 50,
    *,
    es_url: Optional[str] = None,
    index_name: Optional[str] = None,
    source_fields: Optional[List[str]] = None,
    knn_k: int = 200,
    knn_num_candidates: int = 1500,
    rrf_rank_window_size: int = 200,
) -> List[Dict[str, Any]]:
    """
    Hybrid search with ES retriever.rrf:
      - standard (BM25 multi_match)
      - knn (dense_vector cosine) with same filters
    Returns a list of hits (dict), each includes _source and _score.

    filters: see build_es_filters()
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    es_url = es_url or os.getenv("ES_URL", "http://localhost:9200")
    index_name = index_name or os.getenv("ES_INDEX", "products_sku_v1")

    es = get_es(es_url=es_url)
    embedder = get_embedder()

    query_vector = embedder.embed([query])[0]

    body = build_search_body(
        query=query,
        query_vector=query_vector,
        filters=filters,
        topn=topn,
        source_fields=source_fields,
        knn_k=knn_k,
        knn_num_candidates=knn_num_candidates,
        rrf_rank_window_size=rrf_rank_window_size,
    )

    resp = es.search(index=index_name, body=body)
    hits = resp.get("hits", {}).get("hits", []) or []

    # Normalize output a bit
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append({
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "_source": h.get("_source", {}),
        })
    return out


# ----------------------------
# CLI
# ----------------------------

def _parse_cli_filters(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Simple CLI -> filters dict.
    You can expand this as your query parser evolves.
    """
    f: Dict[str, Any] = {}

    # multi-value terms
    if args.season: f["season"] = args.season
    if args.scene: f["scene"] = args.scene
    if args.material: f["material"] = args.material
    if args.style: f["style"] = args.style
    if args.gender: f["people_gender"] = args.gender
    if args.age: f["age_range"] = args.age
    if args.color: f["color"] = args.color
    if args.size: f["size"] = args.size

    # exact brand/group via keyword subfields (in build_es_filters -> brand_name.kw / group_name.kw)
    if args.brand_kw: f["brand_name_kw"] = args.brand_kw
    if args.group_kw: f["group_name_kw"] = args.group_kw

    # range
    if args.price_lte is not None: f["price_lte"] = args.price_lte
    if args.price_gte is not None: f["price_gte"] = args.price_gte
    if args.sales_gte is not None: f["sales_gte"] = args.sales_gte

    return f


def main():
    parser = argparse.ArgumentParser(description="Hybrid search (BM25 + vector) using ES retriever.rrf")
    parser.add_argument("--es", default=os.getenv("ES_URL", "http://localhost:9200"), help="Elasticsearch URL")
    parser.add_argument("--index", default=os.getenv("ES_INDEX", "products_sku_v1"), help="Index name")
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--topn", type=int, default=20, help="Top N results")

    # term filters (repeatable)
    parser.add_argument("--season", action="append", help="Season filter, can repeat")
    parser.add_argument("--scene", action="append", help="Scene filter, can repeat")
    parser.add_argument("--material", action="append", help="Material filter, can repeat")
    parser.add_argument("--style", action="append", help="Style filter, can repeat")
    parser.add_argument("--gender", action="append", help="people_gender filter, can repeat")
    parser.add_argument("--age", action="append", help="age_range filter, can repeat")
    parser.add_argument("--color", action="append", help="Color filter, can repeat")
    parser.add_argument("--size", action="append", help="Size filter, can repeat")

    # exact keyword brand/group
    parser.add_argument("--brand_kw", action="append", help="Exact brand_name.kw filter, can repeat")
    parser.add_argument("--group_kw", action="append", help="Exact group_name.kw filter, can repeat")

    # range filters
    parser.add_argument("--price_lte", type=float, default=None, help="price <= X")
    parser.add_argument("--price_gte", type=float, default=None, help="price >= X")
    parser.add_argument("--sales_gte", type=int, default=None, help="sales >= X")

    # knobs
    parser.add_argument("--knn_k", type=int, default=200, help="kNN k")
    parser.add_argument("--knn_num_candidates", type=int, default=1500, help="kNN num_candidates")
    parser.add_argument("--rrf_window", type=int, default=200, help="RRF rank_window_size")

    args = parser.parse_args()

    filters = _parse_cli_filters(args)
    hits = search(
        query=args.q,
        filters=filters,
        topn=args.topn,
        es_url=args.es,
        index_name=args.index,
        knn_k=args.knn_k,
        knn_num_candidates=args.knn_num_candidates,
        rrf_rank_window_size=args.rrf_window,
    )

    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
