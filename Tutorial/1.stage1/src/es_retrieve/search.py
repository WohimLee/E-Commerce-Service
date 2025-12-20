# search.py
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


from es_client import get_es
from embedder import get_embedder


from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

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
      - bool.filter (BM25)
      - knn.filter (kNN)

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
        # 精确匹配品牌/品类（keyword 子字段）
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
# Query builder (NO ES-RRF)
# ----------------------------

def build_bm25_body(
    query: str,
    filters: Dict[str, Any],
    size: int,
    source_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bool_filters, _ = build_es_filters(filters)

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
        "size": size,
        "track_total_hits": False,
        "_source": source_fields or DEFAULT_SOURCE_FIELDS,
        "query": {
            "bool": {
                "filter": bool_filters,
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": mm_fields,
                            "type": "best_fields",
                        }
                    }
                ],
            }
        },
    }
    return body


def build_knn_body(
    query_vector: List[float],
    filters: Dict[str, Any],
    size: int,
    *,
    knn_k: int,
    knn_num_candidates: int,
    source_fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    _, knn_filters = build_es_filters(filters)

    # ES 8.x：top-level knn 支持 filter
    body: Dict[str, Any] = {
        "size": size,
        "track_total_hits": False,
        "_source": source_fields or DEFAULT_SOURCE_FIELDS,
        "knn": {
            "field": "emb_text",
            "query_vector": query_vector,
            "k": knn_k,
            "num_candidates": knn_num_candidates,
            "filter": knn_filters,
        },
    }
    return body


# ----------------------------
# Python-side RRF (license-free)
# ----------------------------

def rrf_fuse(
    bm25_hits: List[Dict[str, Any]],
    knn_hits: List[Dict[str, Any]],
    *,
    topn: int,
    rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion:
      score(doc) = sum_i 1 / (rank_constant + rank_i)
    rank is 1-based in each list.

    We also keep _source by taking the first seen (prefer bm25 then knn).
    """
    scores: Dict[str, float] = {}
    sources: Dict[str, Dict[str, Any]] = {}

    def add_list(hits: List[Dict[str, Any]]):
        for i, h in enumerate(hits):
            doc_id = str(h.get("_id"))
            if not doc_id or doc_id == "None":
                continue
            rank = i + 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)
            if doc_id not in sources:
                sources[doc_id] = h.get("_source", {}) or {}

    # prefer bm25 source
    add_list(bm25_hits)
    add_list(knn_hits)

    ranked_ids = sorted(scores.keys(), key=lambda _id: scores[_id], reverse=True)[:topn]
    out: List[Dict[str, Any]] = []
    for _id in ranked_ids:
        out.append({
            "_id": _id,
            "_score": scores[_id],
            "_source": sources.get(_id, {}),
        })
    return out


def _normalize_es_hits(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    hits = resp.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []
    for h in hits:
        out.append({
            "_id": h.get("_id"),
            "_score": h.get("_score"),
            "_source": h.get("_source", {}),
        })
    return out


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
    # 下面这俩以前是 ES-RRF 的参数，现在变成“各路召回窗口”和“Python RRF 融合常数”
    rrf_rank_window_size: int = 200,
    rrf_rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """
    Hybrid search WITHOUT ES retriever.rrf (license-free):
      1) BM25 multi_match -> get window hits
      2) kNN vector search -> get window hits
      3) Python-side RRF fuse -> return topn

    filters: see build_es_filters()
    """
    if not query or not query.strip():
        raise ValueError("query must be non-empty")

    es_url = es_url or os.getenv("ES_URL", "http://localhost:9200")
    index_name = index_name or os.getenv("ES_INDEX", "products_sku_v1")

    es = get_es(es_url=es_url)
    embedder = get_embedder()

    query_vector = embedder.embed([query])[0]

    # 召回窗口：越大效果越好但更慢；默认用原来的 rrf_rank_window_size
    window = max(int(rrf_rank_window_size), topn)

    # BM25
    bm25_body = build_bm25_body(
        query=query,
        filters=filters,
        size=window,
        source_fields=source_fields,
    )
    bm25_resp = es.search(index=index_name, body=bm25_body)
    bm25_hits = _normalize_es_hits(bm25_resp)

    # kNN：k 需要 >= window 才能拿到足够候选
    knn_k_eff = max(int(knn_k), window)
    knn_body = build_knn_body(
        query_vector=query_vector,
        filters=filters,
        size=window,
        knn_k=knn_k_eff,
        knn_num_candidates=int(knn_num_candidates),
        source_fields=source_fields,
    )
    knn_resp = es.search(index=index_name, body=knn_body)
    knn_hits = _normalize_es_hits(knn_resp)

    # Python-side RRF fuse
    fused = rrf_fuse(
        bm25_hits=bm25_hits,
        knn_hits=knn_hits,
        topn=topn,
        rank_constant=int(rrf_rank_constant),
    )
    return fused


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
    parser = argparse.ArgumentParser(description="Hybrid search (BM25 + vector) with Python-side RRF (license-free)")
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
    parser.add_argument("--knn_k", type=int, default=200, help="kNN k (will be max(knn_k, window))")
    parser.add_argument("--knn_num_candidates", type=int, default=1500, help="kNN num_candidates")
    parser.add_argument("--window", type=int, default=200, help="Retrieve window for each channel (bm25/knn)")
    parser.add_argument("--rrf_c", type=int, default=60, help="RRF rank_constant (Python-side)")

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
        rrf_rank_window_size=args.window,
        rrf_rank_constant=args.rrf_c,
    )

    print(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
