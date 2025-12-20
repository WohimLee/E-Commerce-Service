# schemas.py
"""
Elasticsearch index schemas for product SKU search (RAG / hybrid retrieval).

- One document per SKU
- Supports:
  - BM25 text search
  - keyword filters (season/scene/material/...)
  - dense_vector kNN (cosine)

mapping + settings 合并在一个 dict
→ create_index_if_not_exists 可以直接接，不用拼

dynamic: strict
→ 提前发现脏字段（电商数据很容易混）

keyword 字段全都可 filter / aggregation
→ 非常适合导购条件筛选

search_text 单独存在
→ embedding 文本可随时调整，不影响 BM25

版本化（V1 / V2）
→ 以后加多向量 / rerank / 冷启动字段不需要重想结构
"""

from __future__ import annotations
from typing import Dict, Any


# =========================
# Common analyzers/settings
# =========================

EMBEDDING_DIM = 1024

COMMON_SETTINGS: Dict[str, Any] = {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "analysis": {
        "analyzer": {
            # 索引时：最大化切分，提高召回
            "ik_index": {
                "type": "custom",
                "tokenizer": "ik_max_word"
            },
            # 搜索时：智能切分，降低噪声
            "ik_search": {
                "type": "custom",
                "tokenizer": "ik_smart"
            }
        }
    }
}

PRODUCT_MAPPING_V1: Dict[str, Any] = {
    "settings": COMMON_SETTINGS,
    "mappings": {
        "dynamic": "strict",
        "properties": {
            # -------- IDs --------
            "skuid": {"type": "keyword"},
            "spuid": {"type": "keyword"},

            # -------- Text fields (BM25 + IK) --------
            "product_name": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search"
            },
            "product_description": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search"
            },
            "marketing_attributes": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search"
            },

            # 拼接后的检索文本（embedding + BM25）
            "search_text": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search"
            },

            # -------- Brand / Category --------
            # text 用于模糊检索，kw 用于精确过滤
            "brand_name": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search",
                "fields": {
                    "kw": {"type": "keyword"}
                }
            },
            "group_name": {
                "type": "text",
                "analyzer": "ik_index",
                "search_analyzer": "ik_search",
                "fields": {
                    "kw": {"type": "keyword"}
                }
            },

            # -------- Business fields --------
            "price": {"type": "float"},
            "sales": {"type": "integer"},

            # -------- Structured filters --------
            # 全部 keyword，多值字段
            "season": {"type": "keyword"},
            "scene": {"type": "keyword"},
            "material": {"type": "keyword"},
            "style": {"type": "keyword"},
            "people_gender": {"type": "keyword"},
            "age_range": {"type": "keyword"},
            "color": {"type": "keyword"},
            "size": {"type": "keyword"},

            # -------- Vector field --------
            "emb_text": {
                "type": "dense_vector",
                "dims": 1024,           # ⚠️ 要和 embedding 模型一致
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}


# =========================
# Mapping v2 (reserved)
# =========================

PRODUCT_MAPPING_V2: Dict[str, Any] = {
    "settings": COMMON_SETTINGS,
    "mappings": {
        "dynamic": "strict",
        "properties": {
            **PRODUCT_MAPPING_V1["mappings"]["properties"]
            # 未来你可以在这里加：
            # - 多向量（emb_query / emb_title）
            # - created_at / updated_at
            # - rerank_score 等
        }
    }
}



# =========================
# Default export
# =========================

# 当前默认用 v1
PRODUCT_MAPPING: Dict[str, Any] = PRODUCT_MAPPING_V1


if __name__ == "__main__":
    from es_client import get_es, create_index_if_not_exists
    es = get_es()

    create_index_if_not_exists(
        es,
        index_name="products_sku_v1",
        mapping=PRODUCT_MAPPING
    )