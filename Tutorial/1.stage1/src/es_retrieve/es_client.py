# es_client.py
from __future__ import annotations

import os
import sys
import time
import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from dotenv import load_dotenv

from schemas import PRODUCT_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ----------------------------
# ES Client
# ----------------------------

def get_es(
    es_url: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    verify_certs: bool = False,
    request_timeout: int = 30,
    max_retries: int = 3,
    retry_on_timeout: bool = True,
) -> Elasticsearch:
    """
    Create and return an Elasticsearch client.

    - Defaults to ES_URL env var or http://localhost:9200
    - Works for local dev (no auth) and secured clusters (api_key or basic auth).
    """
    es_url = es_url or os.getenv("ES_URL")
    api_key = api_key or os.getenv("ES_API_KEY")
    username = username or os.getenv("ES_USERNAME")
    password = password or os.getenv("ES_PASSWORD")

    kwargs: Dict[str, Any] = dict(
        hosts=[es_url],
        verify_certs=verify_certs,
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_on_timeout=retry_on_timeout,
    )

    if api_key:
        kwargs["api_key"] = api_key
    elif username and password:
        kwargs["basic_auth"] = (username, password)

    es = Elasticsearch(**kwargs)

    # Quick health check (fail fast)
    try:
        es.info()
        logger.info("Connected to Elasticsearch at %s", es_url)
    except Exception as e:
        logger.error("Failed to connect to Elasticsearch at %s: %s", es_url, e)
        raise

    return es


# ----------------------------
# Index Management
# ----------------------------

def create_index_if_not_exists(
    es: Elasticsearch,
    index_name: str,
    mapping: Dict[str, Any],
    settings: Optional[Dict[str, Any]] = None,
    wait_for_yellow: bool = True,
) -> bool:
    """
    Create an index if it does not exist.

    Parameters
    - mapping: dict for "mappings" section OR full body containing "mappings".
    - settings: optional dict for "settings" section
    Returns
    - True if created, False if already existed
    """
    if es.indices.exists(index=index_name):
        logger.info("Index already exists: %s", index_name)
        return False

    body: Dict[str, Any] = {}
    # Allow caller to pass either {"mappings": {...}} or just {...properties...}
    if "mappings" in mapping or "settings" in mapping:
        body.update(mapping)
    else:
        body["mappings"] = mapping

    if settings:
        body["settings"] = settings

    es.indices.create(index=index_name, body=body)
    logger.info("Created index: %s", index_name)

    if wait_for_yellow:
        try:
            es.cluster.health(index=index_name, wait_for_status="yellow", timeout="30s")
        except Exception as e:
            logger.warning("Health wait (yellow) failed or timed out: %s", e)

    return True


def delete_index(
    es: Elasticsearch,
    index_name: str,
    ignore_missing: bool = True,
) -> bool:
    """
    Delete index.

    Returns True if deleted, False if not found (when ignore_missing=True).
    """
    exists = es.indices.exists(index=index_name)
    if not exists:
        if ignore_missing:
            logger.info("Index not found (skip delete): %s", index_name)
            return False
        raise ValueError(f"Index not found: {index_name}")

    es.indices.delete(index=index_name)
    logger.info("Deleted index: %s", index_name)
    return True


# ----------------------------
# Bulk Index
# ----------------------------

def bulk_index(
    es: Elasticsearch,
    index_name: str,
    docs: Iterable[Dict[str, Any]],
    *,
    id_field: str = "skuid",
    op_type: str = "index",
    chunk_size: int = 500,
    max_retries: int = 3,
    initial_backoff: float = 0.5,
    request_timeout: int = 60,
    refresh: Union[bool, str] = False,
    raise_on_error: bool = False,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Bulk write documents to Elasticsearch.

    Parameters
    - docs: iterable of documents (dict). Each doc should contain id_field.
    - op_type: "index" (upsert/overwrite), "create" (fail if exists), "update" (requires doc wrapper).
    - refresh: False | True | "wait_for"
    - raise_on_error: if True, raise exception when any item fails

    Returns
    - (success_count, fail_count, failures)
      failures: list of error item dicts (truncated)
    """
    def to_actions() -> Iterable[Dict[str, Any]]:
        for d in docs:
            if id_field not in d:
                raise ValueError(f"Document missing id_field '{id_field}': {list(d.keys())[:20]}")
            _id = str(d[id_field])
            action: Dict[str, Any] = {
                "_op_type": op_type,
                "_index": index_name,
                "_id": _id,
                "_source": d,
            }
            yield action

    attempts = 0
    last_exc: Optional[Exception] = None

    while attempts <= max_retries:
        try:
            # elasticsearch.helpers.bulk returns (success_count, errors_list) when stats_only=False
            success, errors = bulk(
                es,
                to_actions(),
                chunk_size=chunk_size,
                request_timeout=request_timeout,
                refresh=refresh,
                raise_on_error=False,     # we handle errors ourselves
                raise_on_exception=False, # keep going, report failures
            )

            failures: List[Dict[str, Any]] = []
            if errors:
                # errors is a list of per-item error dicts; keep a compact copy
                for item in errors[:200]:  # avoid huge memory usage
                    failures.append(item)

            fail_count = len(errors) if errors else 0
            success_count = int(success) if isinstance(success, (int, float)) else 0

            if fail_count > 0:
                logger.warning(
                    "Bulk indexed with failures. success=%s fail=%s index=%s",
                    success_count, fail_count, index_name
                )
                if raise_on_error:
                    raise RuntimeError(f"Bulk indexing failed for {fail_count} items (see failures)")
            else:
                logger.info(
                    "Bulk indexed successfully. success=%s index=%s",
                    success_count, index_name
                )

            return success_count, fail_count, failures

        except Exception as e:
            last_exc = e
            attempts += 1
            if attempts > max_retries:
                break
            backoff = initial_backoff * (2 ** (attempts - 1))
            logger.warning("Bulk attempt %d/%d failed: %s; retry in %.2fs",
                           attempts, max_retries, e, backoff)
            time.sleep(backoff)

    # exhausted retries
    logger.error("Bulk indexing exhausted retries: %s", last_exc)
    raise last_exc if last_exc else RuntimeError("Bulk indexing failed with unknown error")


if __name__ == "__main__":
    es = get_es()

    create_index_if_not_exists(
        es,
        "products_sku_v1",
        mapping=PRODUCT_MAPPING  # 这里换成 schemas.py 的 mapping
    )

    docs = [{"skuid": "1", "product_name": "test"}]
    bulk_index(es, "products_sku_v1", docs, id_field="skuid", chunk_size=200, refresh="wait_for")