# build_dataset.py
from __future__ import annotations

import argparse
import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


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


def ensure_list(x: Any) -> List[str]:
    """
    Normalize a field into a list of strings.
    - If already list, stringify items
    - If scalar, wrap into list
    - If None/missing, return []
    """
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if i is not None and str(i).strip() != ""]
    s = str(x).strip()
    return [s] if s else []


def join_list(xs: List[str]) -> str:
    """Join list for human-readable text. Keep short & stable."""
    if not xs:
        return ""
    return "、".join(xs)


def build_search_text(doc: Dict[str, Any]) -> str:
    """
    Build search_text for embedding and/or BM25 auxiliary field.

    Template:
      {group_name} {brand_name} {product_name}
      属性: 季节{season} 场景{scene} 材质{material} 风格{style} 人群{people_gender} 年龄{age_range} 颜色{color} 尺码{size}
      卖点: {marketing_attributes}
      描述: {product_description}
    """
    group_name = str(doc.get("group_name", "") or "").strip()
    brand_name = str(doc.get("brand_name", "") or "").strip()
    product_name = str(doc.get("product_name", "") or "").strip()

    season = join_list(ensure_list(doc.get("season")))
    scene = join_list(ensure_list(doc.get("scene")))
    material = join_list(ensure_list(doc.get("material")))
    style = join_list(ensure_list(doc.get("style")))
    people_gender = join_list(ensure_list(doc.get("people_gender")))
    age_range = join_list(ensure_list(doc.get("age_range")))
    color = join_list(ensure_list(doc.get("color")))
    size = join_list(ensure_list(doc.get("size")))

    marketing = str(doc.get("marketing_attributes", "") or "").strip()
    desc = str(doc.get("product_description", "") or "").strip()

    # Keep format stable (helpful for embeddings & debugging)
    lines = []
    title = " ".join([x for x in [group_name, brand_name, product_name] if x])
    if title:
        lines.append(title)

    lines.append(
        "属性: "
        f"季节{season} "
        f"场景{scene} "
        f"材质{material} "
        f"风格{style} "
        f"人群{people_gender} "
        f"年龄{age_range} "
        f"颜色{color} "
        f"尺码{size}"
    )

    if marketing:
        lines.append(f"卖点: {marketing}")
    if desc:
        lines.append(f"描述: {desc}")

    return "\n".join(lines).strip()


def normalize_key(x: Any) -> str:
    """Normalize skuid/spuid keys to string keyword."""
    if x is None:
        return ""
    return str(x).strip()


def merge_docs(
    product_row: Dict[str, Any],
    attrs_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge product core fields with attrs fields.
    Output doc is ES-ready (no vector yet).
    """
    doc: Dict[str, Any] = {}

    # -------- ids --------
    doc["skuid"] = normalize_key(product_row.get("skuid"))
    doc["spuid"] = normalize_key(product_row.get("spuid"))

    # -------- core text --------
    doc["brand_name"] = product_row.get("brand_name")
    doc["group_name"] = product_row.get("group_name")
    doc["product_name"] = product_row.get("product_name")
    doc["marketing_attributes"] = product_row.get("marketing_attributes")
    doc["product_description"] = product_row.get("product_description")

    # -------- numeric/business --------
    doc["price"] = product_row.get("price")
    doc["sales"] = product_row.get("sales")

    # -------- attrs (optional) --------
    attrs_row = attrs_row or {}
    for k in [
        "season", "scene", "material", "style",
        "people_gender", "age_range", "color", "size"
    ]:
        # keep as list for ES keyword multi-value field
        doc[k] = ensure_list(attrs_row.get(k))

    # -------- search_text --------
    doc["search_text"] = build_search_text(doc)

    return doc


def main():
    parser = argparse.ArgumentParser(description="Build merged product dataset for ES indexing.")
    parser.add_argument(
        "--products",
        default=os.path.join(PROJECT_ROOT, "data", "opensearch_product_data.jsonl"),
        help="Path to product core jsonl (opensearch_product_data.jsonl)",
    )
    parser.add_argument(
        "--attrs",
        default=os.path.join(PROJECT_ROOT, "output", "product_attrs.jsonl"),
        help="Path to product attrs jsonl (product_attrs.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "output", "merged_products.jsonl"),
        help="Output jsonl path",
    )
    parser.add_argument(
        "--keep_unmatched_products",
        action="store_true",
        help="Keep products even if attrs missing (default: keep)",
    )
    parser.add_argument(
        "--drop_unmatched_products",
        action="store_true",
        help="Drop products if attrs missing",
    )

    args = parser.parse_args()

    if args.drop_unmatched_products:
        keep_unmatched = False
    else:
        keep_unmatched = True if args.keep_unmatched_products or True else False  # default keep

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 1) Load attrs by skuid (string key)
    attrs_by_skuid: Dict[str, Dict[str, Any]] = {}
    attrs_count = 0
    for row in read_jsonl(args.attrs):
        skuid = normalize_key(row.get("skuid"))
        if not skuid:
            continue
        attrs_by_skuid[skuid] = row
        attrs_count += 1

    # 2) Stream products, merge, write out
    total_products = 0
    merged = 0
    missing_attrs = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for prow in read_jsonl(args.products):
            total_products += 1
            skuid = normalize_key(prow.get("skuid"))
            if not skuid:
                continue

            arow = attrs_by_skuid.get(skuid)
            if arow is None:
                missing_attrs += 1
                if not keep_unmatched:
                    continue

            doc = merge_docs(prow, arow)
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            merged += 1

    print("Done.")
    print(f"  attrs rows:         {attrs_count}")
    print(f"  products rows:      {total_products}")
    print(f"  merged rows:        {merged}")
    print(f"  products w/o attrs: {missing_attrs}")
    print(f"  output:             {args.out}")


if __name__ == "__main__":
    main()
