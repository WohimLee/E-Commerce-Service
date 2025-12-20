import os
import json
import time
import sys
from typing import Optional, Dict, Any, List, Tuple
from textwrap import dedent
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# =========================
# Paths / Config
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "opensearch_product_data.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "queries.jsonl")
FAILED_PATH = os.path.join(PROJECT_ROOT, "output", "gen_failed_ids.txt")
PROBLEM_LINES_PATH = os.path.join(PROJECT_ROOT, "output", "problem_line_idxs.txt")  # 记录所有有问题的行 idx

ID_FIELD = "skuid"

MAX_RETRIES = 3
RETRY_SLEEP_BASE = 2.0  # 秒（指数退避基数）
MODEL_NAME = "qwen3-max"

# 并发线程数（你说大概 10）
MAX_WORKERS = 10

SYSTEM_PROMPT = dedent("""
你现在是一名电商平台的「用户提问生成助手」，目标是：
为【服饰 / 鞋帽类单品】生成高度真实的用户提问，
用于测试 RAG / 向量检索 / 混合检索在“选购阶段”的召回与命中效果。

【用户所处阶段】
- 正在选购、对比、犹豫确认
- 关注穿搭效果、舒适度、尺码、价格是否合适
- 不是学习服装知识，而是想“买得对”

【问题类型必须覆盖（可混合）】

A. 功能 / 使用需求
- 显瘦、显高、显腿长
- 不起球、不变形、不闷、不磨脚
- 透气、舒适、好打理
- 通勤 / 上班 / 运动 / 日常两穿

B. 人群 / 身型
- 男生 / 女生
- 小个子 / 微胖 / 大码
- 中老年 / 儿童 / 宝宝
- 孕妇 / 哺乳期

C. 面料 / 穿着偏好
- 纯棉、羊毛、羊绒是否值得
- 不扎皮肤、亲肤
- 防滑、不磨脚（鞋类）
- 对肤感、季节友好度的关注

D. 价格 / 品牌 / 场景
- 平价好穿 / 高端大牌是否值得
- 某品牌衣服怎么样
- 是否有过更低价 / 是否能低价提醒
- 商场 / 专柜同款

E. 混合需求（多意向）
- 上衣 + 下装 + 鞋子一套搭配
- 户外 / 露营穿搭
- 家居服 / 睡衣
- 当季流行色、流行款

F. 其他真实场景
- 出差 / 旅行穿什么方便
- 进口 / 国外品牌
- 明星同款
- 之前推荐过的
- 测评过的好穿款
- 肤色偏黄穿什么颜色
- 尺码咨询（如 160cm / 120斤穿什么码）
- 错别字、模糊搜索（如“好看的已付/裤子”）

【生成规则】
- 每次必须生成 5 条差异明显的问题
- 不要只做同义改写，要体现不同选购视角
- 问题应贴近真实电商用户表达，允许：
  - 口语化
  - 模糊描述
  - 主观感受（显不显瘦、舒不舒服、会不会闷）
- 可以包含价格、搭配、尺码、场景等关切

【严格约束】
- 所有问题都必须能合理命中这条商品
- 只能使用商品 JSON 中真实存在的信息（品类、性别、季节、面料、版型、适用场景等）
- 不允许编造商品不存在的属性或效果
- 不要出现“这条商品 / 上述 JSON / 商品数据”等描述

【输出格式（严格）】
只输出 JSON，不要解释，不要代码块，不要多余字段：

{
  "product_id": "<直接复用输入中的商品ID>",
  "queries": [
    "问题1",
    "问题2",
    "问题3",
    "问题4",
    "问题5"
  ]
}
""").strip()

# =========================
# Helpers
# =========================
def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def record_problem(ff: Any, idx: int, reason: str) -> None:
    ff.write(f"{idx}\t{reason}\n")
    ff.flush()


def extract_product_id(product: Dict[str, Any]) -> str:
    if ID_FIELD not in product:
        raise KeyError(f"字段 {ID_FIELD} 不在商品数据中，请根据你的 JSON 调整 ID_FIELD。")
    return str(product[ID_FIELD])


def build_user_prompt(product: Dict[str, Any]) -> str:
    product_json = json.dumps(product, ensure_ascii=False)
    return dedent(f"""
        下面是一条电商商品的结构化数据（JSON）：
        {product_json}

        其中 {ID_FIELD} 字段是这条商品的唯一 ID。

        请你站在「真实用户选购服饰 / 鞋帽」的视角，
        基于这条商品数据，按照 system 提示，生成 5 条自然语言用户问题。

        【生成要求】
        - 问题要像真实用户在电商平台会问的
        - 以选购、对比、搭配、确认为主
        - 可关注：显瘦显高、舒适度、面料、尺码、价格、场景
        - 表达允许口语、主观、模糊

        【输出要求（严格）】
        - 只输出 JSON
        - 不要输出解释、分析、说明
        - 字段格式必须是：
        - "product_id"：直接复用输入中的 {ID_FIELD}
        - "queries"：长度为 5 的字符串数组
        """).strip()


def validate_model_output(data: Dict[str, Any]) -> None:
    if "product_id" not in data or "queries" not in data:
        raise ValueError("模型输出缺少必需字段 product_id 或 queries")
    if not isinstance(data["queries"], list) or len(data["queries"]) != 5:
        raise ValueError("queries 必须是长度为 5 的数组")
    if not all(isinstance(q, str) and q.strip() for q in data["queries"]):
        raise ValueError("queries 中每一条必须是非空字符串")


def generate_for_product(
    client: OpenAI,
    product: Dict[str, Any],
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict[str, Any]]:
    product_id = extract_product_id(product)
    user_prompt = build_user_prompt(product)

    for attempt in range(1, max_retries + 1):
        try:
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )

            content = resp.choices[0].message.content
            data = json.loads(content)

            validate_model_output(data)

            # 强制改回真实 product_id，防止模型乱写
            data["product_id"] = product_id

            return data

        except Exception as e:
            print(
                f"[WARN] product_id={product_id} 调用失败，第 {attempt} 次重试，错误：{e}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                sleep_time = RETRY_SLEEP_BASE * (2 ** (attempt - 1))
                time.sleep(sleep_time)
            else:
                print(
                    f"[ERROR] product_id={product_id} 重试 {max_retries} 次仍失败。",
                    file=sys.stderr,
                )
                return None


# =========================
# Worker wrapper (线程任务)
# =========================
def process_one(
    client: OpenAI,
    idx: int,
    raw_line: str,
) -> Tuple[int, str, Optional[str], Optional[Dict[str, Any]]]:
    """
    返回:
      (idx, status, product_id, result)

    status:
      - "ok"
      - "skip_empty"
      - "skip_json_parse_failed"
      - "skip_missing_id"
      - "gen_failed"
    """
    line = raw_line.strip()
    if not line:
        return idx, "skip_empty", None, None

    try:
        product = json.loads(line)
    except Exception as e:
        return idx, f"skip_json_parse_failed: {e}", None, None

    try:
        product_id = extract_product_id(product)
    except KeyError as e:
        return idx, f"skip_missing_id: {e}", None, None

    result = generate_for_product(client, product)
    if result is None:
        return idx, "gen_failed", product_id, None

    return idx, "ok", product_id, result


# =========================
# Main
# =========================
def main() -> None:
    ensure_parent_dir(OUTPUT_PATH)
    ensure_parent_dir(FAILED_PATH)
    ensure_parent_dir(PROBLEM_LINES_PATH)

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_URL"),
    )

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    batch_size = 10  # 并发量
    failed_ids: List[str] = []

    with open(OUTPUT_PATH, "a", encoding="utf-8") as fout, \
         open(FAILED_PATH, "a", encoding="utf-8") as ffail, \
         open(PROBLEM_LINES_PATH, "a", encoding="utf-8") as fprob:

        for start in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
            batch_lines = lines[start : start + batch_size]
            batch_indices = list(range(start, start + len(batch_lines)))

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = executor.map(
                    lambda args: process_one(client, *args),
                    zip(batch_indices, batch_lines),
                )

            # executor.map 返回顺序 == 输入顺序
            for idx, status, product_id, result in results:
                if status == "ok" and result is not None:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    continue

                if status == "skip_empty":
                    record_problem(fprob, idx, "empty_line")
                elif status.startswith("skip_json_parse_failed"):
                    record_problem(fprob, idx, status)
                elif status.startswith("skip_missing_id"):
                    record_problem(fprob, idx, status)
                elif status == "gen_failed":
                    if product_id is not None:
                        failed_ids.append(product_id)
                        ffail.write(str(product_id) + "\n")
                        record_problem(
                            fprob, idx, f"generation_failed product_id={product_id}"
                        )
                    else:
                        record_problem(fprob, idx, "generation_failed product_id=None")
                else:
                    record_problem(fprob, idx, f"unknown_status: {status}")

            # 👉 可选：每个 batch 之间稍微歇一下，防止 QPS 峰值
            # time.sleep(0.1)

    print(
        f"[DONE] 所有商品处理完毕，失败 {len(failed_ids)} 条。\n"
        f"  - 失败商品ID列表：{FAILED_PATH}\n"
        f"  - 问题行idx记录：{PROBLEM_LINES_PATH}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
