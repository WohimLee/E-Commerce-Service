import os
import json
import time
from typing import Dict, Any, List, Iterable, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# =========================
# Config
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "opensearch_product_data.jsonl")
# ✅ 改为 JSONL 输出
OUTPUT_JSONL_PATH = os.path.join(PROJECT_ROOT, "output", "product_attrs.jsonl")

MODEL_NAME = "qwen3-max"
BATCH_SIZE = 5
MAX_RETRIES = 3

FIELDS = ["skuid", "season", "scene", "material", "style", "people_gender", "age_range", "color", "size"]

# =========================
# Prompts (no enum restriction; extract from raw text)
# =========================

SYSTEM_PROMPT = """
你是一个电商商品属性抽取与清洗器，只做结构化抽取与轻量纠错，不闲聊、不解释。

任务：
从每条商品记录的 product_name、marketing_attributes、product_description、brand_name、group_name 等文本中，
抽取并清洗以下字段。输出用于后续检索/过滤，因此要求：
- 信息尽量来自原文
- 但遇到明显错误/错别字/乱码/不可能的值，需要主动纠正或剔除

需要抽取的字段：
- skuid: 原样复制输入
- season: 季节/季候相关词（如：春季、夏季、秋冬、四季、春夏、秋冬、winter 等）
- scene: 使用场景/场合（如：通勤、上班、商务、休闲、运动、户外、居家、旅行、校园、派对、日常、上学等）
- material: 材质/面料/成分（如：棉、纯棉、聚酯、涤纶、羊毛、真皮、PU、羽绒、锦纶、尼龙、亚麻、莱赛尔、粘纤等；可以包含配比：如“94%聚酯纤维+6%氨纶”）
- style: 风格/款式关键词（如：通勤、简约、街头、复古、法式、赫本、轻奢、运动风、国潮、可爱风、工装等）
- people_gender: 适用性别/人群（如：男、女、男女同款/中性、儿童、男童、女童、婴童等）
- age_range: 年龄段（如：婴儿/婴童、幼儿、儿童、小童、中童、大童、青少年、青年、中年、老年、成人等）
- color: 颜色词/配色描述（如：黑色、白色、米白、深灰、藏蓝、牛仔蓝、卡其、咖色、复古棕、画布白、“白/浅卡其”等）
- size: 直接从文本抽取出现的尺码/尺寸原始字符串（如：S、M、L、XL、XXL、120、120cm、40、41、27.5*10.5*24.5cm、85E/38E 等）

输出格式（强制）：
1) 每条商品输出 1 行 JSON（JSON Lines），不输出数组、不输出 markdown、不输出解释。
2) 字段必须且只能包含：
   skuid, season, scene, material, style, people_gender, age_range, color, size
3) 类型强制统一：
   - skuid 为字符串
   - 其余字段全部为字符串数组 list[str]
   - 找不到就输出 []（不要 null、不要空字符串）

核心规则（非常重要）：

A. 复合词拆分（适用于 season / scene / style / people_gender / age_range / color）
- 如果一个值明显由多个概念拼接组成，必须拆分为多个更基础的标签。
- 例：
  - "商务通勤" -> ["商务","通勤"]
  - "日常休闲" -> ["日常","休闲"]
  - "出游旅行" -> ["出游","旅行"]
  - "都市休闲" -> ["都市","休闲"]
  - "时尚运动" -> ["时尚","运动"]
  - "秋冬季" -> ["秋季","冬季"]
  - "春秋冬" -> ["春季","秋季","冬季"]
  - "春夏秋冬" -> ["春季","夏季","秋季","冬季"]
  - "男女同款" -> ["男女同款"]（不拆成男/女，除非文本明确“男款/女款”）
- 拆分时保留顺序、去重。

B. 主动纠错/修正（强制）
- 对明显错别字、乱码、拼写错误、异常短语要“修正为最可能的正确值”，否则删除。
- 纠错优先级：以原文附近上下文（同一句/同段）为证据；无证据则用行业常识做最小改动。
- 颜色纠错要求（重点）：
  1) 颜色必须是“自然语言可理解”的颜色/配色描述。
  2) 遇到明显异常（如随机字母、乱码、无意义组合），必须修正或剔除。
  3) 若能从同条记录中找到更合理颜色（如“咖色/棕色/卡其/米色/咖啡色/浅咖”等），用更合理的那个替换；
     若只能推断到大类，用大类（如“咖色”/“棕色”/“卡其”/“米色”）；
     若完全没有依据，输出 []（不要输出异常词）。
  4) 示例：["咖调CC抱"] 属于明显异常，应优先修正为 ["咖色"] 或 ["卡其"]/["棕色"]（以文本证据为准），无证据则用 ["咖色"] 作为最小修正。
- 材质纠错：
  - 允许保留成分百分比，但避免重复大类（如既有“65.1%绵羊毛”又单独输出“羊毛”），优先保留“带比例/更具体”的那条。
  - 明显不是材质的（如“30.5g充绒量”“600g人造长毛绒”中的“600g”）：
    - 若能提取出材质本体（如“人造长毛绒”），保留材质本体；
    - 计量信息丢弃。
- 场景/风格纠错：
  - 不要生造营销词堆砌；仅在文本明确出现或高度可推断时保留。

C. 规范化与清洗
- 去重：同一字段内去重（保留首次出现顺序）。
- 清洗：去掉首尾空格、去掉明显无意义的标点包裹（如多余的【】），但保留内容本身。
- 不要输出“未知/不详/无/other”等占位词：找不到就 []。
- 不要回显输入原文。

请严格执行以上规则输出。
""".strip()



USER_PROMPT_HEADER = """
请严格按 system prompt 抽取与清洗字段。

输入：JSON Lines，每行一个商品对象，包含字段：
skuid, product_name, marketing_attributes, product_description, price, brand_name, group_name

输出：JSON Lines，每行一个 JSON 对象，只包含字段：
skuid, season, scene, material, style, people_gender, age_range, color, size

注意：
- 除 skuid 外，其余字段必须全部输出为列表（list），缺失用 []。
- 必须拆分复合词（如“商务通勤”->["商务","通勤"]，“秋冬季”->["秋季","冬季"]）。
- 必须主动纠错：明显异常值（尤其颜色乱码/错别字）要修正为合理值或删除。

下面是本批次商品记录：
""".strip()



# =========================
# IO / batching
# =========================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSONL decode error at line {line_no}: {e}") from e


def batched(it: Iterable[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    buf: List[Dict[str, Any]] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def build_prompt_batch(items: List[Dict[str, Any]]) -> str:
    lines = []
    for obj in items:
        minimal = {
            "skuid": obj.get("skuid"),
            "product_name": obj.get("product_name"),
            "marketing_attributes": obj.get("marketing_attributes"),
            "product_description": obj.get("product_description"),
            "brand_name": obj.get("brand_name"),
            "group_name": obj.get("group_name"),
            "price": obj.get("price"),
        }
        lines.append(json.dumps(minimal, ensure_ascii=False))
    return USER_PROMPT_HEADER + "\n" + "\n".join(lines)


def count_non_empty_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# =========================
# Parse
# =========================
def parse_jsonl_relaxed(text: str) -> List[Dict[str, Any]]:
    """宽松解析：逐行 json.loads；解析失败的行直接跳过。"""
    out: List[Dict[str, Any]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def keep_model_output(raw: Dict[str, Any], fallback_skuid: str) -> Dict[str, Any]:
    """
    去掉所有校验/枚举归一化，直接保留大模型原始输出：
    - 只做 skuid 兜底（避免 key 丢失）
    - 其他字段原样写入（可能是 str/list/None/任意类型）
    """
    res: Dict[str, Any] = {k: None for k in FIELDS}

    skuid = raw.get("skuid")
    sid = skuid.strip() if isinstance(skuid, str) and skuid.strip() else fallback_skuid
    res["skuid"] = sid

    for k in FIELDS:
        if k == "skuid":
            continue
        res[k] = raw.get(k, None)

    return res


# =========================
# JSONL output helpers
# =========================
def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_ids_jsonl(path: str) -> Set[str]:
    """断点续跑：读取已生成的 JSONL，收集 skuid。坏行跳过。"""
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = str(obj.get("skuid") or "").strip()
                if sid:
                    done.add(sid)
            except Exception:
                continue
    return done


# =========================
# Model call
# =========================
def call_model(client: OpenAI, prompt: str) -> List[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )
            text = resp.choices[0].message.content or ""
            return parse_jsonl_relaxed(text)
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Model call failed after {MAX_RETRIES} retries: {last_err}") from last_err


# =========================
# Main
# =========================
def main() -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_URL")

    if not api_key or not base_url:
        raise RuntimeError("Missing DASHSCOPE_API_KEY or DASHSCOPE_API_URL in env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    total_items = count_non_empty_lines(INPUT_PATH)

    # ✅ 断点续跑：从 output jsonl 里读取已完成 skuid
    done_ids = load_done_ids_jsonl(OUTPUT_JSONL_PATH)

    with tqdm(total=total_items, desc="Extracting", unit="item") as pbar:
        for batch in batched(iter_jsonl(INPUT_PATH), BATCH_SIZE):
            todo_batch: List[Dict[str, Any]] = []
            for item in batch:
                sid = str(item.get("skuid") or "").strip()
                if sid and sid in done_ids:
                    continue
                todo_batch.append(item)

            # 进度条仍按输入推进（含跳过项）
            if not todo_batch:
                pbar.update(len(batch))
                continue

            pbar.set_postfix_str("calling model")
            prompt = build_prompt_batch(todo_batch)
            parsed = call_model(client, prompt)

            # 建索引：skuid -> 模型输出对象
            pred_map: Dict[str, Dict[str, Any]] = {}
            for obj in parsed:
                sid = obj.get("skuid")
                if sid is None:
                    continue
                sid_str = str(sid).strip()
                if sid_str:
                    pred_map[sid_str] = obj

            # ✅ 写 JSONL：每个商品一行，立即落盘
            pbar.set_postfix_str("writing")
            for item in todo_batch:
                sid = str(item.get("skuid") or "").strip()
                if not sid:
                    continue

                raw = pred_map.get(sid)
                if raw is None:
                    sid_alt = sid.lstrip("0")
                    raw = pred_map.get(sid_alt, {})

                kept = keep_model_output(raw, fallback_skuid=sid)

                out_line = {k: kept.get(k, None) for k in FIELDS}
                append_jsonl(OUTPUT_JSONL_PATH, out_line)
                done_ids.add(sid)

            pbar.update(len(batch))
            pbar.set_postfix_str("")

    print(f"Done. Output: {OUTPUT_JSONL_PATH}", flush=True)


if __name__ == "__main__":
    main()
