import os
import json
from openai import OpenAI
from textwrap import dedent
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ======================
# 配置
# ======================
INPUT_FILE = os.path.join(PROJECT_ROOT, "data", "opensearch_product_data.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "generated_questions.jsonl")
MODEL_NAME = "qwen3-max"
MAX_WORKERS = 10  # ← 控制并发线程数，可根据 API 限流调整（如 DashScope 限流请调低）

# 初始化 DashScope 兼容 OpenAI 的客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_URL")
)

# ======================
# 优化后的 Prompt（聚焦：只输出5条纯问题）
# ======================
SYSTEM_PROMPT = dedent("""
    你是一个真实用户，正在电商平台（比如淘宝、京东、小红书）上逛服饰鞋包类商品。
    你的任务是：
    - 请根据下面的商品信息，以**第一人称或自然提问口吻**，生成5条**简短、口语化、真实**的搜索问题。
    - 这些提问应模拟用户在选购阶段可能提出的疑问，覆盖以下常见意图维度：
        - 功能/效果需求（如显瘦、保暖、防皱）
        - 身材/肤质/使用场景适配（如小个子、敏感肌、通勤）
        - 成分/材质/人群偏好（如纯棉、亚麻、儿童安全）
        - 价格/品牌/促销相关（如平价、高端、是否值得买）
        - 多品类组合或流行趋势（如搭配、爆款、2025春季新款）
        - 特殊需求（如礼盒、明星同款、显白、尺码建议等）

        请确保：
        - 每条必须是完整或省略但可理解的问句，像用户真的会打出来的那样；
        - 问题必须基于商品数据中的具体信息（如品牌、品类、颜色、材质、价格、适用场景等）；
        - 可以带点模糊、错别字、语气词（比如“显白不？”“有推荐吗？”“会起球吗？”）；
        - 语言自然口语化，像真实用户在电商平台搜索框或客服对话中会说的句子；
        - 可包含模糊表达、错别字假设、口语省略（如“有推荐吗？”“显白不？”）；
        - 覆盖不同角度：穿搭效果、尺码建议、是否显瘦/显高/显白、材质舒服吗、值不值得买、有没有同款等；
        - 不要重复示例，但可参考其风格；
        - 每条问题独立成一行，不加编号或引号。
""").strip()


def generate_questions_for_product(product: dict) -> dict:
    """
    返回完整的输出记录：{ "skuid": "...", "questions": [...] }
    """
    skuid = product["skuid"]
    info = {
        "品类": product.get("category", ""),
        "商品名": product.get("product_name", ""),
        "品牌": product.get("brand", ""),
        "价格": product.get("price", ""),
        "颜色": product.get("color", ""),
        "材质": product.get("material", ""),
        "适用人群": product.get("target_audience", ""),
        "适用场景": product.get("scene", ""),
        "卖点": product.get("features", ""),
        "尺码说明": product.get("size_info", ""),
    }
    info = {k: v for k, v in info.items() if v}

    user_content = f"商品信息：{json.dumps(info, ensure_ascii=False, separators=(',', ':'))}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.85,
            max_tokens=180,
            stream=False
        )
        raw = completion.choices[0].message.content.strip()
        lines = [line.strip() for line in raw.split('\n') if line.strip()]
        while len(lines) < 5:
            lines.append("")
        questions = lines[:5]
    except Exception as e:
        print(f"❌ Error for skuid={skuid}: {e}")
        questions = ["", "", "", "", ""]

    return {
        "skuid": skuid,
        "questions": questions
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 第一步：预加载所有有效商品（带 skuid）
    valid_products = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                product = json.loads(line)
                if product.get("skuid"):
                    valid_products.append(product)
            except json.JSONDecodeError:
                continue

    total = len(valid_products)
    print(f"🎯 Found {total} valid products with skuid. Starting generation with {MAX_WORKERS} threads...")

    # 第二步：多线程处理
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_skuid = {
            executor.submit(generate_questions_for_product, product): product["skuid"]
            for product in valid_products
        }

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_skuid), total=total, desc="Generating questions", unit="product"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                skuid = future_to_skuid[future]
                print(f"⚠️ Unexpected error for skuid={skuid}: {e}")

    # 第三步：写入文件（保持与原始顺序无关，如需保持顺序可改用 list + index）
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        for record in results:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        fout.flush()

    print(f"✅ Done! Strict format results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()