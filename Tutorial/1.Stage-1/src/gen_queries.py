
import os
import json
from openai import OpenAI

from textwrap import dedent
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_URL"),
)


SYSTEM_PROMPT = dedent('''

''').strip()

USER_PROMPT = dedent('''

''').strip()

def gen_queries_from_data():

    res = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ],
        stream=False
    )
    for chunk in completion:
        print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == "__main__":

    file = "data/opensearch_product_data.jsonl"

    with open(os.path.join(PROJECT_ROOT, file)) as f:
        for item in f:
            line = json.loads(item)
            pass
    pass
