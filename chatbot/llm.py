import os
from openai import OpenAI

_client = None

def _get_client():
    """获取或创建 OpenAI 客户端实例（延迟初始化）"""
    global _client
    if _client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 DASHSCOPE_API_KEY 环境变量\n"
                "请先调用 config.load_key.load_key() 或设置环境变量"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return _client

def invoke(user_message, model_name="qwen3-max"):
    client = _get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_message}]
    )
    return completion.choices[0].message.content

def invoke_with_stream_log(user_message, model_name="qwen3-max"):
    client = _get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": user_message}],
        stream=True
    )
    result = ""
    for response in completion:
        result += response.choices[0].delta.content
        print(response.choices[0].delta.content, end="")
    return result