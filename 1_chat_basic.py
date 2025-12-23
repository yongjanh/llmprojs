# 说明：本文件涉及
# 1. 通义千问Max模型的简单流式输出
# 2. 通义千问Max模型的搜索增强功能
# 3. 通义千问Max模型的思维链增强功能
# 4. 涉及temperature和top_p的参数设置与影响
# 5. 简单引入knowledge作为背景信息

# 关于temperature:本质上是对logits进行缩放（计算e^x/z），z即是temperature，=1时不变，>1时，概率分布更平缓，<1时，概率分布更集中。
# 关于top_p (核采样): 对softmax后的概率分布进行截断。
# 将token按概率从高到低排序，累加概率直到达到top_p阈值，只在该子集中采样。
# =1时，考虑所有token；<1时，只保留累计概率达到top_p的高概率token子集。
# 例如top_p=0.8，则只在累计概率达到80%的最高概率token中采样。

import os
from config.load_key import load_key
load_key()
# print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')

from openai import OpenAI
import os
import time
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def get_qwen_response(prompt):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            # system message 用于设置大模型的角色和任务
            {"role": "system", "content": "你负责教育内容开发公司的答疑，你的名字叫公司小蜜，你要回答同事们的问题。"},
            # user message 用于输入用户的问题
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_qwen_stream_response(user_prompt, system_prompt, temperature, top_p):
    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        stream=True
    )
    
    for chunk in response:
        yield chunk.choices[0].delta.content

# temperature，top_p的默认值使用通义千问Max模型的默认值
def print_qwen_stream_response(scene,user_prompt, system_prompt, temperature=0.7, top_p=0.8, iterations=10):
    print(f"场景：{scene}")
    for i in range(iterations):
        print(f"输出 {i + 1} : ", end="")
        ## 防止限流，添加延迟
        time.sleep(0.5)
        response = get_qwen_stream_response(user_prompt, system_prompt, temperature, top_p)
        output_content = ''
        for chunk in response:
            output_content += chunk
        print(output_content)


def enable_search_sample():
    completion = client.chat.completions.create(
    model="qwen-max",  # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "巴黎奥运会中国队金牌数"},
    ],
    extra_body={"enable_search": True},
    )
    print(completion.choices[0].message.content)


def enable_thinking_sample():
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model="qwen3-235b-a22b-thinking-2507",  # 此处以 qwen3-235b-a22b-thinking-2507 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": "9.9和9.11谁大"}
        ],
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        stream_options={
            "include_usage": True
        }
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content

def main():
    # ------------------通义千问Max模型：temperature的取值范围是[0, 2)，默认值为0.7
# 设置temperature=0
    # print_qwen_stream_response(scene="temperature=0",user_prompt="马也可以叫做", system_prompt="请帮我续写内容，字数要求是4个汉字以内。", temperature=0)

# 设置temperature=1.9
    # print_qwen_stream_response(scene="temperature=1.9",user_prompt="马也可以叫做", system_prompt="请帮我续写内容，字数要求是4个汉字以内。", temperature=1.9)

# ------------------通义千问Max模型：top_p取值范围为(0,1]，默认值为0.8。
# 设置top_p=0.001
    # print_qwen_stream_response(scene="top_p=0.001",user_prompt="为一款智能游戏手机取名，可以是", system_prompt="请帮我取名，字数要求是4个汉字以内。", top_p=0.001)
# 设置top_p=0.8
    # print_qwen_stream_response(scene="top_p=0.8",user_prompt="为一款智能游戏手机取名，可以是",system_prompt="请帮我取名，字数要求是4个汉字以内。", top_p=0.8)
    # print("simple response: ", get_qwen_response("我们公司项目管理应该用什么工具"))
#-------------------通义千问Max模型的搜索增强功能
    # enable_search_sample()
#-------------------通义千问Max模型的思维链增强功能
    # enable_thinking_sample()

    user_question = "我是软件一组的，请问项目管理应该用什么工具"

    knowledge = """公司项目管理工具有两种选择：
    1. **Jira**：对于软件开发团队来说，Jira 是一个非常强大的工具，支持敏捷开发方法，如Scrum和Kanban。它提供了丰富的功能，包括问题跟踪、时间跟踪等。

    2. **Microsoft Project**：对于大型企业或复杂项目，Microsoft Project 提供了详细的计划制定、资源分配和成本控制等功能。它更适合那些需要严格控制项目时间和成本的场景。
    
    在一般情况下请使用Microsoft Project，公司购买了完整的许可证。软件研发一组、三组和四组正在使用Jira，计划于2026年之前逐步切换至Microsoft Project。
    """

    response = get_qwen_stream_response(
        user_prompt=user_question,
        # 将公司项目管理工具相关的知识作为背景信息传入系统提示词
        system_prompt="你负责教育内容开发公司的答疑，你的名字叫公司小蜜，你要回答学员的问题。"+ knowledge,
        temperature=0.7,
        top_p=0.8
    )

    for chunk in response:
        print(chunk, end="")



if __name__ == "__main__":
    main()