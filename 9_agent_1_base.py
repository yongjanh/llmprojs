# ==============================================================================
# 说明：本文件演示 Agent (智能体) 的基础构建与进阶应用
#
# 【核心目标】
# 从零开始，逐步展示如何构建一个具备工具调用能力的 AI Agent。
#
# 【演示步骤】
# 1. 硬编码工具调用 (Hardcoded Tool Call): 
#    - 理解 Agent 的本质循环：思考(Think) -> 行动(Act) -> 观察(Observe)。
#    - 展示最原始的“人工”Agent 工作流。
#
# 2. 意图识别与结构化输出 (Intent & Structure): 
#    - 解决 Agent 的核心难点：如何让 LLM 输出机器可读的指令。
#    - 核心技术：Prompt Engineering (意图引导) + Pydantic (结构化约束/校验)。
#    - 扩展知识：Controlled Decoding (受控解码) —— 模型服务商通过屏蔽非法 token 直接保证 JSON 语法的合法性。
#
# 3. Function Calling (OpenAI Native): 
#    - 使用 OpenAI 原生 API 简化工具调用流程。
#    - 核心机制：tools 定义注入 + 服务端 JSON 生成 + 客户端执行闭环。
#
# 4. ReAct Agent (AgentScope): 
#    - 引入 Agent 框架 (AgentScope) 来简化开发。
#    - 核心模式：ReAct (Reason+Act) —— 自动维护 "思考-行动-观察" 的历史轨迹。
#
# 5. MCP (Model Context Protocol): 
#    - 展示如何连接远程工具生态，实现规模化的工具管理。
#    - 连接阿里云 DashScope 的联网搜索 MCP 服务。
#
# ==============================================================================

import os
import json
import asyncio
from textwrap import dedent
from typing import Union, Literal
from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from openai import OpenAI

from config.load_key import load_key
from chatbot.llm import _get_client

# 加载环境变量
load_key()

# AgentScope imports (适配 v1.0.9+)
try:
    import asyncio
    import os
    from agentscope.agent import ReActAgent
    from agentscope.mcp import HttpStatelessClient
    from agentscope.tool import Toolkit
    from agentscope.model import DashScopeChatModel
    from agentscope.message import Msg
    from agentscope.formatter import DashScopeChatFormatter 
except ImportError as e:
    print("="*50)
    print(f"CRITICAL WARNING: agentscope import failed!")
    print(f"Error details: {e}")
    print("Step 4 & 5 will fail with NameError because required classes are missing.")
    print("="*50)

# ==============================================================================
# Step 1: 硬编码工具调用
# 【原理】手动模拟 Agent 的完整思考与执行回路，直观理解 "AI + Tool" 的协作方式。
# ==============================================================================
def step1_hardcoded_tool():
    print("\n" + "="*20 + " STEP 1: 硬编码工具调用 " + "="*20)
    
    # 1. 用户的原始请求
    user_request = "你好，请帮我搜集一些关于 Transformer 模型的最新资料。"
    print(f"用户请求: {user_request}")

    # 2. “硬编码”执行工具函数 (模拟)
    def web_search(query: str):
        """模拟执行网络搜索并返回JSON格式的结果"""
        print(f"--- [工具执行中] 正在搜索: {query} ---")
        return '''{
            "results": [
                {"title": "Attention Is All You Need (Transformer 论文原文)", "url": "https://arxiv.org/abs/1706.03762", "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks..."},
                {"title": "The Illustrated Transformer – Jay Alammar", "url": "https://jalammar.github.io/illustrated-transformer/", "snippet": "A visual and intuitive explanation of the Transformer model."}
            ]
        }'''

    # 模拟 LLM 决定调用了工具
    tool_result = web_search(query=user_request)
    print(f"工具结果: {tool_result}\n")

    # 3. 将用户请求和工具结果拼接成一个提示，发送给大模型
    client = _get_client()
    completion = client.chat.completions.create(
        model="qwen-plus", 
        messages=[
            {'role': 'system', 'content': '你是一位课程研究助理，你的任务是根据工具的执行结果，向用户生成一句友好和清晰的回复。'},
            {'role': 'user', 'content': f'用户原始请求: "{user_request}"\n工具执行结果: {tool_result}'}
        ]
    )

    # 4. 输出模型的最终回复
    final_response = completion.choices[0].message.content
    print(f"模型生成的最终回复:\n{final_response}")


# ==============================================================================
# Step 2: 意图识别与结构化输出
# 【原理】构建”引导-校验-重试”闭环，解决 Agent 开发中最核心的"指令遵循"难题。
# 
# 1. 意图识别 (Intent Recognition):
#    - 将用户自然语言转换为确定的工具选择 (Tool Selection)。
#    - 依赖清晰的 Prompt 定义可选工具及其适用场景。
#
# 2. 结构化输出 (Structured Output):
#    - 本质：将 LLM 的输出转换为机器可读的 JSON 格式。
#    - 核心：使用 Pydantic 定义严格的 JSON Schema，并结合 TypeAdapter 进行验证。
#    - 难点：如何设计合理的 JSON 结构，既能满足工具调用的需求，又便于后续处理。
#
# 3. “引导-校验-重试”闭环 (Guiding-Validation-Retry Loop):
#    - 本质：通过反复迭代，让 LLM 逐渐逼近正确的输出格式。
#    - 核心：在每次迭代中，使用不同的提示词引导模型，并结合验证逻辑确保输出符合预期。
#    - 难点：如何设计有效的验证逻辑，既能发现错误，又不会引入新的错误。
# 4. 工具选择 (Tool Selection):
#    - 本质：根据用户意图和工具描述，选择最合适的工具。
#    - 核心：使用 Prompt Engineering 构建清晰、具体的工具选择描述。
#    - 难点：如何让 LLM 准确理解用户意图，并输出正确的工具调用指令。
#【扩展说明】关于Controlled Decoding：
# 某些模型服务商直接提供JSON Mode的输出，其原理是模型推理出所有token概率后，系统根据Schema编译的语法规则，直接屏蔽不可能组成合法JSON的token，从而输出合法的JSON。
# ==============================================================================

# 为相关工具定义对应的参数模型
class WebSearchParams(BaseModel):
    query: str = Field(description="用于网络搜索的关键词。")

class SearchArxivParams(BaseModel):
    query: str = Field(description="用于在 Arxiv.org 上搜索的论文标题或关键词。")

# 为相关工具定义对应的调用模型，并将工具与参数模型绑定
class WebSearchCall(BaseModel):
    tool_name: Literal["web_search"]
    parameters: WebSearchParams

class SearchArxivCall(BaseModel):
    tool_name: Literal["search_arxiv_paper"]
    parameters: SearchArxivParams

ToolCall = Union[WebSearchCall, SearchArxivCall]

# 构建提示词，包含清晰的指令、工具定义和示例
def build_prompt(user_request: str) -> str:
    return dedent(f"""
        你的任务是根据用户的请求，从可用工具列表中选择最合适的工具，并以严格的 JSON 格式输出调用信息。

        # 可用工具:
        - `web_search(query: str)`: 当需要搜索通用信息、新闻或非学术性内容时使用。
        - `search_arxiv_paper(query: str)`: 当需要搜索学术论文，特别是来自 Arxiv.org 的论文时使用。

        # 输出格式要求:
        你必须严格按照以下 JSON 结构输出，不要包含任何额外的自然语言解释。

        {{
        "tool_name": "工具名称",
        "parameters": {{
            "参数名": "参数值"
        }}
        }}

        # 示例:
        用户请求: "最近AI领域有什么好玩的新闻？"
        你的输出:
        {{
        "tool_name": "web_search",
        "parameters": {{
            "query": "AI领域最新新闻"
        }}
        }}

        # 用户请求:
        "{user_request}"

        # 你的输出:
    """)

# 实现”引导-校验-重试”的闭环，让工具调用代码更加健壮
def get_structured_output(user_request: str, max_retries: int = 2):
    client = _get_client()
    messages = [{'role': 'user', 'content': build_prompt(user_request)}]
    adapter = TypeAdapter(ToolCall)
    
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="qwen-plus", messages=messages, temperature=0
        )
        raw_output = response.choices[0].message.content
        
        try:
            # 清理 Markdown 标记
            clean_json = raw_output.strip()
            if clean_json.startswith('```json'):
                clean_json = clean_json[7:]
            if clean_json.startswith('```'):
                clean_json = clean_json[3:]
            if clean_json.endswith('```'):
                clean_json = clean_json[:-3]
            
            data = json.loads(clean_json)
            validated_data = adapter.validate_python(data)
            return validated_data.model_dump()
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"尝试 {attempt+1} 失败: {e}. 重试中...")
            messages.extend([
                {'role': 'assistant', 'content': raw_output},
                {'role': 'user', 'content': f"格式错误: {e}，请严格按照JSON格式重新输出"}
            ])
    return None

def step2_intent_recognition():
    print("\n" + "="*20 + " STEP 2: 意图识别与工具选择 " + "="*20)
    user_request = "帮我找一下那篇经典的 Transformer 论文，标题是 'Attention Is All You Need'"
    print(f"用户请求: {user_request}")
    
    result = get_structured_output(user_request)
    if result:
        print("LLM 意图识别结果 (JSON):")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("意图识别失败。")


# ==============================================================================
# Step 3: Function Calling (OpenAI Native)
# 【原理】使用原生 Function Calling 机制简化 Agent 开发。
# 
# 1. 自动注入: 我们只需定义 `tools` 列表，API 会自动将其注入到系统提示词中。
# 2. 意图识别与参数生成: API 服务端内部完成了"思考"过程，直接返回 `tool_calls`。
# 3. 参数校验: 
#    - 服务端会尽力保证生成的 `arguments` 符合 JSON Schema。
#    - 客户端仍需进行基础的 `json.loads` 解析检查。
#    - 在生产环境中，建议此处依然结合 Pydantic (如 Step 2) 对 `function_args` 进行二次严格校验。
# ==============================================================================
def step3_function_calling():
    print("\n" + "="*20 + " STEP 3: Function Calling " + "="*20)
    
    # 1. 定义工具列表
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_arxiv_paper",
                "description": "在 Arxiv.org 上搜索学术论文",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "论文的标题或关键词"},
                    },
                    "required": ["query"],
                },
            }
        }
    ]
    
    request_content = "帮我找一下那篇经典的 Transformer 论文 'Attention Is All You Need'"
    print(f"用户请求: {request_content}")
    messages = [{"role": "user", "content": request_content}]
    
    # 2. 进行意图识别，并决定调用哪个工具
    client = _get_client()
    response = client.chat.completions.create(
        model="qwen-plus", messages=messages, tools=tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    
    # 3. 检查模型是否决定调用工具，并执行工具调用
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        
        # [校验点] 这里进行了基础的 JSON 解析校验
        function_args = json.loads(tool_call.function.arguments)
        print(f"模型决定调用工具: `{function_name}`")
        print(f"参数: {function_args}")
        
        # 模拟执行工具，并返回结果
        tool_result = json.dumps({
            "paper_id": "1706.03762", 
            "url": "https://arxiv.org/abs/1706.03762", 
            "title": "Attention Is All You Need"
        })
        print(f"工具执行结果: {tool_result}")
        
        # 4. 将工具执行结果传回模型
        messages.append(response_message)
        messages.append(
            {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": tool_result}
        )
        
        final_response = client.chat.completions.create(model="qwen-plus", messages=messages)
        print("\n模型的最终回复:")
        print(final_response.choices[0].message.content)


# ==============================================================================
# Step 4: React Agent (使用 AgentScope 框架)
# 【原理】引入 Agent 框架实现自动化的 ReAct (Reason-Act-Observe) 循环。
# 
# 1. 工具定义自动解析: 
#    - 框架通过解析 `search_arxiv_paper_tool` 函数的 docstring 和类型提示，
#    - 自动生成了类似 Step 3 中的 JSON Schema 工具描述。
# 
# 2. ReAct 循环自动化: 
#    - 框架自动维护 Prompt 中的历史上下文 (Think -> Act -> Observe)。
#    - 如果 LLM 发现参数缺失，会在 Think 阶段自我修正或反问，而不需要手动编写复杂的 `while` 循环。
# ==============================================================================
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg, TextBlock
from agentscope.formatter import DashScopeChatFormatter
def search_arxiv_paper_tool(query: str) -> "ToolResponse":
    """在 Arxiv.org 上搜索学术论文。
    
    Args:
        query (str): 搜索关键词
    """
    print(f"--- [工具执行中] 正在 Arxiv 搜索: {query} ---")
    # 此处模拟搜索结果
    paper_url = "https://arxiv.org/abs/1706.03762"
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"已成功找到论文 '{query}'，你可以在这里访问：{paper_url}",
            )
        ]
    )

async def step4_react_agent():
    print("\n" + "="*20 + " STEP 4: React Agent (AgentScope) " + "="*20)
    # 2. 将工具函数注册到工具箱(Toolkit)中
    toolkit = Toolkit()
    toolkit.register_tool_function(search_arxiv_paper_tool)
    
    # 3. 创建一个 ReActAgent 并为其配备工具箱
    agent = ReActAgent(
        name="Course Research Agent",
        sys_prompt="你是一个课程研究助理，擅长帮人搜集和整理学习资料。",
        model=DashScopeChatModel(
            model_name="qwen-plus", 
            api_key=os.environ.get("DASHSCOPE_API_KEY")
        ),
        toolkit=toolkit,
        formatter=DashScopeChatFormatter()
    )
    
    # 4. 向 Agent 发送消息，它会自动完成所有步骤
    user_request = "帮我找一下那篇经典的 Transformer 论文 'Attention Is All You Need'"
    msg = Msg(name="user", content=user_request, role="user")
    
    print(f"用户请求: {user_request}\n")
    await agent(msg)


# ==============================================================================
# Step 5: MCP (Model Context Protocol)
# 核心思想：将定义工具的职责由消费方转移到提供方。
# 注意：本示例使用 AgentScope v1.0 的 HttpStatelessClient 连接远程 MCP 服务。
# ==============================================================================
async def step5_mcp_example():
    print("\n" + "="*20 + " STEP 5: MCP (Model Context Protocol) " + "="*20)
    # 1. 配置 MCP 客户端 (连接阿里云 DashScope 联网搜索服务)
    web_search_client = HttpStatelessClient(
        name="web_search_service",  # 为客户端指定唯一名称
        transport="sse",            # 指定传输协议
        url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse",
        headers={"Authorization": "Bearer " + os.environ.get("DASHSCOPE_API_KEY")},
    )
    
    # 2. 将 MCP 客户端注册到工具箱
    #    Agent在启动时会自动通过客户端“发现”远程服务提供的所有工具
    toolkit = Toolkit()
    await toolkit.register_mcp_client(web_search_client)
    # 3. 创建 Agent，并配备包含 MCP 工具的工具箱
    agent = ReActAgent(
        name="Research Assistant Agent",
        sys_prompt="你是一个课程研究助理，擅长使用工具搜集和整理最新的教学素材。",
        model=DashScopeChatModel(
            model_name="qwen-plus", api_key=os.environ.get("DASHSCOPE_API_KEY")
        ),
        toolkit=toolkit,
        formatter=DashScopeChatFormatter()
    )
    
    # 4. 提出一个需要远程工具才能回答的问题
    user_request = "我正在为'AI与大模型原理'课程搜集素材，需要一个调用外部实时数据的例子，比如帮我搜索一下最近关于'大型语言模型'的最新进展。注意当前时间是2025年12月18日。"
    msg = Msg(name="user", content=user_request, role="user")
    
    print(f"用户请求: {user_request}\n")    
    # 4. 运行 Agent
    await agent(msg)


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 按顺序执行各个步骤
    
    # 1. 基础原理
    step1_hardcoded_tool()
    
    # 2. 意图识别
    step2_intent_recognition()
    
    # 3. 原生 Function Calling
    step3_function_calling()
    
    # 4. AgentScope ReAct Agent
    try:
        asyncio.run(step4_react_agent())
    except Exception as e:
        print(f"Step 4 运行出错: {e}")

    # 5. MCP 演示
    try:
        # 由于 Step 5 包含 async 代码 (AgentScope 的某些 MCP 操作可能涉及 async)
        # 即使 ReActAgent 本身是同步的，HttpStatelessClient 可能需要异步上下文
        asyncio.run(step5_mcp_example())
    except Exception as e:
        print(f"Step 5 运行出错: {e}")
