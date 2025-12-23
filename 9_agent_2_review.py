#说明：LLM普遍存在幻觉（Hallucinate)，本文件演示如何使用Reflection Agent来解决这个问题。
# 反思的主要模式
# 1. Self Review：LLM自己发现自己的错误，并进行修正。
# 1.1 单步指令式：反思和生成在一个指令中。
# 1.2 多步指令式：先生成，再审查。由于需要多次迭代，所以成本较高。普遍采用Context Cache来优化(将原始文档作为共享内容缓存，后续检测到前缀相同，则直接返回缓存内容)
# 本模式的问题：只能基于内容检查，无法精确识别错误类型。
# 2. External Feedback：生成->与外部工具交互->获得外部反馈->修正问题

"""
使用 AgentScope 实现外部反馈的反思机制（简化版）
场景：润色 Jupyter Notebook 课程，通过代码解释器验证代码正确性
"""

import asyncio
import os
from textwrap import dedent

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_python_code

from config.load_key import load_key

# 加载环境变量
load_key()


# ============================================================
# 创建写作 Agent（装备代码解释器工具）
# ============================================================

def create_writer_agent() -> ReActAgent:
    """创建负责润色课程内容的写作 Agent"""
    
    # 创建工具包并注册内置的代码执行工具
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    
    writer = ReActAgent(
        name="Writer",
        sys_prompt=dedent("""你是一位技术课程作家，负责润色 Jupyter Notebook 课程。

            任务要求：
            1. 优化文本表达，使其更流畅生动
            2. **绝对不要修改代码中的变量名、函数名、参数名**
            3. 润色后，使用 execute_python_code 工具验证所有代码块
            4. 如果代码执行失败，检查是否意外修改了代码并修正

            记住：只改文案，不改代码逻辑！
        """),
        model=DashScopeChatModel(
            model_name="qwen-plus",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            stream=False,
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=10,  # 最多迭代 10轮（示例配置，实际应根据任务复杂度和成本预算调整）
    )
    
    return writer


# ============================================================
# 主流程
# ============================================================

async def main():
    """主函数 - 演示外部反馈的反思机制"""
    
    # 原始课程内容（包含"不规范"的变量名 usr_id）
    original_notebook = """
# Python 函数示例课程

本节学习如何定义和调用函数。

## 定义函数

def get_user_data(usr_id: str):
    return f"Data for {user_id}"

## 调用函数

print(get_user_data(usr_id="u-123"))

## 小结

你学会了定义和调用 Python 函数。
"""
    
    print("="*60)
    print("原始课程内容：")
    print("="*60)
    print(original_notebook)
    
    # 创建 Agent
    writer = create_writer_agent()
    
    # 发送润色请求
    user_msg = Msg(
        name="user",
        content=dedent(f"""请润色以下课程内容，要求：

            {original_notebook}

            1. 让文案更生动易懂
            2. **不要修改代码中的变量名、函数名**
            3. 润色后用 execute_python_code 验证代码能否运行
            4. 如果报错，说明你可能改错了代码，请修正
        """),
        role="user"
    )
    
    print("\n" + "="*60)
    print("Agent 开始工作...")
    print("="*60)
    
    # Agent 自动进行：润色 → 验证 → 修正 → 再验证
    result = await writer(user_msg)
    
    print("\n" + "="*60)
    print("最终输出：")
    print("="*60)
    print(result.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())


# 扩展场景1：优化科研论文中的Matplotlib图标
# 问题：科研论文中的Matplotlib图标存在一些问题，比如颜色不合适、字体不合适、标签不合适等。
# 解决方案：生成-渲染-检查-调整 的闭环。调用代码解释器工具执行Matplotlib代码，并返回生成图片文件。Agent直接视觉检测实际渲染图表发现问题，然后调整代码参数等，再重新执行，直到满意为止。

# 扩展场景2：复杂计算题答案核对
# 问题：例如量子力学中计算题计算可能出错
# 解决方案：生成-计算-检查-调整 的闭环。识别要计算的表达式，用代码解释器或计算器工具执行，获取结果后进行分析并输出反馈，让Agent修正最终答案

# 扩展场景3：结构化输出校验
# 问题：要求生成符合特定Json Schema的输出
# 解决方案：生成-校验-调整 的闭环。生成后，系统使用Pydantic等库校验得到详细错误信息，Agent根据错误信息进行调整，直到符合要求为止。
