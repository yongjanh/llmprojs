# ==============================================================================
# 说明：本文件演示 Agent 的自主规划与执行 (Self-Planning & Execution)
#
# 【背景】
# 复杂的长链条任务往往无法通过一次简单的 LLM 调用完成。
# Agent 需要具备"思考-规划-执行-调整"的元认知能力。
#
# 【解决方案】
# 引入 PlanNotebook (计划本) 机制，让 Agent 像人类项目经理一样工作。
#
# 【本文件演示模式】
# 1. Self-Planning (自主规划): 
#    - 核心机制：Create Plan -> Execute Subtask -> Update State -> Revise Plan
#    - 关键能力：任务拆解、进度追踪、遇阻调整（动态规划）。
#    - 场景：开放式调研、复杂问题排查、长文本生成。
#
# 2. Tool Creation (自主工具创造):
#    - 核心机制：Code Execution -> Register Tool -> Immediate Use
#    - 关键能力：当现有工具不足时，Agent 能够编写代码创造新工具来解决问题。
#    - 场景：临时数据处理、特定格式转换、数学计算。
# ==============================================================================

import asyncio
import os
import sys
from io import StringIO
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg, TextBlock
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, ToolResponse
from agentscope.plan import PlanNotebook
from agentscope.memory import InMemoryMemory
from config.load_key import load_key

# 加载环境变量
load_key()

# ==============================================================================
# 示例1：Agent自主规划 (Self-Planning)
# 场景：Python课程前期调研
# 
# 【原理】
# Agent 配备了一个 PlanNotebook。
# 它不直接回答问题，而是先调用 create_plan 工具写下计划。
# 然后按顺序执行子任务，每完成一步更新状态。
# 如果遇到困难（如工具报错），它会像人类一样思考替代方案，并调用 revise_plan 修改计划。
# ==============================================================================

# 模拟业务工具
async def analyze_competitor_course(url: str) -> ToolResponse:
    """分析竞品课程页面的大纲"""
    # 模拟因网站改版导致解析失败
    return ToolResponse(content=[
        TextBlock(type="text", text=f"❌ 错误：因 {url} 网站布局更新，无法解析课程大纲。")
    ])

async def search_industry_demand(topic: str) -> ToolResponse:
    """查询行业的技能需求"""
    return ToolResponse(content=[
        TextBlock(type="text", text=f"✅ 报告：关于“{topic}”的行业需求分析已完成。")
    ])

async def google_search(query: str) -> ToolResponse:
    """谷歌网页搜索"""
    if "syllabus" in query:
        return ToolResponse(content=[
            TextBlock(type="text", text="搜索结果：找到了'Python入门课程'的大纲PDF，地址 a.com/syllabus.pdf")
        ])
    return ToolResponse(content=[TextBlock(type="text", text="未找到相关信息")])

async def extract_text_from_pdf(url: str) -> ToolResponse:
    """从PDF链接中提取文本"""
    return ToolResponse(content=[
        TextBlock(type="text", text=f"✅ 已从 {url} 提取大纲文本：1. 变量与数据类型... 2. ...")
    ])


# 用于监控计划变化的钩子函数
plan_snapshots = []

def capture_plan_snapshot(notebook, plan):
    """捕获计划快照"""
    if plan:
        plan_snapshots.append({
            "name": plan.name,
            "description": plan.description,
            "state": plan.state,
            "subtasks": [
                {
                    "name": st.name,
                    "state": st.state,
                    "outcome": st.outcome
                }
                for st in plan.subtasks
            ]
        })


async def main_planning():
    load_key()
    
    print("=" * 60)
    print("🤖 Agent自主规划演示")
    print("=" * 60)
    
    # 创建PlanNotebook并注册钩子
    plan_notebook = PlanNotebook()
    plan_notebook.register_plan_change_hook("capture", capture_plan_snapshot)
    
    # 创建工具箱
    toolkit = Toolkit()
    toolkit.register_tool_function(analyze_competitor_course)
    toolkit.register_tool_function(search_industry_demand)
    toolkit.register_tool_function(google_search)
    toolkit.register_tool_function(extract_text_from_pdf)
    
    # 创建Agent
    agent = ReActAgent(
        name="CourseResearcherAgent",
        sys_prompt=(
            "你是课程调研助手。遇到复杂任务时：\n"
            "1. 用create_plan创建计划\n"
            "2. 逐步执行，用finish_subtask标记完成\n"
            "3. 遇到问题灵活调整，例如使用google_search寻找替代方案\n"
            "4. 完成后用finish_plan结束\n"
            "5. 如果已经有简单的结果，就直接使用并完成相关任务，不要重新调研\n"
        ),
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit,
        plan_notebook=plan_notebook,
        max_iters=30
    )
    
    # 用户请求
    print("\n💬 用户: 请帮我完成一门新的 Python 入门课程的前期调研。\n")
    print("-" * 60)
    
    await agent(Msg("user", "请帮我完成一门新的 Python 入门课程的前期调研，竞品是 some-site.com 的课程。", "user"))
    
    # 确保捕获最终状态（去重：仅在状态与最后一个快照不同时才补充）
    current_plan = plan_notebook.current_plan
    if current_plan:
        # 如果没有快照，或者状态发生了变化，则捕获
        needs_capture = (
            not plan_snapshots or 
            plan_snapshots[-1]["state"] != current_plan.state
        )
        if needs_capture:
            capture_plan_snapshot(plan_notebook, current_plan)
    
    # 显示结果（使用 plan_snapshots，更优雅的观察者模式）
    print("\n" + "=" * 60)
    print("📊 执行结果")
    print("=" * 60)
    
    # 使用 plan_snapshots（记录了完整的状态演变历史）
    if plan_snapshots:
        final_plan = plan_snapshots[-1]  # 获取最后一个快照
        
        # 统计完成的子任务
        finished = sum(1 for st in final_plan["subtasks"] if st["state"] == "finished")
        
        print(f"\n✅ 计划: {final_plan['name']}")
        print(f"📊 进度: {finished}/{len(final_plan['subtasks'])}")
        print(f"🎯 状态: {final_plan['state']}\n")
        
        print("子任务详情:")
        for i, subtask in enumerate(final_plan["subtasks"], 1):
            icon = "✅" if subtask["state"] == "finished" else "⏳"
            outcome_info = f" (成果: {subtask['outcome']})" if subtask.get("outcome") else ""
            print(f"  {icon} {i}. {subtask['name']}{outcome_info}")
    else:
        print("无可用计划。")


# ==============================================================================
# 示例2：自主创建工具 (Autonomous Tool Creation)
# 场景：动态计算需求 (ToolMaker)
#
# 【原理】
# Agent 被赋予了 "代码执行" (code_exec) 权限。
# 当面临没有现成工具可用的任务（如计算阶乘）时，
# Agent 会推理出计算逻辑，编写 Python 函数，并将其注册到自己的工具箱中。
# 随后，它立即调用这个新生成的工具来解决用户的问题。
# 这种 "Code as Tool" 的模式赋予了 Agent 无限的扩展能力。
# ==============================================================================

# 全局工具箱 (用于 ToolMaker 演示)
toolkit_maker = None


async def code_exec(code: str) -> ToolResponse:
    """代码解释器 - 用于创建和注册新工具"""
    global toolkit_maker
    
    namespace = {
        'ToolResponse': ToolResponse,
        'TextBlock': TextBlock,
        'asyncio': asyncio,
        'agent_toolkit': toolkit_maker,
        'math': __import__('math'),
    }
    
    stdout, sys.stdout = sys.stdout, StringIO()
    
    try:
        exec(code, namespace)
        output = sys.stdout.getvalue()
        sys.stdout = stdout
        return ToolResponse(content=[TextBlock(
            type="text", 
            text=output or "✅ 执行成功"
        )])
    except Exception as e:
        sys.stdout = stdout
        return ToolResponse(content=[TextBlock(
            type="text",
            text=f"❌ 错误: {e}"
        )])


async def add(a: float, b: float) -> ToolResponse:
    """加法工具"""
    return ToolResponse(content=[TextBlock(
        type="text", 
        text=f"{a} + {b} = {a + b}"
    )])


async def main_tool_maker():
    load_key()
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("❌ 请设置 DASHSCOPE_API_KEY")
        return
    
    global toolkit_maker
    toolkit_maker = Toolkit()
    toolkit_maker.register_tool_function(add)
    toolkit_maker.register_tool_function(code_exec)
    
    agent = ReActAgent(
        name="ToolMaker",
        sys_prompt=(
            "你可以通过 code_exec 创建新工具。\n"
            "模板:\n"
            "async def tool_name(param: type) -> ToolResponse:\n"
            "    '''描述'''\n"
            "    result = ...\n"
            "    return ToolResponse(content=[TextBlock(type='text', text=f'{result}')])\n"
            "agent_toolkit.register_tool_function(tool_name)\n"
            "print('✅ 已注册 tool_name')"
        ),
        model=DashScopeChatModel(
            model_name="qwen-plus",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        ),
        formatter=DashScopeChatFormatter(),
        toolkit=toolkit_maker,
        memory=InMemoryMemory(),
    )
    
    print("=" * 60)
    print("🚀 Agent 自主创建工具演示")
    print("=" * 60)
    
    # 使用现有工具
    print("\n▶️ 场景1: 使用现有工具")
    await agent(Msg("user", "计算 30 + 45", "user"))
    
    # 创建新工具
    print("\n▶️ 场景2: 创建阶乘工具")
    await agent(Msg("user", "创建 factorial 工具计算阶乘", "user"))
    
    # 使用新工具
    print("\n▶️ 场景3: 使用新工具")
    await agent(Msg("user", "用 factorial 计算 5 的阶乘", "user"))
    
    # 显示工具箱
    print("\n📦 最终工具箱:")
    for i, s in enumerate(toolkit_maker.get_json_schemas(), 1):
        print(f"{i}. {s['function']['name']}")


if __name__ == "__main__":
    # 示例1：Agent自主规划演示
    # try:
    #     asyncio.run(main_planning())
    # except Exception as e:
    #     print(f"自主规划演示出错: {e}")

    # 示例2：自主创建工具演示
    try:
        asyncio.run(main_tool_maker())
    except Exception as e:
        print(f"自主创建工具演示出错: {e}")

