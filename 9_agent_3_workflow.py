# ==============================================================================
# 说明：本文件演示如何使用 AgentScope 构建不同模式的工作流 (Workflow)
#
# 【背景】
# 针对复杂任务，单一 Agent 往往存在以下问题：
# 1. 注意力遗忘：随着上下文变长，模型容易忽略早期指令。
# 2. 错误累积：一步错，步步错。
# 3. 缺乏结构：无法处理需要并行、分支或全局视角的任务。
#
# 【解决方案】
# 使用工作流（Workflow）编排多个 Agent 协作。
#
# 【本文件演示模式】
# 1. Pipeline (流水线): 
#    - 优势：结构简单，逻辑清晰，适合标准化工序。
#    - 场景：数据预处理、多阶段审批、内容生成流水线。
#
# 2. Branching (分支选择): 
#    - 优势：灵活处理多样化请求，节省计算资源（只运行必要路径）。
#    - 场景：客户服务路由、故障诊断决策树、个性化推荐。
#
# 3. Parallel (并行执行): 
#    - 优势：大幅缩短总耗时，提供多维度视角。
#    - 场景：代码审查、法律合同审计、多模态信息综合。
#
# 4. Mixture-of-Agents (MoA, 混合专家): 
#    - 优势：利用不同模型/角色的特长，通过"协作性"提升最终质量，克服单一模型偏见。
#    - 场景：复杂推理、创意写作、高精度内容生成。
#
# 5. Human-in-the-Loop (HITL, 人机协作): 
#    - 优势：引入人类判断处理模糊性，保障安全与合规。
#    - 场景：高风险决策（医疗/金融）、内容审核、关键代码部署。
# ==============================================================================

import asyncio
import os
from typing import Literal, List
from textwrap import dedent

from pydantic import BaseModel, Field

# AgentScope imports
from agentscope.message import Msg, TextBlock
from agentscope.pipeline import sequential_pipeline, fanout_pipeline
from agentscope.agent import ReActAgent, UserAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.tool import Toolkit, ToolResponse

# Local imports
from config.load_key import load_key
from chatbot.agent import create_agent, disable_console_output

# 加载环境变量 (API Key)
load_key()

# ==============================================================================
# 模式1：流水线 (Pipeline)
# 场景：课程快速检查流程 (提取代码 -> 验证代码 -> 生成报告)
# 
# 【原理】最基础的工作流，上一个 Agent 的输出直接作为下一个 Agent 的输入。
# ==============================================================================
async def run_pattern_1_pipeline() -> None:
    print("\n" + "="*20 + " Pattern 1: Pipeline (流水线) " + "="*20)
    
    # 节点A：代码提取 Agent
    code_extractor = create_agent(
        name="代码提取器",
        sys_prompt=(
            "你是代码提取专家。请从用户提供的课程文本中，精确地提取出所有 Python 代码块。"
            "只输出代码，不要有任何其他解释。"
        ),
        model_name="qwen-plus", # 使用较快模型
        multi_agent=True,
    )

    # 节点B：代码验证 Agent (可调用外部工具)
    # 注意：为了演示简单，这里暂未配置真实的代码执行工具，而是让模型模拟执行
    # 如果需要真实执行，需配置 execute_python_code 工具
    code_validator = create_agent(
        name="代码验证器",
        sys_prompt=(
            "你是代码执行与验证专家。你将接收到代码文本。请仔细检查代码逻辑（或模拟执行）。"
            "报告代码是否能成功运行，如果不能，请指出错误。"
        ),
        model_name="qwen-plus",
        multi_agent=True,
    )

    # 节点C：报告生成 Agent
    report_generator = create_agent(
        name="报告生成器",
        sys_prompt=(
            "你是审阅报告撰写助理。根据上一步的代码验证结果，为课程设计师生成一份简洁明了的检查报告。"
        ),
        model_name="qwen-max",
        multi_agent=True,
    )

    agents = [code_extractor, code_validator, report_generator]
    # 禁用中间 Agent 的控制台输出，保持界面整洁（可选）
    disable_console_output(agents)

    course_draft = (
        "这是我们的新课程。第一部分是`print('Hello, World!')`。"
        "第二部分是一个有问题的代码`x = 1 / 0`。"
    )
    
    print(f"输入文本: {course_draft}\n")
    
    # 执行顺序流水线
    result = await sequential_pipeline(
        agents=agents,
        msg=Msg("user", course_draft, "user"),
    )

    print("-" * 30)
    print("流水线最终输出：")
    print(result.content)
    print("-" * 30)


# ==============================================================================
# 模式2：分支选择 (Branching)
# 场景：课程审阅任务分发 (路由 -> 代码检查/风格润色/全面评审)
#
# 【原理】引入 Router (路由) Agent，根据用户意图分类，动态决定后续执行路径。
# ==============================================================================

class RouteChoice(BaseModel):
    choice: Literal["code_check", "style_guide", "full_review", None] = Field(
        description="根据用户意图选择分支：code_check/style_guide/full_review/None"
    )
    extra: str | None = Field(default=None, description="对任务的简要说明")

async def branch_code_check(user_msg: Msg) -> Msg:
    agent = create_agent(
        name="代码快检专家",
        sys_prompt="你是代码快检专家。根据用户需求，快速验证课程中的代码片段是否能运行。",
        model_name="qwen-plus",
        multi_agent=True,
    )
    disable_console_output([agent])
    return await agent(user_msg)

async def branch_style_guide(user_msg: Msg) -> Msg:
    agent = create_agent(
        name="语言润色专家",
        sys_prompt="你是语言润色专家。请根据公司风格指南，改写和润色用户提供的课程文本。",
        model_name="qwen-max",
        multi_agent=True,
    )
    disable_console_output([agent])
    return await agent(user_msg)

async def branch_full_review(user_msg: Msg) -> Msg:
    agent = create_agent(
        name="首席评审",
        sys_prompt="你是首席评审。告知用户，你将启动一个包含代码、事实和教学法在内的全面评审流程。",
        model_name="qwen-plus", 
        multi_agent=True,
    )
    disable_console_output([agent])
    return await agent(user_msg)

async def run_pattern_2_branching() -> None:
    print("\n" + "="*20 + " Pattern 2: Branching (分支选择) " + "="*20)
    
    router = create_agent(
        name="审阅任务分发员",
        sys_prompt=(
            "你是课程审阅任务的分发员，根据用户输入选择分支：\n"
            "- 如果只是想检查代码，输出 code_check\n"
            "- 如果是想润色文笔，输出 style_guide\n"
            "- 如果是需要完整、全面的评审，输出 full_review\n"
            "仅通过结构化输出来表达你的选择，不要正文回答。"
        ),
        model_name="qwen-plus",
        multi_agent=False, # Router 通常作为单体交互
    )

    user_text = "这篇课程写的差不多了，帮我全面检查一下，特别是代码和难度。"
    print(f"用户请求: {user_text}")

    # 1. 路由决策
    res = await router(
        Msg("user", user_text, "user"),
        structured_model=RouteChoice,
    )
    
    # 兼容性处理：检查是否成功解析为结构化数据
    if hasattr(res, 'parsed') and res.parsed:
        choice = res.parsed.choice
    elif isinstance(res.content, dict):
        choice = res.content.get("choice")
    elif res.metadata and "choice" in res.metadata:
        choice = res.metadata.get("choice")
    else:
        # Fallback provided in case parsing fails slightly differently in versions
        try:
             # 尝试从 json 字符串解析
             import json
             data = json.loads(res.content)
             choice = data.get("choice")
        except:
             choice = "full_review" # 默认 fallback

    print(f"路由选择: {choice}")

    # 2. 分支执行
    if choice == "code_check":
        out = await branch_code_check(Msg("user", user_text, "user"))
    elif choice == "style_guide":
        out = await branch_style_guide(Msg("user", user_text, "user"))
    elif choice == "full_review":
        out = await branch_full_review(Msg("user", user_text, "user"))
    else:
        # 默认走全面评审
        out = await branch_full_review(Msg("user", user_text, "user"))

    print("-" * 30)
    print("分支执行结果：")
    print(out.content)
    print("-" * 30)


# ==============================================================================
# 模式3：并行执行 (Parallel Execution)
# 场景：课程完整审阅 (代码/事实/教学法/风格 同时进行 -> 汇总)
#
# 【原理】让多个任务独立的 Agent 同时运行，最后由一个 Aggregator 汇总结果。
# ==============================================================================
async def run_pattern_3_parallel() -> None:
    print("\n" + "="*20 + " Pattern 3: Parallel Execution (并行执行) " + "="*20)
    
    # 四个独立子任务的“专家”Agent
    code_checker = create_agent(
        name="代码检查员",
        sys_prompt="验证课程中的代码是否正确无误，并给出修复建议。",
        model_name="qwen-plus",
        multi_agent=True,
    )
    fact_checker = create_agent(
        name="事实核查员",
        sys_prompt="核对课程中的技术概念、函数解释是否准确，引用是否规范。",
        model_name="qwen-plus",
        multi_agent=True,
    )
    pedagogy_evaluator = create_agent(
        name="教学法评估师",
        sys_prompt="评估课程的难度曲线、案例趣味性和练习有效性。",
        model_name="qwen-plus", # 统一使用 plus
        multi_agent=True,
    )
    style_editor = create_agent(
        name="风格编辑",
        sys_prompt="根据公司风格指南，检查并报告语言风格、术语一致性问题。",
        model_name="qwen-plus",
        multi_agent=True,
    )

    experts = [code_checker, fact_checker, pedagogy_evaluator, style_editor]
    disable_console_output(experts)

    course_content = "这是我们新开发的 Python 数据分析入门课... (此处省略长文)"
    print(f"正在并行审阅课程内容...")

    # 并行执行
    msgs = await fanout_pipeline(
        agents=experts,
        msg=Msg("user", course_content, "user"),
        enable_gather=True, # 收集所有结果
    )

    # 汇总 Agent
    summarizer = create_agent(
        name="总编辑",
        sys_prompt="将来自多位专家的审阅意见汇总成一份结构清晰、条理分明的总审阅报告。",
        model_name="qwen-max",
        multi_agent=True,
    )
    disable_console_output([summarizer])

    merged_text: List[str] = [m.get_text_content() for m in msgs]
    prompt = "\n\n".join(merged_text)
    
    print("各专家审阅完成，正在汇总...")
    summary = await summarizer(Msg("user", prompt, "user"))

    print("-" * 30)
    print("并行审阅汇总报告：")
    print(summary.content)
    print("-" * 30)


# ==============================================================================
# 模式4：混合专家 (Mixture-of-Agents, MoA)
# 场景：课程核心卖点提炼 (多模型生成 -> 聚合优化)
#
# 【原理】利用多个 LLM (Proposers) 生成候选回答，再由一个 LLM (Aggregator) 融合生成最佳结果。
# ==============================================================================
async def run_pattern_4_moa() -> None:
    print("\n" + "="*20 + " Pattern 4: MoA (混合专家) " + "="*20)
    
    # 使用不同的模型角色（实际代码中可以用不同的 system prompt 或 temperature 模拟多样性）
    
    proposer1 = create_agent(
        name="Proposer-Marketing",
        sys_prompt="你是一个资深市场营销专家。请从市场推广和用户痛点的角度，提炼课程核心卖点。",
        model_name="qwen-plus",
        multi_agent=True,
    )
    proposer2 = create_agent(
        name="Proposer-Tech",
        sys_prompt="你是一个资深技术专家。请从技术深度、实战价值和先进性的角度，提炼课程核心卖点。",
        model_name="qwen-plus",
        multi_agent=True,
    )
    proposer3 = create_agent(
        name="Proposer-Education",
        sys_prompt="你是一个教育心理学家。请从学习体验、成长路径和教学设计的角度，提炼课程核心卖点。",
        model_name="qwen-plus",
        multi_agent=True,
    )

    proposers = [proposer1, proposer2, proposer3]
    disable_console_output(proposers)

    task = "这是一门新的'面向Web开发者的AI大模型应用'课程，请为其提炼核心卖点和宣传文案。"
    print(f"任务: {task}")

    # 并行生成提案
    msgs = await fanout_pipeline(
        agents=proposers,
        msg=Msg("user", task, "user"),
        enable_gather=True,
    )

    # 聚合器(Aggregator)
    aggregator = create_agent(
        name="聚合器",
        sys_prompt=(
            "你的任务是综合多个专家对同一问题的回答。"
            "请批判性地评估这些回答，识别其中的优点，融合生成一个高质量、全面且极具吸引力的最终方案。"
        ),
        model_name="qwen-max",
        multi_agent=True,
    )
    disable_console_output([aggregator])

    merged = "\n\n".join([
        f"[{m.name}] 的回答：\n{m.get_text_content()}"
        for i, m in enumerate(msgs)
    ])
    
    print("正在聚合专家意见...")
    final = await aggregator(Msg("user", merged, "user"))

    print("-" * 30)
    print("MoA 最终方案：")
    print(final.content)
    print("-" * 30)

# ==============================================================================
# 模式4.1：进阶 Multi-Layer MoA (多层混合专家)
# 场景：高难度营销方案生成 (Layer 1 提案 -> Layer 2 优化 -> Layer 3 终极聚合)
#
# 【原理】
# 类似深度神经网络，通过叠加多层 MoA 结构来进一步提升复杂任务的表现。
# Layer 1: 生成多样化初稿
# Layer 2: 基于初稿进行批判性改进 (Refinement)
# Layer 3: 最终的一致性聚合
# ==============================================================================
async def run_pattern_4_1_multi_layer_moa() -> None:
    print("\n" + "="*20 + " Pattern 4.1: Multi-Layer MoA (多层混合专家) " + "="*20)
    
    # 通用的聚合提示词
    aggregate_prompt = (
        "你的任务是综合多个大语言模型对同一问题的回答。"
        "请批判性地评估这些回答，识别其中的优点和不足，"
        "然后融合这些信息，生成一个高质量、准确、全面的最终回答。"
    )

    # Layer 1: 初始提议者层 (3个不同角色的模型)
    # 模拟不同模型：这里为了演示使用相同模型名，实际生产中应使用不同模型(如 qwen-max, deepseek, etc.)
    layer1_proposers = [
        create_agent(name="Proposer-L1-1", sys_prompt="你是一个激进的营销策划，擅长制造爆点。", model_name="qwen-plus", multi_agent=True),
        create_agent(name="Proposer-L1-2", sys_prompt="你是一个稳健的产品经理，注重功能落地。", model_name="qwen-plus", multi_agent=True),
        create_agent(name="Proposer-L1-3", sys_prompt="你是一个数据分析师，喜欢用数据说话。", model_name="qwen-plus", multi_agent=True),
    ]
    disable_console_output(layer1_proposers)

    task = "这是一门新的'面向Web开发者的AI大模型应用'课程，请为其提炼核心卖点和宣传文案。"
    print(f"Layer 1 正在并行生成初稿...")
    layer1_outputs = await fanout_pipeline(
        agents=layer1_proposers,
        msg=Msg("user", task, "user"),
        enable_gather=True,
    )

    # Layer 1 聚合器
    layer1_aggregator = create_agent(
        name="Aggregator-L1",
        sys_prompt=aggregate_prompt,
        model_name="qwen-plus",
        multi_agent=True,
    )
    disable_console_output([layer1_aggregator])
    
    layer1_merged = "\n\n".join([f"[{m.name}]:\n{m.get_text_content()}" for i, m in enumerate(layer1_outputs)])
    print(f"Layer 1 聚合中...")
    layer1_result = await layer1_aggregator(Msg("user", layer1_merged, "user"))

    # Layer 2: 优化层 (基于 Layer 1 的结果进行进一步打磨)
    layer2_proposers = [
        create_agent(name="Reviewer-L2-1", sys_prompt="你是资深编辑，请优化文案的流畅度和感染力。", model_name="qwen-plus", multi_agent=True),
        create_agent(name="Reviewer-L2-2", sys_prompt="你是技术总监，请确保技术描述的准确性和专业度。", model_name="qwen-plus", multi_agent=True),
        create_agent(name="Reviewer-L2-3", sys_prompt="你是用户体验专家，请确保方案对用户有吸引力。", model_name="qwen-plus", multi_agent=True),
    ]
    disable_console_output(layer2_proposers)

    layer2_prompt = f"以下是初步生成的营销方案：\n\n{layer1_result.get_text_content()}\n\n请在此基础上提出改进建议或优化版本。"
    print(f"Layer 2 正在并行优化方案...")
    layer2_outputs = await fanout_pipeline(
        agents=layer2_proposers,
        msg=Msg("user", layer2_prompt, "user"),
        enable_gather=True,
    )

    # Layer 3: 最终聚合层
    final_aggregator = create_agent(
        name="Final-Aggregator",
        sys_prompt=aggregate_prompt,
        model_name="qwen-max", # 最终层使用最强模型
        multi_agent=True,
    )
    disable_console_output([final_aggregator])

    layer2_merged = "\n\n".join([f"[{m.name}]:\n{m.get_text_content()}" for i, m in enumerate(layer2_outputs)])
    print(f"Layer 3 最终聚合生成...")
    final_output = await final_aggregator(Msg("user", layer2_merged, "user"))

    print("-" * 30)
    print("Multi-Layer MoA 最终输出：")
    print(final_output.content)
    print("-" * 30)


# ==============================================================================
# 模式5：人机协作 (Human-in-the-Loop, HITL)
# 场景：课程疑难点审核 (AI 建议 -> 人类决策 -> AI 执行)
#
# 【原理】
# 将"人类"视为一个特殊的工具或 Agent。当 AI 遇到低置信度或高风险任务时，
# 挂起当前流程，等待人类输入反馈后，再恢复执行。
# ==============================================================================

# 定义人类介入工具
async def ask_human_decision(question: str) -> ToolResponse:
    """向人类专家征求决策或意见。
    
    Args:
        question (str): 想要请人类确认或补充的具体问题。
    """
    print(f"\n[系统] AI 正在请求人类介入...")
    print(f"[AI 提问] {question}")
    human_expert = UserAgent(name="教学专家")
    reply = await human_expert(
        Msg(
            "assistant",
            question,
            "assistant",
        )
    )
    return ToolResponse(
        content=[
            TextBlock(type="text", text=reply.get_text_content()),
        ]
    )

async def run_pattern_5_hitl() -> None:
    print("\n" + "="*20 + " Pattern 5: HITL (人机协作) " + "="*20)
    
    # 1. AI 提出建议
    suggester = create_agent(
        name="疑难点分析师",
        sys_prompt=(
            "你是一名资深教学设计师。请找出课程中对初学者可能最难理解的一个概念，"
            "并提供一个更通俗易懂的解释作为修改建议。"
        ),
        model_name="qwen-plus",
        multi_agent=True,
    )
    disable_console_output([suggester])

    course_content = "在Python中，装饰器本质上是一个接收函数作为参数并返回一个新函数的函数..."
    print(f"原文片段: {course_content[:30]}...")
    
    suggestion = await suggester(Msg("user", course_content, "user"))
    print(f"AI 初步建议: {suggestion.content[:50]}...")

    # 2. 改写 Agent (配备人类咨询工具)
    toolkit = Toolkit()
    toolkit.register_tool_function(ask_human_decision)

    rewriter = ReActAgent(
        name="内容改写器",
        sys_prompt=(
            "你是课程内容改写器。基于提供的 AI 建议完成最终修改。\n"
            "- 如果建议很完美，可以直接采纳；\n"
            "- **演示要求**：请务必调用 ask_human_decision 工具向人类专家确认一下，以展示 HITL 流程。\n"
            "- 得到人类反馈后，整合生成最终内容。"
        ),
        model=DashScopeChatModel(
            model_name="qwen-max",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        ),
        formatter=DashScopeMultiAgentFormatter(),
        toolkit=toolkit,
    )
    # 不禁用 rewriter 的输出，让我们看到工具调用过程

    task = (
        "下面是课程摘录与 AI 的修改建议。请根据流程完成修改：\n\n"
        f"[课程内容]\n{course_content}\n\n"
        f"[AI 建议]\n{suggestion.get_text_content()}\n"
    )
    
    print("启动改写 Agent (即将触发工具调用)...")
    final_action = await rewriter(Msg("user", task, "user"))

    print("-" * 30)
    print("HITL 最终成果：")
    print(final_action.content)
    print("-" * 30)


# ==============================================================================
# 主程序入口
# ==============================================================================
async def main():
    # 依次演示各个模式
    # 在实际开发中，这些模式通常是独立运行或组合使用的
    
    # await run_pattern_1_pipeline()
    
    # await run_pattern_2_branching()
    
    # await run_pattern_3_parallel()
    
    # await run_pattern_4_moa()
    
    # await run_pattern_4_1_multi_layer_moa() # Added Multi-Layer MoA
    
    await run_pattern_5_hitl()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序已终止")
    except Exception as e:
        print(f"\n运行出错: {e}")
