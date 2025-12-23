# ==============================================================================
# 说明：本文件演示 Agent 协作的两种典型模式
# ==============================================================================
# 在复杂任务中，单个 Agent 往往力不从心。通过多 Agent 协作，可以：
# - 利用专家分工，提升输出质量
# - 并行处理任务，提高效率
# - 通过讨论和迭代，激发创新
#
# 【本文件演示的两种协作模式】
#
# 模式 1: Hierarchical/Team Leader Pattern (分层规划)
# -------------------------------------------------------
# 【架构】Leader Agent 作为项目主管，Member Agents 作为专家团队成员
# 【实现】通过 Handoff 机制，将成员 Agent 注册为工具供 Leader 调用
# 【工作流】Leader 分析任务 → 调用专家工具 → 收集结果 → 整合输出
# 【优势】清晰的指挥链、可追溯的决策过程、易于扩展新专家
# 【场景】需要明确分工和顺序执行的任务（如课程开发、项目规划）
#
# 模式 2: Co-creation/Blackboard Pattern (共创模式)
# -------------------------------------------------------
# 【架构】所有 Agents 平等参与，通过共享记忆空间（Blackboard）协作
# 【实现】使用 MsgHub 作为"黑板"，Agents 读取他人贡献后迭代完善
# 【工作流】发布任务到黑板 → 各 Agent 并行贡献 → 相互启发 → 涌现共识
# 【优势】激发创造性、多视角融合、适合开放性问题
# 【场景】需要头脑风暴和多轮迭代的任务（如创意策划、方案设计）
#
# ------------------------------------------------------------------------------
# 【与其他脚本的对比】
# ------------------------------------------------------------------------------
# 本脚本 vs 9_agent_3_workflow.py (基础编排)：
#   - workflow.py: 关注静态编排逻辑（Pipeline/Parallel/MoA 等固定模式）
#   - 本脚本: 关注 Agent 间的协作机制（Handoff 工具化、MsgHub 共享记忆）
#   - 核心区别: workflow 是"怎么组织流程"，coop 是"怎么让 Agent 互通信息"
#
# 本脚本 vs 9_agent_4_auto.py (自主规划)：
#   - auto.py: Agent 自己制定和调整计划（PlanNotebook），高度自主
#   - 本脚本: Agent 按预设角色协作，Leader 指挥或平等讨论，结构化协作
#   - 核心区别: auto 强调"自主性"，coop 强调"专业分工和信息共享"
#
# 技术实现的本质:
#   - 模式 1 (Hierarchical): 将 Agent 包装为 Tool，本质是 Function Calling 变体
#   - 模式 2 (Co-creation): 通过 MsgHub 实现共享上下文，类似 MoA + 多轮迭代
#
# 【关键洞察：信息流模式的差异】
# ------------------------------------------------------------------------------
# workflow.py (线性传播 - Agent → Agent):
#   ├─ Agent A 输出 → Agent B 输入 → Agent C 输入
#   └─ 每个 Agent 只看到前一个的输出（链式传递）
#   └─ 信息特点：单向流动，逐步加工
#
# 本脚本-模式1 (单向汇聚 - Agent → Leader):
#   ├─ 专家 A → Leader  ┐
#   ├─ 专家 B → Leader  ├→ Leader 整合所有信息
#   └─ 专家 C → Leader  ┘
#   └─ 专家之间互不可见（信息隔离）
#   └─ 信息特点：星形拓扑，中心聚合
#
# 本脚本-模式2 (全局共享 - Agent ↔ MsgHub):
#   ├─ Agent A ─┐
#   ├─ Agent B ─┼→ MsgHub (黑板) ←─ 所有 Agent 都能读写
#   └─ Agent C ─┘
#   └─ 信息对所有参与者透明（共享记忆空间）
#   └─ 信息特点：全连接拓扑，实时同步
#
# auto.py (状态管理 - Agent ↔ PlanNotebook):
#   ├─ Agent 读取 ← PlanNotebook (外部状态管理器)
#   ├─ Agent 写入 → PlanNotebook
#   └─ 通过工具调用双向交互：
#       create_plan() / finish_subtask() / revise_plan() / update_state()
#   └─ 信息特点：Agent 与"外部记忆"对话，而非 Agent 间直接通信
#   └─ 关键差异：PlanNotebook 不是 Agent，是持久化的状态容器
#
# 四种模式的本质区别：
#   - workflow: Agent 之间传递消息（消息驱动）
#   - coop-模式1: Agent 向中心节点汇报（集中式）
#   - coop-模式2: Agent 通过共享空间协作（分布式共享内存）
#   - auto: Agent 操作外部状态（事件溯源/CQRS 模式）
# ==============================================================================

import asyncio
import os
from textwrap import dedent

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub
from agentscope.tool import ToolResponse, Toolkit

from config.load_key import load_key


# ==============================================================================
# 模式 1: Hierarchical/Team Leader Pattern (分层规划模式)
# ==============================================================================

# ---- 专家 Agent 的系统提示 ----

DESIGNER_LI_PROMPT = """
你是李老师，一位经验丰富的教学设计师。你的任务是为"Pandas 数据分析入门"课程设计出清晰、有逻辑的教学大纲。专注于：1. 定义每个模块清晰的学习目标。2. 确保知识点由浅入深，循序渐进。3. 提出互动性的练习和项目来巩固学习效果。
"""

SCIENTIST_WANG_PROMPT = """
你是王工，一位资深数据科学家，也是 Pandas 的实战专家。你的任务是为课程提供准确、实用的技术内容。专注于：1. 提供最核心、最常用的 Pandas 知识点。2. 设计源于真实工作场景的案例和数据集。3. 编写简洁、规范、易于理解的代码示例。
"""

WRITER_ZHANG_PROMPT = """
你是小张，一位充满创意的课程内容编写者。你的任务是把技术内容讲得通俗易懂、但不失严谨性、用词冷静克制的课程文稿。专注于：1. 用通俗易懂的语言和比喻来解释复杂概念。2. 设计真实性高的案例场景和模块标题。3. 确保课程的整体基调是鼓励性和启发性的。
"""

LEADER_PROMPT = """
你是一个课程项目主管，负责协调团队完成"Pandas入门课程"的初稿开发。
你有三名团队成员可以作为工具调用，他们每个人的工作都依赖于前一个人的输出。

你的工作流程必须严格遵循以下顺序：
1.  **首先，调用 invoke_designer_li**，让他为课程创建一个初步的大纲和学习目标。
2.  **其次，调用 invoke_scientist_wang**。将李老师生成的大纲作为 `context` 参数传递给他，要求他根据这个大纲填充技术要点和代码示例。
3.  **接着，调用 invoke_writer_zhang**。将李老师和王工的全部产出合并后作为 `context` 参数传递给她，要求她在此基础上撰写完整的、对学习者友好的课程文稿。
4.  **最后**，在收到所有专家的最终结果后，将它们整合成一份格式统一、内容完整的最终课程文档，然后作为你的最终回复。
"""


# ---- 工具函数：模型和 Agent 创建 ----

def get_model_instance() -> DashScopeChatModel:
    """获取一个统一配置的模型实例。"""
    return DashScopeChatModel(
        model_name="qwen-plus",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )

def create_member_agent(name: str, sys_prompt: str) -> ReActAgent:
    """根据给定的名称和系统提示创建一个团队成员 Agent。"""
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=get_model_instance(),
        formatter=DashScopeMultiAgentFormatter(),
    )


# ---- 将团队成员封装为工具 (Handoff 机制) ----

async def invoke_designer_li(task_description: str, context: str = "") -> ToolResponse:
    """
    当需要设计课程大纲、学习目标或教学活动时，调用教学设计师李老师。

    Args:
        task_description (str): 清晰地描述你需要李老师完成的设计任务。
        context (str): 可选。传递相关的背景信息或先前的工作成果。
    """
    print("\n--- 任务分派：正在调用教学设计师李老师 ---")
    agent = create_member_agent("DesignerLi", DESIGNER_LI_PROMPT)
    
    content_for_agent = task_description
    if context:
        content_for_agent = f"背景信息：\n{context}\n\n你的任务：{task_description}"
        
    result_msg = await agent(Msg(name="user", role="user", content=content_for_agent))
    return ToolResponse(content=result_msg.get_text_content())

async def invoke_scientist_wang(task_description: str, context: str = "") -> ToolResponse:
    """
    当需要提供专业技术知识、代码示例或真实案例时，调用数据科学家王工。

    Args:
        task_description (str): 清晰地描述你需要王工完成的技术任务。
        context (str): 可选。传递课程大纲等先前的工作成果，以便他在此基础上工作。
    """
    print("\n--- 任务分派：正在调用数据科学家王工 ---")
    agent = create_member_agent("ScientistWang", SCIENTIST_WANG_PROMPT)
    
    content_for_agent = task_description
    if context:
        content_for_agent = f"请基于以下课程大纲和背景信息来完成你的任务：\n{context}\n\n你的具体任务是：{task_description}"
        
    result_msg = await agent(Msg(name="user", role="user", content=content_for_agent))
    return ToolResponse(content=result_msg.get_text_content())

async def invoke_writer_zhang(task_description: str, context: str = "") -> ToolResponse:
    """
    当需要将技术内容转化为易于理解的文稿时，调用内容编写者小张。

    Args:
        task_description (str): 清晰地描述你需要小张完成的写作任务。
        context (str): 可选。传递大纲和技术要点等先前的工作成果，作为写作基础。
    """
    print("\n--- 任务分派：正在调用内容编写者小张 ---")
    agent = create_member_agent("WriterZhang", WRITER_ZHANG_PROMPT)
    
    content_for_agent = task_description
    if context:
        content_for_agent = f"请基于以下课程的草稿（包含大纲和技术点）来完成你的写作任务：\n{context}\n\n你的具体任务是：{task_description}"
        
    result_msg = await agent(Msg(name="user", role="user", content=content_for_agent))
    return ToolResponse(content=result_msg.get_text_content())


# ---- 模式 1 主函数 ----

async def demo_hierarchical_pattern() -> None:
    """
    演示分层规划模式：Leader Agent 协调专家团队完成课程开发。
    
    【流程】
    1. Leader Agent 分析顶层任务
    2. 依次调用专家工具：设计师 → 数据科学家 → 内容编写者
    3. Leader Agent 整合所有专家的输出，生成最终成果
    """
    print("\n" + "="*60)
    print("  模式 1: Hierarchical/Team Leader Pattern (分层规划)")
    print("="*60 + "\n")
    
    # 1. 创建 Leader 的工具箱并注册团队成员
    leader_toolkit = Toolkit()
    leader_toolkit.register_tool_function(invoke_designer_li)
    leader_toolkit.register_tool_function(invoke_scientist_wang)
    leader_toolkit.register_tool_function(invoke_writer_zhang)

    # 2. 创建 Leader Agent
    leader_agent = ReActAgent(
        name="ProjectLeader",
        sys_prompt=LEADER_PROMPT,
        model=get_model_instance(),
        toolkit=leader_toolkit,
        formatter=DashScopeMultiAgentFormatter(),
    )

    # 3. 定义顶层任务
    top_level_task = (
        "请为初学者创建一节关于Pandas 数据分析的简短课程初稿。"
    )
    
    print(f"项目主管收到的顶层任务：\n{top_level_task}\n" + "="*50)

    # 4. 将任务交给 Leader Agent 执行
    final_response_msg = await leader_agent(Msg(name="user", role="user", content=top_level_task))

    # 5. 展示最终成果
    print("\n" + "="*50)
    print("  项目主管最终的汇总报告：")
    print("="*50 + "\n")
    print(final_response_msg.get_text_content())


# ==============================================================================
# 模式 2: Co-creation/Blackboard Pattern (共创模式)
# ==============================================================================

# ---- 专家 Agent 的系统提示（共创版本）----

DESIGNER_LI_PROMPT_COOP = """
你是李老师，一位经验丰富的教学设计师。你的任务是为"Pandas 数据分析入门"课程设计出清晰、有逻辑的教学大纲。在讨论中，你专注于：1. 定义每个模块清晰的学习目标。2. 确保知识点由浅入深，循序渐进。3. 提出互动性的练习和项目来巩固学习效果。
"""

SCIENTIST_WANG_PROMPT_COOP = """
你是王工，一位资深数据科学家，也是 Pandas 的实战专家。你的任务是为课程提供准确、实用的技术内容。在讨论中，你专注于：1. 提供最核心、最常用的 Pandas 知识点。2. 设计源于真实工作场景的案例和数据集。3. 编写简洁、规范、易于理解的代码示例。
"""

WRITER_ZHANG_PROMPT_COOP = """
你是小张，一位充满创意的课程内容编写者。你的任务是把技术内容讲得通俗易懂、但不失严谨性、用词冷静克制的课程文稿。在讨论中，你专注于：1. 用通俗易懂的语言和比喻来解释复杂概念。2. 设计真实性高的案例场景和模块标题。3. 确保课程的整体基调是鼓励性和启发性的。
"""


# ---- 工具函数：创建协作 Agent ----

def create_expert_agent(name: str, sys_prompt: str) -> ReActAgent:
    """根据给定的名称和系统提示创建一个专家 Agent（用于共创模式）。"""
    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=DashScopeChatModel(
            model_name="qwen-plus",
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
        ),
        formatter=DashScopeMultiAgentFormatter(),
    )


# ---- 模式 2 主函数 ----

async def demo_cocreation_pattern() -> None:
    """
    演示共创模式：多个专家 Agent 通过 MsgHub 共享记忆空间进行协作。
    
    【流程】
    1. 发布开放性任务到 MsgHub（黑板）
    2. 各专家 Agent 轮流发言，贡献想法
    3. 每个 Agent 看到其他人的贡献后，迭代完善自己的方案
    4. 经过多轮讨论，涌现出团队共识
    5. 最后由"会议秘书" Agent 整理讨论记录，生成最终成果
    """
    print("\n" + "="*60)
    print("  模式 2: Co-creation/Blackboard Pattern (共创模式)")
    print("="*60 + "\n")
    print("=== 开始课程开发会议：构思 'Pandas 入门' 课程大纲和案例 ===")

    # 1. 创建课程开发团队
    designer_li = create_expert_agent("李老师 (教学设计师)", DESIGNER_LI_PROMPT_COOP)
    scientist_wang = create_expert_agent("王工 (数据科学家)", SCIENTIST_WANG_PROMPT_COOP)
    writer_zhang = create_expert_agent("小张 (内容编写者)", WRITER_ZHANG_PROMPT_COOP)

    # 2. 定义会议开场白
    announcement = Msg(
        "system",
        (
            "团队好，我们今天的目标是共同协作，为'Pandas 数据分析入门'课程制定一个完整的、吸引人的**课程大纲和核心案例**。"
            "请大家集思广益，从教学设计师李老师开始，提出你的第一轮建议。"
        ),
        "system",
    )
    
    # 3. 启动多轮讨论（通过 MsgHub 实现共享记忆）
    async with MsgHub(
        participants=[designer_li, scientist_wang, writer_zhang],
        announcement=announcement,
    ) as hub:
        for i in range(2):  # 进行 2 轮协作讨论
            print(f"\n--- 第 {i + 1} 轮协作 ---")
            # 按照发言顺序依次调用
            await designer_li()
            await scientist_wang()
            await writer_zhang()

    print("\n=== 会议结束 ===")

    # ==================== 汇总阶段 ====================
    print("\n=== 开始生成最终团队成果（课程大纲初稿） ===")
    
    # 4. 定义一个"会议秘书" Agent 来整理会议纪要
    secretary_prompt = dedent("""
        你是一位专业的会议秘书，非常擅长整理会议纪要。
        你的任务是阅读下面的团队讨论记录，然后根据讨论内容，以清晰的 Markdown 格式，
        生成"Pandas 入门课程"的**课程大纲初稿**。

        大纲应包含以下部分：
        - **模块标题**：一个吸引人的标题。
        - **学习目标**：清晰列出学生学完本模块后能做什么。
        - **核心概念**：涵盖的关键技术点。
        - **核心案例**：贯穿本模块的实践案例和数据集。
        - **代码示例**：需要包含的关键代码演示。
        - **课后练习**：一个具体的动手练习任务。
    """)
    secretary_agent = create_expert_agent("会议秘书", secretary_prompt)

    # 5. 准备完整的讨论记录
    full_transcript_msgs = await designer_li.memory.get_memory()

    transcript_text = "以下是团队的讨论记录：\n\n"
    for msg in full_transcript_msgs:
        if msg.role != "system":
            transcript_text += f"[{msg.name}]: {msg.content}\n"
            
    # 6. 指派汇总任务
    final_task_prompt = dedent(
        f"{transcript_text}\n"
        "请根据以上讨论记录，整理出课程大纲初稿。"
    )
    
    # 7. 调用秘书 Agent 来完成任务
    final_output_msg = await secretary_agent(Msg("user", final_task_prompt, "user"))

    # 8. 展示最终成果
    print("\n" + "="*50)
    print("  最终团队成果：课程大纲初稿")
    print("="*50 + "\n")
    print(final_output_msg.content)


# ==============================================================================
# 主程序入口
# ==============================================================================

async def main():
    """主程序：依次运行两种协作模式的演示。"""
    load_key()
    
    # 示例 1：分层规划模式
    try:
        await demo_hierarchical_pattern()
    except Exception as e:
        print(f"模式 1 运行出错: {e}")
    
    # 示例 2：共创模式
    try:
        await demo_cocreation_pattern()
    except Exception as e:
        print(f"模式 2 运行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
