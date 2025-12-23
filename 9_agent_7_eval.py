# ==============================================================================
# 说明：本文件演示 Agent 系统的评测驱动开发 (Evaluation-Driven Development)
# ==============================================================================
# 在生产环境中，Agent 系统的质量直接影响业务成败。
# 没有评测，就没有优化；没有数据，就只能靠感觉调试。
#
# 【核心思想】
# 将评测作为开发的第一步，通过数据驱动的方式持续优化 Agent 系统。
#
# 【本文件演示内容】
# 1. 端到端评测 (End-to-End Evaluation)
#    - 从业务视角出发，评估 Agent 是否完成用户意图
#    - 评测流程：迭代式确定评估标准 → 设计评估指标 → 提升稳定性
#    - 主观评估：人工评测（黄金标准）→ LLM 自动化评测（规模化）
#
# 2. 白盒化评测 (White-box Evaluation)
#    - 深入系统内部，为具体组件设计针对性评估指标
#    - 精准定位问题：工具调用失败？记忆检索不准？
#
# 3. 实战案例：教育课程写作评测
#    - 任务：生成 Pandas 课程草稿（两节课）
#    - 评测：五维度 LLM 评分（clarity、factual_correctness、consistency、
#            redundancy、readability）
#    - 输出：结构化草稿 + 量化评分 + 改进建议
#
# 【评测体系的价值】
# ┌─────────────┬────────────────────────────────────────┐
# │ 评测类型    │ 核心价值                               │
# ├─────────────┼────────────────────────────────────────┤
# │ 端到端评测  │ 验证业务目标是否达成（宏观）           │
# │ 白盒化评测  │ 精准定位问题根源（微观）               │
# │ 持续迭代    │ 数据驱动优化，形成飞轮效应             │
# └─────────────┴────────────────────────────────────────┘
# ==============================================================================

import asyncio
import copy
import json
import os
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

import agentscope
from agentscope.message import Msg
from agentscope.agent import ReActAgent
from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.evaluate import (
    Task,
    MetricBase,
    MetricResult,
    MetricType,
    SolutionOutput,
)

from config.load_key import load_key


# ==============================================================================
# 步骤 1: 定义评测基准 (Benchmark)
# ==============================================================================
# 一个好的基准应该：
# 1. 覆盖核心业务场景
# 2. 任务难度有梯度（简单 → 中等 → 复杂）
# 3. 包含明确的评估维度和标准

COURSE_BENCHMARK = [
    {
        "id": "pandas_intro",
        "prompt": (
            "请用简洁中文写一个 pandas DataFrame 入门课节草稿。输出必须包含结构化字段：\n"
            "- title: 课程标题；\n"
            "- learning_objectives (3-5 条，每条不超过 25 字)；\n"
            "- lesson_content: 至少 180 字的课程正文，包含开场引导、核心概念解释、逐步示例讲解和课堂小结；\n"
            "- code_example: 包含注释的最小可运行 pandas 代码（展示 import pandas as pd 和 read_csv 或 DataFrame 创建以及 head() 使用示例）；\n"
            "- quiz: 一道带选项的单选题，选项不少于 4 个，并标记正确答案。\n"
            "lesson_content 应强调新手常见误区，并与代码示例呼应。"
        ),
        "tags": {"topic": "intro", "min_objectives": 3},
    },
    {
        "id": "pandas_groupby",
        "prompt": (
            "请用简洁中文写一个 pandas groupby 与聚合课节草稿。输出必须包含结构化字段：\n"
            "- title: 课程标题；\n"
            "- learning_objectives (3-5 条，每条不超过 25 字)；\n"
            "- lesson_content: 至少 200 字的课程正文，先解释 groupby 思路，再用真实业务背景拆解聚合步骤，包含 agg 与 describe 差异，并加入常见错误提示；\n"
            "- code_example: 包含注释的最小可运行 pandas 代码，至少展示一次 groupby 与一次 agg 或 describe；\n"
            "- quiz: 一道带选项的单选题，选项不少于 4 个，并标记正确答案。\n"
            "lesson_content 应提供逐步操作讲解和拓展思考。"
        ),
        "tags": {"topic": "groupby", "min_objectives": 3},
    },
]


# ==============================================================================
# 步骤 2: 定义结构化输出模型
# ==============================================================================
# 使用 Pydantic 确保 Agent 输出的结构一致性，便于后续评测

class CourseDraft(BaseModel):
    """课程草稿的结构化模型"""
    title: str = Field(description="课节标题")
    learning_objectives: List[str] = Field(description="学习目标（3-5 条）")
    lesson_content: str = Field(description="课程正文，至少 180 字的详细草稿")
    code_example: str = Field(description="最小可运行的 pandas 代码示例")
    quiz: str = Field(description="一道选择题（简短）")


# ==============================================================================
# 步骤 3: 定义评测指标 - 五维度 LLM 评分
# ==============================================================================
# LLM-as-Judge 范式：使用大模型作为评审专家
# 
# 【核心挑战】
# 1. 评分标准的一致性：通过详细的评分细则减少随机性
# 2. 评审 Agent 的状态隔离：避免跨任务的记忆污染
# 3. 结构化输出：强制 LLM 输出标准化的评分字段

class EvalScore(BaseModel):
    """五维度评分模型"""
    clarity: int = Field(description="语言表达清晰度/歧义性，1-5（高=更清晰）")
    factual_correctness: int = Field(description="事实正确性，1-5（高=更正确）")
    consistency: int = Field(description="前后表达一致性，1-5（高=更一致）")
    redundancy: int = Field(description="表达冗余度，1-5（高=更简洁）")
    readability: int = Field(description="易读性，1-5（高=更易读）")
    overall: Optional[float] = Field(default=None, description="可选，总分 0-1")
    feedback: str = Field(description="一句话改进建议")


class LLMEvalMetric(MetricBase):
    """
    基于 LLM 的评测指标实现。
    
    【核心机制】
    1. 使用独立的 Evaluator Agent 进行评分
    2. 每次评测前重置 Agent 状态，避免记忆污染
    3. 通过详细的评分细则引导 LLM 输出稳定的结果
    4. 将多维度评分加权汇总为最终分数
    
    【评分流程】
    输入草稿 → Evaluator Agent（结构化输出）→ 五维度评分 → 加权计算总分
    """
    
    def __init__(self, eval_agent: ReActAgent, axis_weights: Optional[Dict[str, float]] = None):
        super().__init__(
            name="llm_eval_course_draft",
            metric_type=MetricType.NUMERICAL,
            description="LLM-as-Judge for five axes",
            categories=[],
        )
        self.eval_agent = eval_agent
        self.axis_weights = axis_weights or {
            "clarity": 1.0,
            "factual_correctness": 1.0,
            "consistency": 1.0,
            "redundancy": 1.0,
            "readability": 1.0,
        }
        # 保存 Evaluator 的初始状态快照，确保每次评测都是"干净"的
        self._initial_state = copy.deepcopy(self.eval_agent.state_dict())

    async def __call__(self, solution: SolutionOutput) -> MetricResult:
        """
        对单个解决方案进行评分。
        
        Args:
            solution: 包含 Agent 生成的课程草稿
            
        Returns:
            MetricResult: 包含最终评分和详细信息
        """
        # 重置评审 Agent 的状态，避免跨任务的记忆污染
        try:
            self.eval_agent.load_state_dict(copy.deepcopy(self._initial_state))
        except Exception as exc:  # pragma: no cover - defensive
            return MetricResult(
                name=self.name,
                result=0.0,
                message=f"failed to reset evaluator state: {exc}",
            )

        draft = solution.output or {}
        
        # 构建详细的评分提示词，包含：
        # 1. 评分维度的明确定义
        # 2. 评分标准的参考锚点（5分=卓越，4分=优秀，3分=合格，2分及以下=需大幅修改）
        # 3. 具体的检查要点
        prompt = (
            "请作为教育内容评审，对以下课节草稿按 1-5 分评测五个维度，并给出一句话改进建议。\n"
            "评分维度：\n"
            "1) clarity: 语言表达清晰度/歧义性（更清晰得分更高）；\n"
            "2) factual_correctness: 是否有事实错误（更正确得分更高）；\n"
            "3) consistency: 前后表达一致性（更一致得分更高）；\n"
            "4) redundancy: 表达冗余度（冗余越少得分越高）；\n"
            "5) readability: 易读性（更易读得分更高）。\n"
            "评分参考：5 分仅限几乎无需修改的卓越稿件；4 分代表优秀但仍需轻微调整；3 分表示基本合格但存在明显问题；2 分或以下意味着需要大幅修改。若 lesson_content 字数不足 180、缺少逐步讲解或未覆盖常见误区，请将 clarity 与 readability 的评分上限设为 3 分，并在反馈中说明。\n"
            "检查要点：学习目标数量是否符合要求、lesson_content 是否包含引入→概念→示例→总结且与代码呼应、代码示例是否可运行并附注释、测验是否明确标注正确答案。\n"
            "只输出结构化字段：clarity、factual_correctness、consistency、redundancy、readability、overall(可选)、feedback。\n\n"
            f"标题: {draft.get('title','')}\n"
            f"学习目标: {draft.get('learning_objectives', [])}\n"
            f"正文:\n{draft.get('lesson_content','')}\n\n"
            f"代码示例:\n{draft.get('code_example','')}\n"
            f"测验: {draft.get('quiz','')}\n"
        )

        try:
            res = await self.eval_agent(
                Msg("user", prompt, role="user"),
                structured_model=EvalScore,
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                result=0.0,
                message=f"evaluator call failed: {exc}",
            )

        s = res.metadata or {}
        if not isinstance(s, dict):
            return MetricResult(
                name=self.name,
                result=0.0,
                message=f"invalid evaluator metadata type: {type(s).__name__}",
            )

        axes = ["clarity", "factual_correctness", "consistency", "redundancy", "readability"]

        def norm(v: int) -> float:
            """将 1-5 分转换为 0-1 区间"""
            return max(0.0, min(1.0, (float(v) - 1.0) / 4.0))

        def _coerce(axis: str) -> int:
            """强制转换为整数，缺失时默认为 1 分"""
            if axis not in s:
                return 1
            return int(s[axis])

        try:
            # 提取各维度评分，缺失时使用基线分数
            values = {axis: _coerce(axis) for axis in axes}
        except Exception as exc:
            return MetricResult(
                name=self.name,
                result=0.0,
                message=f"invalid evaluator payload: {exc}",
            )

        # 加权平均计算最终得分
        weighted = sum(self.axis_weights[axis] * norm(values.get(axis, 1)) for axis in axes)
        denom = sum(self.axis_weights.values()) or 1.0
        score = weighted / denom

        # 生成详细的评分报告
        msg = (
            f"clarity={values.get('clarity')} | factual={values.get('factual_correctness')} | "
            f"consistency={values.get('consistency')} | redundancy={values.get('redundancy')} | "
            f"readability={values.get('readability')} | feedback={s.get('feedback','')}"
        )
        return MetricResult(name=self.name, result=score, message=msg)


# ==============================================================================
# 步骤 4: 组装评测任务
# ==============================================================================

def build_tasks() -> list[Task]:
    """
    将基准数据转换为评测任务列表。
    
    每个 Task 包含：
    - input: 给 Agent 的提示词
    - ground_truth: 期望的评分（这里设为 1.0，表示期望全部通过）
    - tags: 任务的元数据标签
    - metrics: 评测指标（稍后注入）
    """
    tasks: list[Task] = []
    for item in COURSE_BENCHMARK:
        tasks.append(
            Task(
                id=item["id"],
                input=item["prompt"],
                ground_truth=1.0,  # 期望全部通过客观检查
                tags=item["tags"],
                metrics=[],  # 稍后注入 LLM 评分指标
                metadata={},
            )
        )
    return tasks


# ==============================================================================
# 步骤 5: 创建 Agent（生成器和评审器）
# ==============================================================================

def create_agents():
    """创建课程生成 Agent 和评审 Agent"""
    
    # 生成器 Agent：负责生成课程草稿
    generator = ReActAgent(
        name="Friday",
        sys_prompt=(
            "你是一名教育课程作者，专注于 pandas 数据分析。请用简洁中文撰写课节草稿，"
            "严格输出结构化字段：title、learning_objectives(list[str])、lesson_content(str)、code_example(str)、quiz(str)。"
            "lesson_content 至少 180 字，包含引入、概念讲解、逐步示例、常见错误提醒与总结。"
        ),
        model=DashScopeChatModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model_name="qwen-plus",
            stream=False,
        ),
        formatter=DashScopeChatFormatter(),
        enable_meta_tool=False,
    )
    
    # 评审器 Agent：负责评分
    evaluator = ReActAgent(
        name="Evaluator",
        sys_prompt=(
            "你是一名严格的教育内容评审，按照评分标准输出结构化分数（1-5）和一句话建议，"
            "不要输出除结构化以外的任何内容。"
        ),
        model=DashScopeChatModel(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model_name="qwen-plus",
            stream=False,
        ),
        formatter=DashScopeChatFormatter(),
        enable_meta_tool=False,
    )
    
    return generator, evaluator


# ==============================================================================
# 步骤 6: 执行评测循环
# ==============================================================================

async def run_minimal_eval() -> None:
    """
    最小化评测循环演示。
    
    【流程】
    1. 构建评测任务
    2. 为每个任务注入评测指标
    3. 让生成 Agent 完成任务
    4. 让评审 Agent 对结果进行评分
    5. 汇总并输出平均分
    """
    print("\n" + "="*60)
    print("  开始评测：Pandas 课程草稿生成质量")
    print("="*60 + "\n")
    
    # 创建 Agent
    agent, evaluator = create_agents()
    
    # 构建任务
    tasks = build_tasks()
    
    # 注入五维度 LLM 评分指标
    metric = LLMEvalMetric(eval_agent=evaluator)
    for t in tasks:
        t.metrics = [metric]
    
    scores = []
    
    for task in tasks:
        print(f"\n{'─'*60}")
        print(f"任务 ID: {task.id}")
        print(f"{'─'*60}\n")
        
        # 让生成 Agent 完成任务
        res = await agent(
            Msg("user", task.input, role="user"),
            structured_model=CourseDraft,
        )
        
        # 提取结构化输出
        draft = {
            "title": res.metadata.get("title"),
            "learning_objectives": res.metadata.get("learning_objectives"),
            "lesson_content": res.metadata.get("lesson_content"),
            "code_example": res.metadata.get("code_example"),
            "quiz": res.metadata.get("quiz"),
        }
        
        print(f"✅ 生成的课程草稿:\n")
        print(json.dumps(draft, ensure_ascii=False, indent=2))
        
        # 让评审 Agent 评分
        solution = SolutionOutput(success=True, output=draft, trajectory=[])
        metric_res = await task.metrics[0](solution)
        scores.append(metric_res.result)
        
        print(f"\n📊 评分结果: {metric_res.result:.2f}")
        print(f"📝 详细信息: {metric_res.message}\n")
    
    # 计算平均分
    avg = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "="*60)
    print(f"  平均评分: {avg:.2f} / 1.00")
    print("="*60 + "\n")



# ==============================================================================
# 主程序入口
# ==============================================================================

async def main():
    """主程序：执行评测演示"""
    load_key()
    
    # 检查 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # （可选）开启追踪：优先连接 Studio；否则连接任意 OTLP 兼容后端
    studio_url = os.getenv("AGENTSCOPE_STUDIO_URL")
    otel_url = os.getenv("OTEL_TRACING_URL")
    if studio_url:
        agentscope.init(studio_url=studio_url)
    elif otel_url:
        agentscope.init(tracing_url=otel_url)
    
    # 执行评测
    try:
        await run_minimal_eval()
    except Exception as e:
        print(f"评测运行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())


# ==============================================================================
# 总结与展望：上下文工程视角下的 Agent 系统
# ==============================================================================

"""
1 一切始于"用户意图"
─────────────────────────────────────────────────────────────
你从一个常见的痛点出发：单一的 LLM 调用，虽然能处理语言任务，但面对需要多步骤、
与外部世界交互的复杂任务时，往往力不从心。

要解决这个问题，你的目标是构建一个能够稳定、可靠地理解并完成用户真实意图的系统。
这正是 Agent 的价值所在。如果说 LLM 是一个聪明但被束缚在"黑盒"里的"大脑"，
那么你这章所学的，就是如何围绕这个"大脑"，构建起一套完整的"工程系统"，
让它能完成真实世界的复杂任务。


2 核心方法论：上下文工程 (Context Engineering)
─────────────────────────────────────────────────────────────
在探索如何构建 Agent 的过程中，你学习了多种工程技术。现在，让我们从一个核心的
工程思想上来审视这些技术：上下文工程 (Context Engineering)。

你最开始接触大模型时，学习的是提示词工程 (Prompt Engineering)，它的核心是在
单次交互中，通过精心设计的指令，让模型给出最佳输出。

而你现在面对的，是跨越多轮交互、涉及多个外部信息源的复杂任务。此时，你的工作
重心就从"写好一段 Prompt"，升级为系统性地为模型的每一步决策，都提供最充分、
最精准的"上下文"。这就是上下文工程。

从这个视角看，你作为 Agent 的设计者，就不再仅仅是一个"提问者"，而是一个
"信息架构师"。你的核心任务，是通过设计流程、调用工具，为最终负责决策的 LLM 
节点，构建出一个完美的"信息茧房"，其中包含了解决问题所需的一切信息，不多也不少。
上下文的质量，直接决定了 Agent 的能力上限。

在"上下文工程"这一工程路线下，你在本章前半部分学到的这些能力，都有了清晰的定位：
它们都是"上下文注入"这门学问中的具体手段：

- 工具使用 (Tool Use)：向上下文中注入来自外部世界的实时、确定性信息。
- 反思能力 (Reflection)：将上一步输出的"评估结果"作为新的反馈，修订下一步的
  上下文，以指导下一步的行动。
- 工作流 (Workflow)：设计一条相对固定的任务流水线，规定在什么时机、由哪个节点
  向模型中注入什么样的上下文，就像一条专门"操作上下文"的流水线，驱动后续所有
  步骤稳定运行。
- 记忆系统 (Memory)：从海量的历史信息中检索最相关的部分，注入当前上下文，为
  后续的决策提供参考。

这些能力并非孤立存在，而是共同服务于"优化上下文"这一核心目标，让 Agent 更高效、
更可靠地完成任务。


3 质量保障：评测驱动的开发
─────────────────────────────────────────────────────────────
既然上下文是核心，那么你如何确保自己构建的上下文是"高质量"的？答案是：评测。

没有测量，就没有优化。将 Agent 的开发过程从"凭感觉调试"变为"数据驱动"，是其
能否在生产环境中落地的关键。为此，你学习了三个互补的评测维度：

- 端到端评测：它回答了最重要的问题："Agent 是否最终完成了用户的意图？" 
  这是从业务视角出发的宏观评估。
- 白盒化评测：它能帮助你精准定位问题所在："是工具调用失败了，还是记忆检索不准？" 
  这是深入系统内部，对每一个"上下文注入"节点的微观诊断。
- 持续迭代：将评测融入开发的每一个环节，形成一个由数据驱动、持续优化的闭环。
  这确保了 Agent 的能力能够随着业务发展和数据积累而不断进化。

通过建立起这样一套"宏观+微观+迭代"的评测体系，你就拥有了一个驱动 Agent 不断
迭代优化的飞轮。更重要的是，这套体系并不局限于上下文工程，而是作为整个 Agent 
系统的"反馈神经系统"，贯穿于后续的自主规划和多智能体协作。


4 展望：更强大的自主性
─────────────────────────────────────────────────────────────
到目前为止，我们讨论的"上下文工程"，核心是围绕一个任务，构建一个"单兵能力"
极强、行为边界清晰可控的 Agent。你已经掌握了让它使用各种工具、拥有记忆、学会
反思的基础范式。这条以固定工作流为中心的工程化路线，已经非常强大，并且是当前
业界最主流、最务实的落地范式。

但单个 Agent 的进化并未止步于此。其发展的下一个阶段，是赋予它更高的"自主性"：

- 自主规划 (Planning)：Agent 不再严格执行你预设的静态工作流，而是能像一个
  项目经理，根据最终目标自主规划、拆解和执行任务。这意味着 Agent 拥有了动态
  适应问题、并自主生成解决方案的能力。
  
- 多智能体协作 (Multi-Agent Systems)：在自主规划的基础上，更进一步的是让多个
  具备自主决策能力的 Agent 组成团队，让它们围绕共同目标自行分工、发起协作、
  交换信息和整合中间成果，而不是由你逐条写死协作流程。此时，系统的"自主性"
  不再局限于单个 Agent 的规划能力，而是通过多智能体之间的互动，被放大为一种
  更接近真实团队的群体智能。

这代表了 Agent 系统从以工程可控为主的模式，迈向更开放的自主性和涌现行为的一次
跃迁。虽然你学到的结构化工作流是当前工程实践中最可靠的基石，但探索自主规划、
多智能体协作等高级能力，将是解锁 Agent 下一阶段潜力的关键。

最终，你需要与时俱进。先将你学到的上下文工程和评测体系运用到实践中，构建出一个
以固定工作流为骨架、行为可控且高度可靠的 Agent 工程系统。在此基础上，在同一套
评测与反馈机制的护航下，逐步引入自主规划和多智能体协作，让系统在保证可控性的
前提下，向更开放的自主性和涌现智能迈进。


5 最终总结
─────────────────────────────────────────────────────────────
一个优秀的 Agent 系统，并非某种单一的先进技术，而是两条互补路线的结合：

1. 工程化的可控自治路线：深入理解业务场景，将现实世界高效的流程与评价原则，
   转化为 Agent 的工作流与评测体系，并通过上下文工程精细控制每一个环节，在
   明确边界内发挥模型的"单兵能力"，构建起一个可靠、可控的 Agent 工程系统。
   这是战略与战术层面的"工程基座"。

2. 开放式的自主进化路线：在这个工程基座之上，在同一套评测与反馈机制的约束和
   指引下，引入自主规划和多智能体协作，让 Agent 不再只是被动执行预设流程，
   而是能够围绕目标自行规划、分工协作并产生涌现行为，在保证可控性的前提下，
   逐步提升系统的整体智能上限。这是面向未来的"智能进化"。

掌握了这两条路线，你就具备了构建强大、可靠、且能够持续进化的智能系统的基础。
现在，是时候用它们去实现你的创意，打造属于你的 AI 应用、系统和服务了。
"""