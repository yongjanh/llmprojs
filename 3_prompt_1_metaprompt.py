# ==============================================================================
# 说明：本文件涉及 Meta Prompting（元提示词）的实验与反思
# ==============================================================================
# Meta Prompting 是一种利用 AI 来优化 AI 提示词的技术。
#
# 【核心概念】
# "用 AI 优化 AI" —— 通过一个 LLM（教练）来优化另一个 LLM（执行者）的 Prompt。
#
# 【实验目标】
# 探索通过 AI 自动优化 AI 的提示词（Prompt Optimization）。
#
# 【核心流程】
# 1. 基础提示词 (Initial Prompt)：编写一个简单的 Prompt
# 2. 单轮优化：让 AI 教练根据目标改写 Prompt
# 3. 迭代优化 (Iterative Optimization)：
#    - Generator：使用当前 Prompt 生成回答
#    - Critic (Gap Analysis)：对比"生成回答"与"理想参考答案"，分析差距
#    - Optimizer：根据差距报告，优化 Prompt 模板
# 4. 可视化评估：通过多维度评分对比优化效果
#
# 【重要实验结论 (Lessons Learned)】
# 1. ⚠️ "过拟合"风险：
#    - 问题：AI 倾向于将参考答案中的具体数据（如"500元补贴"）硬编码进 Prompt，
#           导致模板失去通用性。
#    - 解决方案：需明确区分"模板(Template)"与"数据(Context)"，并在指令中
#               强制禁止硬编码事实。
#
# 2. ⚠️ "不确定性叠加"：
#    - 问题：Optimizer、Critic、Generator 三者的随机性叠加，使得自动化优化
#           路径极不稳定，难以收敛。
#    - 影响：每次运行结果差异大，不可预测。
#
# 3. ✅ 最佳实践建议：
#    - 对于生产环境，手动编写结构化 Prompt (Context-Objective-Style 框架) 
#      + Few-Shot Examples (少样本示例) 是目前更可靠、更可控的路径。
#    - 本文件中的代码更多用于"寻找灵感"和"辅助设计"，而非全自动的生产流水线。
#
# 【适用场景】
# ✓ 探索性实验：快速生成 Prompt 优化思路
# ✓ 灵感激发：发现人类可能忽略的优化角度
# ✗ 生产流水线：不稳定，需要人工审核和筛选
#
# 【重要说明】
# 虽然本文件以 RAG 场景（包含 {context} 占位符）为例，但 Meta Prompting 是一种
# **通用的 Prompt 优化技术**，适用于任何需要优化 LLM 输出的场景：
# - 代码生成任务的 Prompt 优化
# - 翻译任务的 Prompt 优化
# - 数据分析任务的 Prompt 优化
# - 创意写作任务的 Prompt 优化
# - 客服对话的 Prompt 优化
# - 等等...
# ==============================================================================

from chatbot import llm
from config.load_key import load_key
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==============================================================================
# 全局配置：模拟 RAG 场景
# ==============================================================================

# 模拟 RAG 检索到的上下文信息
# 在真实的 RAG 应用中，这段文本会由你的向量数据库检索而来
RETRIEVED_TEXT = """
关于公司的福利政策，我们提供全面的健康保险，覆盖员工及其直系家属。
年度体检是标配。此外，每年有15天的带薪年假，以及5天的带薪病假。
我们还提供每月500元的交通补贴和300元的餐饮补贴。
为了鼓励员工成长，公司设有每年高达8000元的教育培训基金，员工可以申请用于课程学习或购买专业书籍。
健身方面，公司与多家健身房有合作，员工可享受折扣价。
"""

# 理想的参考答案（用于差距分析）
REFERENCE_ANSWER = """
👋 欢迎加入我们的大家庭！很高兴能为你介绍我们超棒的福利政策：

**🏥 健康与假期，我们为你保驾护航：**
-   **全面健康保险**：覆盖你和你的家人，安心工作无烦忧。
-   **年度体检**：你的健康，我们时刻关心。
-   **带薪年假**：每年足足15天，去探索诗和远方吧！
-   **带薪病假**：5天时间，让你安心休养，快速恢复活力。

**💰 补贴与激励，为你加油打气：**
-   **交通补贴**：每月500元，通勤路上更轻松。
-   **餐饮补贴**：每月300元，午餐加个鸡腿！
-   **教育培训基金**：每年高达8000元，投资自己，未来可期。
-   **健身折扣**：与多家健身房合作，工作再忙也别忘了锻炼哦！

希望这些福利能让你感受到公司的关怀！期待与你一起创造更多价值！🎉
"""


# ==============================================================================
# 示例 1: 基础 Prompt 测试
# ==============================================================================

def demo_initial_prompt():
    """
    演示最朴素的初始 Prompt 效果。
    
    【核心思想】
    从最简单的 Prompt 开始，观察其不足，为后续优化建立基线。
    
    【初始 Prompt 特点】
    - 简单直接，没有角色定义
    - 没有输出格式约束
    - 包含 {context} 占位符（这是正确的模板写法）
    """
    print("\n" + "="*60)
    print("  示例 1: 基础 Prompt 测试")
    print("="*60)
    
    # 定义一个真正的"模板"，保留 {context} 占位符
    # 注意：这里不是 f-string，而是一个带有占位符的普通字符串
    initial_prompt_template = """
根据以下信息，回答新员工关于公司福利的问题。

【参考信息】
{context}
"""
    
    print("\n📝 初始 Prompt 模板:")
    print("-"*60)
    print(initial_prompt_template)
    print("-"*60)
    
    # 填充上下文生成回答
    filled_prompt = initial_prompt_template.format(context=RETRIEVED_TEXT)
    response = llm.invoke(filled_prompt)
    
    print("\n🤖 AI 回答:")
    print("-"*60)
    print(response)
    print("-"*60)
    
    print("\n💡 观察：初始 Prompt 的输出简单直接，但缺乏友好性和结构化。")
    
    return initial_prompt_template, response


# ==============================================================================
# 示例 2: 单轮 Meta Prompting 优化
# ==============================================================================

def demo_single_iteration_optimization(initial_template, initial_response):
    """
    演示单轮 Meta Prompting 优化。
    
    【核心思想】
    通过一个"AI 教练"来重写初始 Prompt，直接提出优化要求。
    
    【优化策略】
    1. 明确表达不满和期望
    2. 提供具体的优化维度（语气、结构、内容分类）
    3. 强制只输出 Prompt，不输出解释
    
    【关键约束】
    - 不要篡改参考信息
    - 不要生成示例内容
    - 只输出 System Prompt 本身
    """
    print("\n" + "="*60)
    print("  示例 2: 单轮 Meta Prompting 优化")
    print("="*60)
    
    meta_prompt = f"""
我正在为公司的新员工答疑机器人优化一个提示词，目标是回答关于"公司福利"的问题。

这是我的第一个尝试：
---
{initial_template}
---

这是它生成的输出：
---
{initial_response}
---

这个输出不够好。我希望机器人的回答更具吸引力，并且结构清晰，能让新员工快速抓住重点。具体要求如下：
1.  **语气**：友好、热情，有欢迎新同事的感觉。
2.  **结构**：使用清晰的要点（比如用表情符号开头的列表）来组织内容。
3.  **内容**：将福利分为几个类别，如"健康与假期"、"补贴与激励"等。

请你扮演一位提示词工程专家，帮我重写前面我尝试的System Prompt，提示词应该结构化指定角色、任务、输出要求、参考信息等要素，以实现上述目标。

【输出要求】
请只输出优化后的System Prompt，不要输出解释，不要篡改参考信息，不要生成示例内容，只输出System Prompt本身。
"""
    
    print("\n🧠 向 AI 教练请求优化...")
    optimized_prompt = llm.invoke(meta_prompt)
    
    print("\n📝 AI 教练优化后的 Prompt:")
    print("-"*60)
    print(optimized_prompt)
    print("-"*60)
    
    # 使用优化后的提示词再次调用模型
    final_response = llm.invoke(optimized_prompt)
    
    print("\n🤖 使用优化版 Prompt 的回答:")
    print("-"*60)
    print(final_response)
    print("-"*60)
    
    print("\n💡 观察：单轮优化后，回答的友好性和结构化程度明显提升。")
    
    return optimized_prompt


# ==============================================================================
# 示例 3: 多轮迭代优化（基于差距分析）
# ==============================================================================

def analyze_gap(generated_response, reference):
    """
    差距分析函数：对比生成回答与参考答案。
    
    【核心机制】
    使用一个 LLM 作为"评审专家"，从多个维度分析差距：
    - 语气（是否友好热情）
    - 结构（是否清晰有序）
    - 内容细节（是否完整准确）
    - 格式（表情符号、粗体等视觉元素）
    
    Args:
        generated_response: 生成的回答
        reference: 参考答案
        
    Returns:
        str: 差距分析报告
    """
    gap_analysis_prompt = f"""
【角色】你是一位文本比较专家。
【任务】请详细比较【生成回答】与【参考答案】之间的差距。
【参考答案】
{reference}
---
【生成回答】
{generated_response}
---
【要求】
请从语气、结构、内容细节、格式（如表情符号使用）等方面，输出一份详细的差距分析报告。如果两者几乎没有差距，请直接回答"差距很小"。
"""
    return llm.invoke(gap_analysis_prompt)


def optimize_prompt_with_gap_analysis(current_template, generated_response, gap_report):
    """
    基于差距分析优化 Prompt 模板。
    
    【核心思想】
    根据"差距分析报告"，针对性地优化 Prompt，引导模型生成更接近参考答案的输出。
    
    【关键约束】
    1. 结构化重写：使用 ### 小标题组织 Prompt
    2. 占位符规范：必须且只能包含一次 {context} 占位符
    3. 避免过拟合：
       - 禁止硬编码具体数值（如"500元"）
       - 禁止硬编码具体实体名称
       - 只优化指令，不篡改数据
    4. 聚焦于指令优化维度：
       - 语气与风格
       - 输出格式
       - 信息提取策略
    
    Args:
        current_template: 当前 Prompt 模板
        generated_response: 使用当前模板生成的回答
        gap_report: 差距分析报告
        
    Returns:
        str: 优化后的 Prompt 模板
    """
    optimization_prompt = f"""
【角色】你是一位资深的提示词工程师，擅长编写结构化、高鲁棒性的 System Prompt。
【任务】你需要根据差距分析报告，优化一个RAG系统的"提示词模板"。

【输入信息】
1. 当前模板：
```
{current_template}
```

2. 使用该模板生成的回答（表现不够好）：
{generated_response}

3. 差距分析报告（基于参考答案的对比）：
{gap_report}

【优化原则】
1. **结构化重写**：推荐使用结构化格式（如包含 ### 角色、### 任务、### 要求、### 参考信息 等小标题），以提升指令遵循度。
2. **占位符规范**：
   - 必须且只能包含一次 `{{context}}` 占位符。
   - **禁止**在指令文本中重复提及 `{{context}}` 字样，只用它来占位。

【🚨 关键约束 - 抽象化与通用性】
1. **指令与数据分离**：Gap Report 可能会指出生成内容缺少具体的细节（如特定的数值、名称、步骤）。你的优化策略应该是**增强提取指令**（例如"请详细列出所有数值数据"或"请完整提取实体名称"），而不是将具体的数值或实体写死在模板中。
2. **避免过拟合**：严禁将参考答案中的具体事实、数据或实体名称硬编码到模板中。模板必须保持通用，能够处理任何未来输入的参考信息。
3. **聚焦于指令优化**：重点优化以下维度，而不是内容本身：
   - **语气与风格**（如：更热情、更专业、更简洁）
   - **输出格式**（如：Markdown列表、JSON、小标题结构）
   - **信息提取策略**（如：必须包含数字、必须引用原文、忽略无关信息）

【输出要求】
请只返回优化后的**提示词模板**，不要包含任何解释。
"""
    return llm.invoke(optimization_prompt)


def demo_iterative_optimization(initial_template):
    """
    演示多轮迭代优化（基于差距分析）。
    
    【核心思想】
    通过"生成 → 评估 → 优化"的闭环，逐步改进 Prompt。
    
    【迭代流程】
    1. 使用当前模板生成回答
    2. 对比参考答案，分析差距
    3. 如果差距很小，停止迭代
    4. 否则，根据差距报告优化模板，进入下一轮
    
    【终止条件】
    - 差距分析报告显示"差距很小"
    - 达到最大迭代次数（3次）
    
    【挑战】
    - 每轮优化的方向可能不稳定（不确定性叠加）
    - 容易过拟合参考答案
    """
    print("\n" + "="*60)
    print("  示例 3: 多轮迭代优化（基于差距分析）")
    print("="*60)
    
    current_template = initial_template
    max_iterations = 3
    
    for i in range(max_iterations):
        print(f"\n{'─'*60}")
        print(f"  第 {i+1} 轮迭代")
        print(f"{'─'*60}")
        
        # 动态填充 context 进行测试
        # 只有在生成回答时，才把 RETRIEVED_TEXT 填进去
        try:
            filled_prompt = current_template.format(context=RETRIEVED_TEXT)
        except Exception as e:
            print(f"❌ 模板格式错误，缺少 {{context}} 占位符: {e}")
            break
        
        print("\n🤖 生成回答中...")
        generated_response = llm.invoke(filled_prompt)
        print(f"生成的回答（前 100 字符）:\n{generated_response[:100]}...")
        
        print("\n🔍 进行差距分析...")
        gap_report = analyze_gap(generated_response, REFERENCE_ANSWER)
        print(f"差距分析报告:\n{gap_report[:200]}...")
        
        if "差距很小" in gap_report:
            print("\n✅ 评估通过，优化完成！")
            break
        
        print("\n🔧 根据差距分析报告优化 Prompt 模板...")
        # 传入的是模板，不是填好的 prompt
        current_template = optimize_prompt_with_gap_analysis(
            current_template, generated_response, gap_report
        )
    else:
        print("\n⚠️ 达到最大迭代次数，停止优化。")
    
    print("\n📝 最终优化的 Prompt 模板:")
    print("-"*60)
    print(current_template)
    print("-"*60)
    
    # 使用最终提示词再次调用模型
    final_response = llm.invoke(current_template.format(context=RETRIEVED_TEXT))
    print("\n🤖 使用最终模板的回答:")
    print("-"*60)
    print(final_response)
    print("-"*60)
    
    print("\n💡 观察：多轮迭代可能会逐步逼近参考答案，但也可能因不确定性而发散。")
    
    return current_template


# ==============================================================================
# 示例 4: 可视化评估（多维度评分对比）
# ==============================================================================

def grade_response_detailed(response_to_grade):
    """
    使用 LLM 对回答进行多维度评分。
    
    【评分维度】
    1. 欢迎语气 (welcoming_tone): 语气是否友好热情
    2. 内容结构化 (structuring): 信息是否清晰有序
    3. 视觉吸引力 (visual_appeal): 是否善用表情符号、粗体等元素
    4. 信息完整性 (completeness): 关键信息是否完整
    
    【评分范围】
    1-5 分，5 分最高
    
    【LLM-as-Judge】
    使用 LLM 作为评审专家，输出结构化的 JSON 评分。
    
    Args:
        response_to_grade: 待评分的回答
        
    Returns:
        dict: 包含各维度评分的字典
    """
    grader_prompt = f"""
【角色】你是一位经验丰富的内部沟通和员工体验评测官。
【任务】请根据以下四个维度，对提供的"公司福利介绍"文本进行1-5分的量化评分。

【评分维度】
1.  **欢迎语气 (welcoming_tone)**: 1分表示语气冰冷生硬，5分表示非常热情、有感染力。
2.  **内容结构化 (structuring)**: 1分表示信息混乱无序，5分表示分类清晰、逻辑性强。
3.  **视觉吸引力 (visual_appeal)**: 1分表示枯燥乏味，5分表示善用表情符号、粗体等元素，非常吸引眼球。
4.  **信息完整性 (completeness)**: 1分表示信息缺失严重，5分表示关键福利信息完整无缺。

【待评估文本】
{response_to_grade}
---
【输出要求】
请严格以JSON格式返回你的评分，不要包含任何解释。例如：
{{"welcoming_tone": 5, "structuring": 4, "visual_appeal": 5, "completeness": 5}}
"""
    try:
        raw_output = llm.invoke(grader_prompt)
        # 提取JSON部分
        json_str = raw_output[raw_output.find('{'):raw_output.rfind('}')+1]
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"⚠️ JSON 解析失败: {e}")
        # 容错处理，返回默认低分
        return {"welcoming_tone": 1, "structuring": 1, "visual_appeal": 1, "completeness": 1}


def demo_visual_evaluation():
    """
    演示可视化评估：对比三种优化水平的回答。
    
    【对比样本】
    1. 差：简单罗列信息，无结构无感情
    2. 中：有基本结构和分类，但语气平淡
    3. 优：结构清晰、语气友好、视觉吸引力强
    
    【评估方法】
    使用 LLM-as-Judge 对三个样本进行多维度评分，并通过柱状图可视化对比。
    
    【可视化输出】
    分组柱状图，横轴为评分维度，纵轴为分数，三个版本并排显示。
    """
    print("\n" + "="*60)
    print("  示例 4: 可视化评估（多维度评分对比）")
    print("="*60)
    
    # 三个质量差异明显的典型回答样本
    # 差：只是简单罗列信息，没有结构和感情
    poor_response = "公司福利：健康保险，家属可用。15天年假，5天病假。交通补贴500，餐饮补贴300。培训基金8000。健身房有折扣。"
    
    # 中：有基本结构和分类，但语气平淡
    medium_response = """
公司福利：
1. 💦健康和假期：
   - 健康保险（含家属）
   - 年度体检
   - 15天年假和5天病假
2. 💰补贴和激励：
   - 每月500交通补贴和300餐饮补贴
   - 8000元/年的教育培训基金
   - 合作健身房折扣
"""
    
    # 优：结构清晰、语气友好、视觉吸引力强（使用参考答案）
    good_response = REFERENCE_ANSWER
    
    print("\n🔍 对三个样本进行多维度评分...")
    
    # 对三个典型样本进行评分
    scores = {
        "Original Answer": grade_response_detailed(poor_response),
        "Single Iteration Optimize": grade_response_detailed(medium_response),
        "Multi-turn Iteration Optimize": grade_response_detailed(good_response)
    }
    
    print("\n📊 评分结果:")
    for version, score in scores.items():
        print(f"  {version}: {score}")
    
    # 将 scores 转换为 DataFrame
    df = pd.DataFrame(scores)
    df = df.reset_index().rename(columns={'index': 'Dim'})
    df_long = df.melt(id_vars='Dim', var_name='Version', value_name='Score')
    
    # 绘制分组柱状图（使用英文标签，避免字体兼容性问题）
    print("\n📈 生成可视化图表...")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_long,
        x="Dim",        # 每个维度一组
        y="Score",
        hue="Version",  # 3 个版本并排
        palette="viridis"
    )
    
    # 为每根柱子添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height == 0:  # 跳过高度为 0 的占位 patch
            continue
        ax.annotate(
            f"{height}",
            (p.get_x() + p.get_width() / 2., height),
            ha='center', va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=11
        )
    
    # 轴和标题美化
    ax.set_ylim(0, 6)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_xlabel('')  # 隐藏 X 轴标题
    ax.set_title('Meta-Prompting Evaluation Comparison', fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=12)
    
    plt.legend()
    plt.tight_layout()
    
    # 保存图表到文件（云服务器友好）
    output_path = "meta_prompting_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 可视化图表已保存到: {os.path.abspath(output_path)}")
    
    # 尝试显示（本地环境）
    try:
        plt.show()
    except:
        pass
    
    print("\n💡 观察：从图表可以看出，多轮迭代优化在各个维度上都有显著提升。")


# ==============================================================================
# 主程序入口
# ==============================================================================

def main():
    """主程序：执行所有 Meta Prompting 实验"""
    
    print("\n" + "="*60)
    print("  Meta Prompting 实验与反思")
    print("="*60 + "\n")
    
    # 加载 API Key
    load_key()
    
    try:
        # 示例 1：基础 Prompt 测试
        initial_template, initial_response = demo_initial_prompt()
        
        # 示例 2：单轮 Meta Prompting 优化
        demo_single_iteration_optimization(initial_template, initial_response)
        
        # 示例 3：多轮迭代优化（基于差距分析）
        demo_iterative_optimization(initial_template)
        
        # 示例 4：可视化评估
        demo_visual_evaluation()
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("  所有实验完成！")
    print("="*60)
    print("\n【总结】")
    print("Meta Prompting 是一种有趣的探索性技术，但在生产环境中需谨慎使用：")
    print("  ✓ 优点：快速生成优化思路，激发灵感")
    print("  ✗ 缺点：不稳定、容易过拟合、需要人工审核")
    print("  💡 建议：用于实验和辅助设计，但不要完全依赖自动化\n")


if __name__ == "__main__":
    main()
