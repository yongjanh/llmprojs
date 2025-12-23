# ==============================================================================
# 说明：本文件涉及 Reasoning LLM（推理模型）的调用与应用探索
# ==============================================================================
# Reasoning LLM 是新一代具有"深度思考"能力的大语言模型。
#
# 【核心概念】
# Reasoning LLM (如 Qwen-Thinking, OpenAI o1, DeepSeek-R1) 的特点是在输出最终答案前，
# 会先进行隐式的、长链条的思维推理 (Chain of Thought)，从而显著提升处理复杂逻辑、
# 代码生成、数学计算等任务的能力。
#
# 【与普通 LLM 的区别】
# ┌─────────────────┬─────────────────────┬─────────────────────┐
# │ 维度            │ 普通 LLM            │ Reasoning LLM       │
# ├─────────────────┼─────────────────────┼─────────────────────┤
# │ 推理过程        │ 隐式（不可见）      │ 显式（可见）        │
# │ 思考深度        │ 浅层、快速          │ 深层、缓慢          │
# │ 复杂任务        │ 容易出错            │ 更加可靠            │
# │ 响应速度        │ 快                  │ 慢（需思考时间）    │
# │ 适用场景        │ 简单对话、文案生成  │ 数学、代码、逻辑    │
# └─────────────────┴─────────────────────┴─────────────────────┘
#
# 【本脚本演示内容】
# 1. 基础调用：如何调用支持思考过程的推理模型
# 2. 流式解析：如何解析并打印模型的思考过程和最终回复
# 3. Prompt 技巧：
#    - 技巧 1：提供清晰的背景信息和任务目标
#    - 技巧 2：结构化输入（XML 标签风格）
#    - 技巧 3：避免手动 CoT（模型自带推理能力）
# 4. 高级应用：使用 Reasoning LLM 作为 "Prompt Coach"（提示词教练）
#
# 【关键优势】
# ✓ 可见的思考过程：便于理解和调试
# ✓ 更高的准确率：复杂任务表现更佳
# ✓ 自我纠错能力：推理过程中发现并修正错误
#
# 【注意事项】
# ⚠️ 成本更高：推理过程消耗更多 tokens
# ⚠️ 响应更慢：需要额外的思考时间
# ⚠️ 适用性：并非所有任务都需要深度推理
# ==============================================================================

from openai import OpenAI
import os
from config.load_key import load_key


# ==============================================================================
# 核心函数：调用 Reasoning LLM
# ==============================================================================

def call_reasoning_model(user_prompt, system_prompt="你是一个编程助手。", 
                         model="qwen3-235b-a22b-thinking-2507", 
                         show_thinking=True):
    """
    调用 Reasoning LLM 并解析思考过程与最终回答。
    
    【核心机制】
    Reasoning LLM 的响应包含两部分：
    1. reasoning_content: 模型的思考过程（可选显示）
    2. content: 最终的回答
    
    【流式解析】
    通过 stream=True 实现实时输出，用户可以看到模型"思考"的过程。
    
    【支持的模型】
    - qwen3-235b-a22b-thinking-2507 (Qwen-Thinking)
    - deepseek-r1 (DeepSeek Reasoning)
    - o1 系列 (OpenAI)
    
    Args:
        user_prompt: 用户输入的提示词
        system_prompt: 系统提示词（定义角色）
        model: 推理模型名称
        show_thinking: 是否显示思考过程
        
    Returns:
        tuple: (思考过程文本, 最终回答文本)
    """
    # 初始化客户端
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 初始化状态变量
    is_answering = False
    thinking_content = ""
    answer_content = ""
    
    # 发起流式请求
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
        )
        
        # 打印思考过程标题
        if show_thinking:
            print("\n" + "="*20 + " 思考过程 " + "="*20 + "\n")
        
        # 处理流式响应
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                
                # 处理思考过程内容
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    thinking_content += delta.reasoning_content
                    if show_thinking:
                        print(delta.reasoning_content, end='', flush=True)
                else:
                    # 切换到答案输出模式
                    if delta.content and not is_answering:
                        print("\n" + "="*20 + " 完整回复 " + "="*20 + "\n")
                        is_answering = True
                    
                    # 处理答案内容
                    if delta.content:
                        answer_content += delta.content
                        print(delta.content, end='', flush=True)
        
        print("\n")  # 结束后换行
        return thinking_content, answer_content
        
    except Exception as e:
        print(f"\n❌ 调用模型出错: {e}")
        return "", ""


# ==============================================================================
# 示例 1: 基础调用
# ==============================================================================

def demo_basic_call():
    """
    演示最基础的 Reasoning LLM 调用。
    
    【核心思想】
    观察推理模型如何"思考"一个简单问题。
    
    【对比】
    - 普通 LLM：直接输出答案
    - Reasoning LLM：先思考，再输出答案
    """
    print("\n" + "="*60)
    print("  示例 1: 基础调用")
    print("="*60)
    
    print("\n💬 用户提问: 你是谁？")
    
    call_reasoning_model(user_prompt="你是谁？")
    
    print("\n💡 观察：推理模型会先思考如何回答这个问题，然后给出答案。")


# ==============================================================================
# 示例 2: Prompt 技巧对比
# ==============================================================================

def demo_prompt_techniques():
    """
    演示不同 Prompt 技巧对推理模型输出的影响。
    
    【对比三种 Prompt】
    1. 差：只给代码，无背景，无任务说明
    2. 中：有任务说明，但结构松散
    3. 优：结构化输入（XML 标签），背景清晰
    
    【核心建议】
    对于 Reasoning LLM，结构化的 Prompt 能帮助模型更好地组织思考过程。
    """
    print("\n" + "="*60)
    print("  示例 2: Prompt 技巧对比")
    print("="*60)
    
    # 差：只给代码，无背景
    print("\n【❌ 差的 Prompt：无背景信息】")
    print("-"*60)
    bad_prompt = """
def example(a):
  b = []
  for i in range(len(a)):
    b.append(a[i]*2)
  return sum(b)
"""
    print(f"Prompt:\n{bad_prompt}")
    call_reasoning_model(user_prompt=bad_prompt, show_thinking=False)
    
    # 中：有任务目标，但结构松散
    print("\n【✓ 中等的 Prompt：有任务说明】")
    print("-"*60)
    medium_prompt = """
如下 python 代码有什么问题？怎么优化？
def example(a):
  b = []
  for i in range(len(a)):
    b.append(a[i]*2)
  return sum(b)
"""
    print(f"Prompt:\n{medium_prompt}")
    call_reasoning_model(user_prompt=medium_prompt, show_thinking=False)
    
    # 优：结构化输入（XML 标签风格）
    print("\n【✅ 优秀的 Prompt：结构化输入】")
    print("-"*60)
    good_prompt = """
<audience>初级Python开发者</audience>

<task>函数性能优化，优化 code 中的代码。</task>

<format>
如有多种优化方案请按照如下格式进行输出：
【优化方案X】
问题描述：[描述]
优化方案：[描述]
示例代码：[代码块]
</format>

<code>
def example(a):
  b = []
  for i in range(len(a)):
    b.append(a[i]*2)
  return sum(b)
</code>
"""
    print(f"Prompt:\n{good_prompt}")
    call_reasoning_model(user_prompt=good_prompt, show_thinking=False)
    
    print("\n💡 观察：结构化的 Prompt 能获得更清晰、更有针对性的回答。")


# ==============================================================================
# 示例 3: Reasoning LLM 作为 Prompt Coach
# 最适合进行metaprompting优化的模型是推理模型
# ==============================================================================

def demo_prompt_coach():
    """
    演示使用 Reasoning LLM 作为 "Prompt Coach" 来优化普通模型的提示词。
    
    【核心思想】
    利用推理模型的深度思考能力，帮助我们设计更好的 Prompt。
    
    【应用场景】
    - 当你不知道如何优化 Prompt 时
    - 当你需要专家级的 Prompt 设计建议时
    - 当你想快速探索多种 Prompt 优化方向时
    
    【优势】
    相比普通 LLM，Reasoning LLM 能：
    1. 更深入地分析 Prompt 的问题
    2. 考虑多种优化角度
    3. 给出更合理的优化建议
    
    【与 Meta Prompting 的关系】
    这是 Meta Prompting 的一个变体：用"更聪明的模型"来优化"普通模型"的 Prompt。
    """
    print("\n" + "="*60)
    print("  示例 3: Reasoning LLM 作为 Prompt Coach")
    print("="*60)
    
    # 复用公司福利场景
    retrieved_text = """
关于公司的福利政策，我们提供全面的健康保险，覆盖员工及其直系家属。
年度体检是标配。此外，每年有15天的带薪年假，以及5天的带薪病假。
我们还提供每月500元的交通补贴和300元的餐饮补贴。
为了鼓励员工成长，公司设有每年高达8000元的教育培训基金，员工可以申请用于课程学习或购买专业书籍。
健身方面，公司与多家健身房有合作，员工可享受折扣价。
"""
    
    # 初始的、比较简单的提示词
    initial_prompt = f"""
根据以下信息，回答新员工关于公司福利的问题。

【参考信息】
{retrieved_text}
"""
    
    # 假设这是普通模型的、不甚理想的输出
    initial_response = """
我们公司提供全面的健康保险，覆盖员工及其家属。每年有15天带薪年假和5天病假。还有每月500元交通补贴和300元餐饮补贴。公司提供8000元的年度教育基金，并与健身房有合作折扣。
"""
    
    print("\n📝 初始 Prompt:")
    print("-"*60)
    print(initial_prompt)
    print("-"*60)
    
    print("\n🤖 初始回答（不够好）:")
    print("-"*60)
    print(initial_response)
    print("-"*60)
    
    print("\n🎓 现在请 Reasoning LLM 担任 Prompt Coach，帮我们优化...")
    
    # 构建 Meta Prompt，请求推理模型帮助优化
    meta_prompt = f"""
我正在为公司的新员工答疑机器人优化一个提示词，目标是回答关于"公司福利"的问题。

这是我的第一个尝试：
---
{initial_prompt}
---

这是它生成的输出：
---
{initial_response}
---

这个输出不够好。我希望机器人的回答更具吸引力，并且结构清晰，能让新员工快速抓住重点。具体要求如下：
1. **语气**：友好、热情，有欢迎新同事的感觉。
2. **结构**：使用清晰的要点（比如用表情符号开头的列表）来组织内容。
3. **内容**：将福利分为几个类别，如"健康与假期"、"补贴与激励"等。

请你扮演一位提示词工程专家，帮我重写这个提示词，以实现上述目标。
请在最终答案中，只给出优化后的提示词本身，不要包含其他解释性文字。
"""
    
    call_reasoning_model(
        user_prompt=meta_prompt,
        system_prompt="你是一位顶级的提示词工程专家。",
        show_thinking=True
    )
    
    print("\n💡 观察：Reasoning LLM 会深入分析问题，并给出经过深思熟虑的优化建议。")


# ==============================================================================
# 主程序入口
# ==============================================================================

def main():
    """主程序：执行所有 Reasoning LLM 实验"""
    
    print("\n" + "="*60)
    print("  Reasoning LLM 调用与应用探索")
    print("="*60)
    
    # 加载 API Key
    load_key()
    
    # 检查 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n❌ 请先设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    try:
        # 示例 1：基础调用
        demo_basic_call()
        
        # 示例 2：Prompt 技巧对比
        demo_prompt_techniques()
        
        # 示例 3：Reasoning LLM 作为 Prompt Coach
        demo_prompt_coach()
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("  所有实验完成！")
    print("="*60)
    print("\n【总结】")
    print("Reasoning LLM 通过显式的思考过程，在复杂任务上表现更佳。")
    print("\n【关键要点】")
    print("  ✓ 适用场景：数学、代码、逻辑推理等复杂任务")
    print("  ✓ Prompt 技巧：结构化输入（XML 标签）效果更好")
    print("  ✓ 高级应用：作为 Prompt Coach 辅助优化")
    print("\n【注意事项】")
    print("  ⚠️ 成本更高：推理过程消耗更多 tokens")
    print("  ⚠️ 响应更慢：需要额外的思考时间")
    print("  ⚠️ 并非万能：简单任务用普通 LLM 即可\n")


if __name__ == "__main__":
    main()
