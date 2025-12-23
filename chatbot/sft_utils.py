# ==============================================================================
# 说明：SFT（监督微调）共用工具函数
# ==============================================================================
# 本模块提供微调和评测过程中共用的工具函数，包括：
# - 设备检测（MPS/CUDA/CPU）
# - 模型初始化
# - 单次推理
# - 基准测试
# - 训练曲线可视化
# ==============================================================================

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无界面模式，适合服务器和脚本运行


def detect_device():
    """
    自动检测最佳可用设备。
    
    【功能】
    按优先级检测：MPS (Mac GPU) > CUDA (NVIDIA GPU) > CPU
    
    【返回】
    - device: torch.device 对象
    - device_name: 设备名称字符串（用于显示）
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA (NVIDIA GPU: {torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3) 的 Metal Performance Shaders
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    return device, device_name


def initialize_model(model_path="./model"):
    """
    初始化 Qwen2.5-1.5B-Instruct 基座模型。
    
    【功能】
    加载本地模型和 tokenizer，自动检测并配置最佳设备（MPS/CUDA/CPU）。
    基于 transformers 库，适配 PEFT 框架。
    
    【参数】
    - model_path: 本地模型文件路径
    
    【返回】
    - llm: 加载的语言模型
    - tokenizer: 对应的 tokenizer
    - template: None（为了兼容性保留，PEFT 不需要 template）
    - device: 使用的设备
    """
    print("\n" + "="*80)
    print("📦 模型初始化")
    print("="*80)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在于 {model_path}")
        print("请先下载模型：")
        print(f"  mkdir -p {model_path}")
        print(f"  modelscope download --model qwen/Qwen2.5-1.5B-Instruct --local_dir '{model_path}'")
        return None, None, None, None
    
    # 检测最佳设备
    device, device_name = detect_device()
    print(f"🔍 检测到设备: {device_name}")
    
    print(f"🔍 正在从 {model_path} 加载模型...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    llm.eval()  # 评估模式
    
    # 配置生成参数
    llm.generation_config.max_new_tokens = 256
    llm.generation_config.do_sample = False  # 贪心解码，结果确定性
    
    print("✅ 模型初始化完成")
    print(f"   - 模型路径: {model_path}")
    print(f"   - 设备: {device_name}")
    print(f"   - 最大生成长度: {llm.generation_config.max_new_tokens} tokens")
    
    if device.type == "mps":
        print(f"   - 💡 提示: 使用 Apple Silicon GPU 加速，推理速度比 CPU 快 5-10 倍")
    
    return llm, tokenizer, None, device  # template 返回 None（PEFT 不需要）


def run_single_query(llm, tokenizer, template, query):
    """
    运行单个查询的推理。
    
    【功能】
    使用模型对单个问题进行推理，并返回回答。
    使用 transformers 标准 API，兼容 PEFT 框架。
    
    【参数】
    - llm: 语言模型
    - tokenizer: tokenizer
    - template: 格式化模板（兼容性参数，实际不使用）
    - query: 用户问题
    
    【返回】
    - response: 模型的回答文本
    """
    # 构建 chat 格式的输入
    messages = [{'role': 'user', 'content': query}]
    
    # 使用 tokenizer.apply_chat_template 构建输入（Qwen2.5 原生支持）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        output_ids = llm.generate(
            input_ids=inputs['input_ids'].to(llm.device),
            attention_mask=inputs['attention_mask'].to(llm.device),
            max_new_tokens=llm.generation_config.max_new_tokens,
            do_sample=False,  # 贪心解码，结果更稳定
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出（只取生成的部分）
    input_len = inputs['input_ids'].shape[1]  # 注意是 shape[1]（序列长度）
    response_ids = output_ids[0][input_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response


def score_answer(response, ans, ans_core, support_partial=True):
    """
    统一的答案评分函数。
    
    【功能】
    对模型回答进行评分，支持完全正确、部分正确、错误三级评分。
    
    【评分规则】
    - 完全正确：完整答案（含格式）在回答中 → 1.0 分
    - 部分正确：核心答案在回答中但格式不对 → 0.5 分（如果支持）
    - 错误：答案不在回答中 → 0.0 分
    
    【参数】
    - response: 模型的回答文本
    - ans: 完整标准答案（含格式，如 "{ans: \"42\"}"）
    - ans_core: 核心答案（不含格式，如 "42"）
    - support_partial: 是否支持部分正确评分（默认True）
    
    【返回】
    - score: 分数（1.0 / 0.5 / 0.0）
    - label: 评分标签（"完全正确" / "部分正确" / "错误"）
    """
    if ans in response:
        return 1.0, "✅ 完全正确"
    elif support_partial and ans_core in response:
        return 0.5, "⚠️  部分正确（格式有误）"
    else:
        return 0.0, "❌ 错误"


def run_model_evaluation(
    model, 
    tokenizer, 
    template, 
    test_file="./resources/test.jsonl",
    max_samples=None,
    support_partial=True,
    model_name="模型"
):
    """
    通用的模型评估函数，适用于基座模型、LoRA模型、合并模型。
    
    【功能】
    统一处理所有类型模型的评估，确保评分逻辑完全一致。
    
    【测试数据格式】
    每行是一个 JSON 对象，包含：
    - messages[1].content: 用户问题（格式：#数学题#\n{具体问题}）
    - messages[2].content: 标准答案（格式：...{ans: "答案"}...）
    
    【评分规则】
    使用统一的 score_answer() 函数：
    - 完全正确：+1.0 分
    - 部分正确：+0.5 分（如果 support_partial=True）
    - 错误：0 分
    
    【参数】
    - model: 任意模型（基座/LoRA/合并）
    - tokenizer: tokenizer
    - template: 格式化模板
    - test_file: 测试集文件路径
    - max_samples: 最多测试的样本数（None 表示全部）
    - support_partial: 是否支持部分正确评分（默认True）
    - model_name: 模型名称（用于显示）
    
    【返回】
    - accuracy: 加权准确率（0-100）
    - results: 详细结果列表 [{"question": str, "answer": str, "response": str, 
                                "score": float, "label": str}]
    """
    print("\n" + "="*80)
    print(f"📊 {model_name}评估")
    print("="*80)
    
    if not os.path.exists(test_file):
        print(f"❌ 错误：测试文件不存在于 {test_file}")
        return 0.0, []
    
    print(f"📝 测试文件: {test_file}")
    if max_samples:
        print(f"📝 最多测试样本数: {max_samples}")
    
    total_score = 0.0
    total_count = 0
    results = []
    
    # 统计数量
    full_correct = 0  # 完全正确数
    partial_correct = 0  # 部分正确数
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if max_samples and total_count >= max_samples:
                break
                
            # 解析测试样本
            math_question = json.loads(line)
            query = math_question["messages"][1]["content"]
            
            # 提取问题文本（去掉 #数学题# 标记）
            if "#数学题#\n" in query:
                question_text = query.split("#数学题#\n")[1]
            else:
                question_text = query
            
            # 模型推理
            response = run_single_query(model, tokenizer, template, query)
            
            # 提取标准答案
            ans_full = math_question["messages"][2]["content"]
            pos = ans_full.find("ans")
            if pos != -1:
                end_pos = ans_full[pos:].find('}}')
                ans = ans_full[pos - 2: end_pos + pos + 2]  # 提取 {ans: "xxx"} 格式
                ans_core = ans[6:-2]  # 提取 xxx 部分
            else:
                ans = ans_full
                ans_core = ans_full
            
            # 统一评分
            score, label = score_answer(response, ans, ans_core, support_partial)
            total_score += score
            total_count += 1
            
            # 统计
            if score == 1.0:
                full_correct += 1
            elif score == 0.5:
                partial_correct += 1
            
            # 保存详细结果
            results.append({
                "question": question_text,
                "answer": ans_core,
                "response": response,
                "score": score,
                "label": label
            })
            
            # 打印详细结果
            print(f"\n{'='*80}")
            print(f"问题 {total_count}/{max_samples if max_samples else '?'}: {question_text}")
            print(f"标准答案: {ans_core}")
            print(f"----- {model_name}回答 -----")
            print(response)
            print(f"----- 评分 -----")
            print(label)
    
    # 计算准确率
    accuracy = 100.0 * total_score / total_count if total_count > 0 else 0.0
    
    print("\n" + "="*80)
    print(f"🎯 {model_name}测试结果：")
    print(f"   - 完全正确: {full_correct}/{total_count} ({100.0 * full_correct / total_count:.1f}%)")
    if support_partial:
        print(f"   - 部分正确: {partial_correct}/{total_count} ({100.0 * partial_correct / total_count:.1f}%)")
        print(f"   - 加权准确率: {accuracy:.1f}%")
    else:
        print(f"   - 准确率: {accuracy:.1f}%")
    print("="*80)
    
    return accuracy, results


def run_benchmark_test(llm, tokenizer, template, test_file="./resources/test.jsonl", max_samples=None):
    """
    基准测试（兼容性包装函数）。
    
    【功能】
    为了保持向后兼容，包装 run_model_evaluation 函数。
    
    【注意】
    建议直接使用 run_model_evaluation() 以获得更详细的结果。
    
    【参数】
    - llm: 语言模型
    - tokenizer: tokenizer
    - template: 格式化模板
    - test_file: 测试集文件路径
    - max_samples: 最多测试的样本数（None 表示全部）
    
    【返回】
    - accuracy: 准确率（0-100）
    """
    accuracy, _ = run_model_evaluation(
        llm, tokenizer, template,
        test_file=test_file,
        max_samples=max_samples,
        support_partial=True,
        model_name="基座模型"
    )
    return accuracy


def plot_training_curves(checkpoint_path, output_dir="./output", output_filename="training_loss_curve.png"):
    """
    绘制训练过程中的 loss 曲线。
    
    【功能】
    从 checkpoint 目录中读取训练日志，绘制训练 loss 和评估 loss 的变化趋势。
    
    【参数】
    - checkpoint_path: checkpoint 路径
    - output_dir: 图表保存目录
    - output_filename: 输出文件名（默认: training_loss_curve.png）
    
    【返回】
    - success: 是否成功绘制
    """
    print("\n" + "="*80)
    print("📈 生成训练 Loss 曲线")
    print("="*80)
    
    try:
        # 查找 trainer_state.json 文件
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
        if not os.path.exists(trainer_state_path):
            # 尝试在父目录查找
            parent_dir = os.path.dirname(checkpoint_path)
            trainer_state_path = os.path.join(parent_dir, "trainer_state.json")
            
        if not os.path.exists(trainer_state_path):
            print(f"⚠️  未找到训练日志文件: trainer_state.json")
            return False
        
        # 读取训练日志
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # 提取 loss 数据
        log_history = trainer_state.get('log_history', [])
        if not log_history:
            print(f"⚠️  训练日志为空")
            return False
        
        # 分离训练和评估数据
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_losses = []
        final_train_loss = None
        final_step = 0
        
        for entry in log_history:
            # 中间步骤的训练 loss（由 logging_steps 控制）
            if 'loss' in entry and 'train_loss' not in entry:
                train_steps.append(entry.get('step', 0))
                train_losses.append(entry['loss'])
            # 评估 loss
            if 'eval_loss' in entry:
                eval_steps.append(entry.get('step', 0))
                eval_losses.append(entry['eval_loss'])
            # 最终训练 loss（训练结束时的平均值）
            if 'train_loss' in entry:
                final_train_loss = entry['train_loss']
                final_step = entry.get('step', 0)
        
        if not train_losses and not eval_losses:
            print(f"⚠️  未找到 loss 数据")
            return False
        
        # 设置中文字体（Mac 使用 Arial Unicode MS）
        if os.uname().sysname == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制训练 loss
        if train_losses:
            ax.plot(train_steps, train_losses, label='训练 Loss (中间步骤)', 
                   color='#1f77b4', linewidth=2, marker='o', markersize=6)
        
        # 添加最终训练 loss 的标注（整体平均值）
        if final_train_loss is not None:
            ax.scatter([final_step], [final_train_loss], 
                      color='#d62728', marker='*', s=200, zorder=5,
                      label=f'最终训练 Loss (平均)', edgecolors='black', linewidths=1)
        
        # 绘制评估 loss
        if eval_losses:
            ax.plot(eval_steps, eval_losses, label='评估 Loss', 
                   color='#ff7f0e', linewidth=2, marker='s', markersize=4)
        
        # 设置标题和标签
        ax.set_xlabel('训练步数 (Steps)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('训练过程 Loss 变化曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 添加信息文本
        info_text_lines = []
        if final_train_loss is not None:
            info_text_lines.append(f'最终训练 Loss (平均): {final_train_loss:.4f}')
        if eval_losses:
            info_text_lines.append(f'最终评估 Loss: {eval_losses[-1]:.4f}')
        
        if info_text_lines:
            info_text = '\n'.join(info_text_lines)
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, output_filename)
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Loss 曲线已保存至: {os.path.abspath(plot_filename)}")
        
        # 打印统计信息
        print(f"\n📊 训练统计:")
        if train_losses:
            print(f"   - 训练 Loss (中间): {len(train_losses)} 个记录点")
            print(f"     第一个: {train_losses[0]:.4f} (step {train_steps[0]})")
            print(f"     最后一个: {train_losses[-1]:.4f} (step {train_steps[-1]})")
        if final_train_loss is not None:
            print(f"   - 训练 Loss (最终平均): {final_train_loss:.4f}")
        if eval_losses:
            print(f"   - 评估 Loss: {eval_losses[0]:.4f} → {eval_losses[-1]:.4f} "
                  f"({len(eval_losses)} 个记录点, 下降 {(eval_losses[0] - eval_losses[-1]) / eval_losses[0] * 100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 绘制 Loss 曲线失败: {str(e)}")
        return False

