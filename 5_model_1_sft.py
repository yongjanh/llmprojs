# ==============================================================================
# 说明：本地模型监督微调（SFT - Supervised Fine-Tuning）
# ==============================================================================
# 基于 PEFT + Transformers 框架对 Qwen2.5-1.5B-Instruct 模型进行 LoRA 微调，
# 针对特定数学问答任务优化模型性能。
#
# 【核心概念】
# 1. **预训练（Pre-training）**：
#    - 目的：在大规模无标注数据上学习语言的通用表征能力
#    - 方法：自监督学习（如 Next Token Prediction）
#    - 结果：得到具备基础语言理解和生成能力的基座模型
#    - 特点：计算量大（需要数千张 GPU 卡、数周至数月训练）
#
# 2. **微调（Fine-Tuning）**：
#    - 目的：在特定任务的标注数据上优化模型，使其适应特定领域或任务
#    - 方法：监督学习（使用标注的输入-输出对）
#    - 结果：得到针对特定任务优化的模型
#    - 特点：计算量小（单卡或少量卡、数小时至数天）
#
# 3. **全参微调（Full Fine-Tuning）**：
#    - 定义：更新模型的所有参数
#    - 优点：性能天花板高，可以充分适应新任务
#    - 缺点：显存占用大（需要存储所有参数的梯度），训练慢，容易过拟合
#
# 4. **参数高效微调（PEFT - Parameter-Efficient Fine-Tuning）**：
#    - 定义：只更新模型的一小部分参数，冻结大部分预训练权重
#    - 代表方法：**LoRA（Low-Rank Adaptation）**
#    - 优点：显存占用小，训练快，防止过拟合，多任务部署方便
#    - 缺点：性能天花板略低于全参微调
#
# 【LoRA 原理】
# LoRA 的核心思想：模型适应新任务时，权重更新矩阵是"低秩"的（可以用两个小矩阵的乘积表示）。
#
# - 假设原始权重矩阵为 W ∈ R^(d×k)，LoRA 不直接更新 W，而是学习两个小矩阵：
#   - A ∈ R^(d×r)  （下投影矩阵）
#   - B ∈ R^(r×k)  （上投影矩阵）
#   其中 r << min(d, k) 是 LoRA 秩（rank），通常取 4、8、16 等
#
# - 微调后的权重为：W' = W + α * (B · A)
#   - W 保持冻结（不更新）
#   - 只训练 A 和 B（参数量为 r*(d+k)，远小于 d*k）
#   - α 是缩放因子，控制 LoRA 更新的强度
#
# - 参数量对比（以 d=4096, k=4096, r=8 为例）：
#   - 全参：4096 * 4096 = 16,777,216 参数
#   - LoRA：8 * (4096 + 4096) = 65,536 参数（仅 0.4%）
#
# 【预设配置说明】
# 本脚本内置5种预设配置，通过 --config-id 参数选择：
#
# ┌────┬─────────────────────┬─────────┬──────┬───────┬─────────────┬───────┬────────┬─────────┬────────┬──────────────────────────┐
# │ ID │ 名称                │ 学习率  │ Rank │ Epoch │ 数据集      │ Batch │ 累积步数│有效Batch│保存/评估│ 说明                     │
# ├────┼─────────────────────┼─────────┼──────┼───────┼─────────────┼───────┼────────┼─────────┼────────┼──────────────────────────┤
# │ 0  │ 过大学习率测试      │ 0.1     │ 4    │ 1     │ train_100   │ 8     │ 2      │ 16      │ 1步    │ 观察训练不稳定/发散现象  │
# │ 1  │ 快速验证（推荐）✨  │ 5e-5    │ 4    │ 1     │ train_100   │ 2     │ 8      │ 16      │ 1步    │ 内存优化，3-5分钟完成    │
# │ 2  │ 小数据集长训练      │ 5e-5    │ 4    │ 50    │ train_100   │ 4     │ 4      │ 16      │ 20步   │ 观察过拟合现象           │
# │ 3  │ 大数据集标准训练    │ 5e-5    │ 8    │ 3     │ train_1k    │ 4     │ 4      │ 16      │ 20步   │ 性能/时间平衡，推荐      │
# │ 4  │ 大数据集长训练      │ 5e-5    │ 8    │ 15    │ train_1k    │ 4     │ 4      │ 16      │ 20步   │ 追求最佳性能，耗时较长   │
# └────┴─────────────────────┴─────────┴──────┴───────┴─────────────┴───────┴────────┴─────────┴────────┴──────────────────────────┘
#
# 【重要说明】
# - ⚡ **已启用内存优化**：
#   1. Gradient Checkpointing（节省70%激活值内存）
#   2. 禁用训练中评估（节省2-3GB峰值内存）
#   3. 只保留最终checkpoint（节省磁盘空间）
# - Batch：单次前向传播的样本数（per_device_batch_size）
# - 累积步数：梯度累积步数（gradient_accumulation_steps）
# - 有效Batch：真实批次大小 = Batch × 累积步数
# - 💡 降低Batch、增加累积步数 = 降低峰值内存占用，但保持训练效果不变
# - 保存：每 save_steps 保存一次checkpoint
# - 评估：训练完成后用 `5_model_2_eval.py` 单独评估（不影响训练效果）
# - 有效Batch越大，梯度估计越准确，训练越稳定
# - 配置参数经过精心设计，脚本会严格按配置执行，不会动态调整
# - 如果遇到内存不足，使用 --force-cpu 参数强制CPU训练
# - ✅ 支持MPS（Apple Silicon，18GB可用）、CUDA（NVIDIA）、CPU设备
#
# 【使用方法】
# ```bash
# # 基本训练
# python 5_model_1_sft.py --config-id 1
#
# # 训练并合并模型
# python 5_model_1_sft.py --config-id 3 --merge
# ```
#
# 【输出文件结构】
# ```
# output/
# ├── config_0/
# │   └── checkpoints/             # LoRA权重checkpoints
# │       ├── checkpoint-1/        # 保存点1
# │       ├── checkpoint-2/        # 保存点2（保留最近2个）
# │       └── trainer_state.json   # 训练状态
# ├── config_1/
# │   └── checkpoints/
# ├── config_0_training_loss.png   # Loss曲线
# └── config_1_training_loss.png
# ```
#
# 【相关脚本】
# - `5_model_2_eval.py`: 评估微调后的模型性能（含基准测试）
# - `chatbot/sft_utils.py`: 共用的工具函数模块
# ==============================================================================

import os
import argparse
import torch
import glob
import shutil
import json
from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from chatbot.sft_utils import detect_device, plot_training_curves


def disable_mps_if_needed():
    """在force_cpu模式下完全禁用MPS"""
    if torch.backends.mps.is_available():
        # 设置环境变量禁用MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # 直接设置torch不使用MPS
        torch.set_default_device('cpu')
        print("   → MPS已禁用，所有操作将在CPU上进行")


# ==============================================================================
# 预设配置：5种内置配置方案
# ==============================================================================
PRESET_CONFIGS = {
    0: {
        "name": "过大学习率测试",
        "description": "故意设置过大的学习率（0.1），观察训练不稳定、loss震荡或发散现象",
        "learning_rate": 0.1,
        "lora_rank": 4,
        "num_train_epochs": 1,
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "save_steps": 1,
    },
    1: {
        "name": "快速验证（推荐）",
        "description": "使用标准配置快速验证流程，已启用内存优化（Gradient Checkpointing）",
        "learning_rate": 5e-5,
        "lora_rank": 4,
        "num_train_epochs": 1,
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 2,  # 降低batch_size以适应内存限制
        "gradient_accumulation_steps": 8,  # 增加累积步数，保持有效batch_size=16
        "max_length": 512,
        "save_steps": 1,
    },
    2: {
        "name": "小数据集长训练",
        "description": "在小数据集上进行长时间训练（50轮），观察过拟合现象",
        "learning_rate": 5e-5,
        "lora_rank": 4,
        "num_train_epochs": 50,
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "save_steps": 20,
    },
    3: {
        "name": "大数据集标准训练",
        "description": "使用1k样本进行标准3轮训练，较好的性能/时间平衡",
        "learning_rate": 5e-5,
        "lora_rank": 8,
        "num_train_epochs": 3,
        "train_file": "./resources/train_1k.jsonl",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "save_steps": 20,
    },
    4: {
        "name": "大数据集长训练",
        "description": "使用1k样本进行15轮长训练，追求最佳性能（耗时较长）",
        "learning_rate": 5e-5,
        "lora_rank": 8,
        "num_train_epochs": 15,
        "train_file": "./resources/train_1k.jsonl",
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "save_steps": 20,
    },
}


def load_and_prepare_dataset(train_file, split_ratio=0.01):
    """
    加载并准备数据集。
    
    【参数】
    - train_file: 训练数据文件路径（JSONL格式）
    - split_ratio: 验证集分割比例
    
    【返回】
    - train_dataset: 训练数据集
    - eval_dataset: 验证数据集
    """
    # 读取JSONL文件
    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # 转换为HuggingFace Dataset格式
    dataset = hf_load_dataset('json', data_files={'train': train_file}, split='train')
    
    # 分割训练集和验证集
    if split_ratio > 0:
        split_dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset


def tokenize_function(examples, tokenizer, max_length=512):
    """
    数据预处理函数：将对话格式转换为模型输入。
    
    【参数】
    - examples: 数据样本
    - tokenizer: tokenizer实例
    - max_length: 最大序列长度
    
    【返回】
    - 处理后的数据字典
    """
    # 提取messages并格式化
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for messages in examples['messages']:
        # 构建完整对话
        # messages格式：[{role: system, content: ...}, {role: user, content: ...}, {role: assistant, content: ...}]
        
        # 跳过system消息，只处理user和assistant
        user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
        assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')
        
        # 使用chat template格式化（如果tokenizer支持）
        if hasattr(tokenizer, 'apply_chat_template'):
            # 使用标准chat template
            formatted_messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # 简单拼接
            text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        
        # Tokenize
        tokenized = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,  # 动态padding由DataCollator处理
        )
        
        model_inputs["input_ids"].append(tokenized["input_ids"])
        model_inputs["attention_mask"].append(tokenized["attention_mask"])
        model_inputs["labels"].append(tokenized["input_ids"].copy())  # labels与input_ids相同
    
    return model_inputs


def run_fine_tune(config_id, model_path="./model", device=None):
    """
    运行 LoRA 微调实验。
    
    【功能】
    根据配置ID自动选择预设参数进行微调，输出文件按配置ID区分。
    
    【参数】
    - config_id: 配置ID（0-4）
    - model_path: 基座模型路径
    - device: 训练设备（None 则自动检测）
    
    【返回】
    - checkpoint_path: 微调后的 checkpoint 路径
    - loss_curve_path: Loss 曲线图片路径
    """
    if config_id not in PRESET_CONFIGS:
        print(f"❌ 错误：无效的配置ID {config_id}，请选择 0-4 之间的整数")
        return None, None
    
    config = PRESET_CONFIGS[config_id]
    
    print("\n" + "🎓 " + "="*76 + " 🎓")
    print("    本地模型监督微调（SFT）")
    print(f"    配置ID: {config_id} - {config['name']}")
    print("🎓 " + "="*76 + " 🎓")
    
    if not os.path.exists(config['train_file']):
        print(f"❌ 错误：训练文件不存在于 {config['train_file']}")
        return None, None
    
    # 检测设备
    if device is None:
        device, device_name = detect_device()
    else:
        if device.type == "mps":
            device_name = "MPS (Apple Silicon GPU)"
        elif device.type == "cuda":
            device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            device_name = "CPU"
    
    # 严格使用配置中的 batch_size 和 gradient_accumulation_steps，不做动态调整
    batch_size = config['batch_size']
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)  # 默认值4
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    print(f"\n📝 配置说明: {config['description']}")
    print(f"\n📊 训练参数:")
    print(f"   - 配置ID: {config_id}")
    print(f"   - 配置名称: {config['name']}")
    print(f"   - 设备: {device_name}")
    print(f"   - 学习率: {config['learning_rate']}")
    print(f"   - LoRA Rank: {config['lora_rank']}")
    print(f"   - 训练轮数: {config['num_train_epochs']}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 梯度累积步数: {gradient_accumulation_steps}")
    print(f"   - 有效批次大小: {effective_batch_size} (batch × accumulation)")
    print(f"   - 最大长度: {config['max_length']}")
    print(f"   - 保存间隔: {config['save_steps']} steps")
    print(f"   - 训练数据: {config['train_file']}")
    print(f"   - 内存优化: 已启用 Gradient Checkpointing + 禁用训练中评估")
    
    # 设备相关提示
    if device.type == "cpu":
        print("\n⚠️  注意：CPU 环境下训练速度较慢")
        if batch_size > 4:
            print(f"💡 提示：当前 batch_size={batch_size} 可能导致内存不足")
            print(f"   建议：选择配置0或1（batch_size=8）或在出现内存错误时降低配置")
    elif device.type == "mps":
        print("\n✅ 使用 Apple Silicon GPU (MPS) 加速")
        print(f"   LoRA内存占用：~{50 if config['lora_rank'] == 4 else 100}MB，完全可用")
        if batch_size > 8:
            print(f"💡 提示：当前 batch_size={batch_size}，预计总内存占用~5-6GB")
    else:
        print("\n✅ 使用 NVIDIA GPU 加速")
    
    # 配置输出目录
    config_dir = f"./output/config_{config_id}"
    output_dir = os.path.join(config_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 输出目录: {config_dir}")
    print("="*80)
    print("🚀 开始训练...")
    print("="*80)
    
    try:
        # ========== 步骤1：加载模型和tokenizer ==========
        print("\n[1/5] 加载模型和tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 设置pad_token（如果没有）
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32,
            device_map={"": device},  # 直接加载到指定设备
            trust_remote_code=True,
        )
        
        print(f"✅ 模型加载完成（设备: {device}）")
        
        # 🔥 启用梯度检查点（Gradient Checkpointing）以节省内存
        # 原理：不保存所有中间激活值，反向传播时重新计算
        # 效果：节省 ~70% 激活值内存，训练速度下降 10-20%
        # 不影响训练效果，只是用时间换空间
        model.gradient_checkpointing_enable()
        print("⚡ 已启用 Gradient Checkpointing（内存优化，不影响训练效果）")
        
        # ========== 步骤2：配置LoRA ==========
        print("\n[2/5] 配置LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['lora_rank'],  # LoRA秩
            lora_alpha=config['lora_rank'] * 2,  # 通常设置为rank的2倍
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数数量
        
        # ========== 步骤3：加载和准备数据集 ==========
        print("\n[3/5] 加载数据集...")
        train_dataset, eval_dataset = load_and_prepare_dataset(
            config['train_file'],
            split_ratio=0  # 不分割验证集（禁用训练中评估以节省内存）
        )
        
        print(f"✅ 训练样本: {len(train_dataset)}")
        if eval_dataset:
            print(f"✅ 验证样本: {len(eval_dataset)}")
        else:
            print(f"ℹ️  无验证集（已禁用训练中评估以节省内存）")
        
        # 数据预处理
        print("   预处理数据...")
        tokenize_fn = lambda examples: tokenize_function(examples, tokenizer, config['max_length'])
        
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing eval dataset"
            )
        
        # ========== 步骤4：配置Trainer ==========
        print("\n[4/5] 配置训练器...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=config['learning_rate'],
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=config['save_steps'],  # 与 save_steps 保持一致，确保记录更多训练 loss
            save_steps=config['save_steps'],
            save_strategy="steps",
            save_total_limit=1,  # 只保留最终checkpoint，节省磁盘和内存
            # 🔥 内存优化：禁用训练中评估（节省 2-3 GB 峰值内存）
            eval_strategy="no",               # 训练时不评估（训练后用 eval.py 单独评估）
            load_best_model_at_end=False,     # 不保存最佳模型副本
            fp16=device.type == "cuda",  # CUDA使用fp16
            # MPS暂不支持fp16训练，使用fp32
            report_to=[],  # 不上报到wandb等
            remove_unused_columns=False,
            # 内存优化参数（不影响训练效果）
            dataloader_num_workers=0,  # 不使用多进程加载数据，节省内存
            dataloader_pin_memory=False,  # 不固定内存，减少RAM占用
            gradient_checkpointing=True,  # 配合模型的gradient_checkpointing_enable
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # ========== 步骤5：开始训练 ==========
        print("\n[5/5] 开始训练...")
        print("="*80)
        
        # 训练前清理缓存（MPS/CUDA）
        if device.type == "mps":
            torch.mps.empty_cache()
            print("   → 已清理 MPS 缓存")
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            print("   → 已清理 CUDA 缓存")
        
        train_result = trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        trainer.save_state()
        
        print("\n✅ 微调完成！")
        
        # 查找最佳checkpoint
        checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)
            checkpoint_path = latest_checkpoint
        else:
            checkpoint_path = output_dir
        
        print(f"   Checkpoint: {checkpoint_path}")
        
        # 绘制 loss 曲线
        loss_curve_path = f"./output/config_{config_id}_training_loss.png"
        plot_training_curves(checkpoint_path, output_dir="./output", 
                            output_filename=f"config_{config_id}_training_loss.png")
        print(f"   Loss 曲线: {loss_curve_path}")
        
        return checkpoint_path, loss_curve_path
            
    except Exception as e:
        print(f"\n❌ 微调失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """
    主函数：解析命令行参数并执行微调。
    """
    parser = argparse.ArgumentParser(
        description="本地模型监督微调（SFT）- 基于 ms-swift + LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设配置说明：
  ID 0: 过大学习率测试 - 观察训练不稳定/发散现象
  ID 1: 快速验证（推荐）- 首次运行，5-10分钟完成
  ID 2: 小数据集长训练 - 观察过拟合现象
  ID 3: 大数据集标准训练 - 性能/时间平衡，推荐
  ID 4: 大数据集长训练 - 追求最佳性能，耗时较长

使用示例：
  # 配置1：快速验证（推荐首次运行）
  python 5_model_1_sft.py --config-id 1

  # 配置3：大数据集标准训练
  python 5_model_1_sft.py --config-id 3
        """
    )
    
    # 必需参数
    parser.add_argument("--config-id", type=int, required=True,
                       help="配置ID（0-4），每个ID对应一组预设的训练参数")
    
    # 可选参数
    parser.add_argument("--model-path", type=str, default="./model",
                       help="基座模型路径（默认: ./model）")
    parser.add_argument("--force-cpu", action="store_true",
                       help="强制使用CPU训练（当MPS内存不足时使用）")
    
    args = parser.parse_args()
    
    # 验证配置ID
    if args.config_id not in PRESET_CONFIGS:
        print(f"\n❌ 错误：无效的配置ID {args.config_id}")
        print("请选择 0-4 之间的整数：")
        for cid, cfg in PRESET_CONFIGS.items():
            print(f"  {cid}: {cfg['name']}")
        return
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"\n❌ 错误：模型文件不存在于 {args.model_path}")
        print("请先下载模型：")
        print(f"  mkdir -p {args.model_path}")
        print(f"  modelscope download --model qwen/Qwen2.5-1.5B-Instruct --local_dir '{args.model_path}'")
        return
    
    # 检测或强制指定设备
    if args.force_cpu:
        device = torch.device("cpu")
        print("\n⚠️  强制使用 CPU 训练模式")
        print("   （适用于 MPS 内存不足的场景）")
        disable_mps_if_needed()  # 完全禁用MPS
    else:
        device = None  # 自动检测
    
    # 运行微调
    checkpoint_path, loss_curve_path = run_fine_tune(
        config_id=args.config_id,
        model_path=args.model_path,
        device=device
    )
    
    # 输出总结
    print("\n" + "🏆 " + "="*76 + " 🏆")
    print("    训练完成")
    print("🏆 " + "="*76 + " 🏆")
    
    if checkpoint_path:
        print(f"\n✅ 配置 {args.config_id} 训练成功！")
        print(f"📁 LoRA Checkpoint: {checkpoint_path}")
        if loss_curve_path:
            print(f"📈 Loss 曲线: {loss_curve_path}")
        print(f"\n💡 下一步操作：")
        print(f"   # 评估微调后的模型")
        print(f"   python 5_model_2_eval.py --config-id {args.config_id}")
        print(f"\n   # 合并 LoRA 权重到基座模型")
        print(f"   python 5_model_3_merge.py --config-id {args.config_id}")
    else:
        print(f"\n❌ 配置 {args.config_id} 训练失败")
    
    print("\n" + "="*80)
    print("🎉 流程结束")
    print("="*80)


if __name__ == "__main__":
    main()
