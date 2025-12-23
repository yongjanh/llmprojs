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

# 消除训练过程中的警告信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datasets import load_dataset as hf_load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from chatbot.sft_utils import plot_training_curves


# ==============================================================================
# 预设配置：5种内置配置方案（已优化，对齐原始swift配置）
# ==============================================================================
# 【优化要点】
# - warmup_steps=100（固定步数，而非比例）→ Config 3预热占比大，训练相对不足
# - weight_decay=0.1（增强正则化）→ 防止小数据集过拟合
# - lr_scheduler_type="cosine"（余弦衰减）→ 学习率更平滑优雅
# ==============================================================================
PRESET_CONFIGS = {
    0: {
        "name": "过大学习率测试",
        "description": "故意设置过大的学习率（0.1），观察训练不稳定、loss震荡或发散现象",
        "learning_rate": 0.1,  # 过大的学习率（正常值的2000倍）
        "lora_rank": 4,
        "num_train_epochs": 1,
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "save_steps": 1,  # 每步都保存，观察训练初期的细节变化
        "split_ratio": 0.1,
    },
    1: {
        "name": "快速验证（推荐）",
        "description": "使用标准配置快速验证流程（1轮训练）",
        "learning_rate": 5e-5,  # 标准学习率
        "lora_rank": 4,  # 小秩，适合100样本的小数据集
        "num_train_epochs": 1,  # 仅1轮，用于快速验证流程可行性
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,  # 有效批次=16
        "max_length": 512,
        "save_steps": 1,  # 每步都保存，便于观察训练初期
        "split_ratio": 0.1,  # 90训练+10验证
    },
    2: {
        "name": "小数据集长训练",
        "description": "在小数据集上进行长时间训练（50轮），观察过拟合现象",
        "learning_rate": 5e-5,
        "lora_rank": 4,
        "num_train_epochs": 50,  # 充分训练，使模型在小数据集上过拟合
        "train_file": "./resources/train_100.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "save_steps": 20,  # 每20步保存/评估（总300步，保存15次）
        "split_ratio": 0.1,  # 用于观察训练Loss与验证Loss的分离
    },
    3: {
        "name": "大数据集标准训练",
        "description": "使用1k样本进行3轮训练，由于warmup占比大（100/178步=56%），预期训练不够充分（欠拟合示例）",
        "learning_rate": 5e-5,
        "lora_rank": 8,  # 较大的秩，匹配更多的训练数据
        "num_train_epochs": 3,  # 标准训练轮数，平衡效果与速度
        "train_file": "./resources/train_1k.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "save_steps": 20,
        "split_ratio": 0.05,  # 950训练+50验证，验证集比例适中
    },
    4: {
        "name": "大数据集长训练",
        "description": "使用1k样本进行15轮训练，warmup占比合理（100/890步=11%），充分训练达到最佳效果",
        "learning_rate": 5e-5,
        "lora_rank": 8,
        "num_train_epochs": 15,  # 充分训练，追求最佳效果
        "train_file": "./resources/train_1k.jsonl",
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "max_length": 512,
        "save_steps": 20,  # 总1000步，保存50次
        "split_ratio": 0.05,
    },
}


def load_and_prepare_dataset(train_file, split_ratio=0.1):
    """
    加载并准备数据集（GPU训练，始终使用验证集）。
    
    【参数】
    - train_file: 训练数据文件路径（JSONL格式）
    - split_ratio: 验证集分割比例（>0，所有配置都有验证集）
    
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
    
    # 分割训练集和验证集（所有配置都使用验证集）
    split_dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
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


def run_fine_tune(config_id, model_path="./model"):
    """
    运行 LoRA 微调实验。
    
    【功能】
    根据配置ID自动选择预设参数进行微调，输出文件按配置ID区分。
    
    【参数】
    - config_id: 配置ID（0-4）
    - model_path: 基座模型路径
    
    【返回】
    - checkpoint_path: 微调后的 checkpoint 路径
    - loss_curve_path: Loss 曲线图片路径
    """
    if config_id not in PRESET_CONFIGS:
        print(f"❌ 错误：无效的配置ID {config_id}，请选择 0-4 之间的整数")
        return None, None
    
    config = PRESET_CONFIGS[config_id]
    
    print("\n" + "🎓 " + "="*76 + " 🎓")
    print("    本地模型监督微调（SFT - GPU）")
    print(f"    配置ID: {config_id} - {config['name']}")
    print("🎓 " + "="*76 + " 🎓")
    
    if not os.path.exists(config['train_file']):
        print(f"❌ 错误：训练文件不存在于 {config['train_file']}")
        return None, None
    
    # 检测GPU设备
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到CUDA GPU，本脚本仅支持GPU训练")
        return None, None
    
    device = torch.device("cuda")
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    
    # 严格使用配置中的 batch_size 和 gradient_accumulation_steps
    batch_size = config['batch_size']
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
    effective_batch_size = batch_size * gradient_accumulation_steps
    
    print(f"\n📝 配置说明: {config['description']}")
    print(f"\n📊 训练参数:")
    print(f"   - GPU: {device_name}")
    print(f"   - 学习率: {config['learning_rate']}")
    print(f"   - Warmup步数: 100步（固定，对齐swift配置）")
    print(f"   - 学习率衰减: Cosine（余弦衰减）")
    print(f"   - 正则化强度: Weight Decay = 0.1")
    print(f"   - LoRA Rank: {config['lora_rank']}")
    print(f"   - 训练轮数: {config['num_train_epochs']}")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 梯度累积步数: {gradient_accumulation_steps}")
    print(f"   - 有效批次大小: {effective_batch_size}")
    print(f"   - 验证集比例: {config.get('split_ratio', 0.1)*100:.0f}%")
    print(f"   - 训练数据: {config['train_file']}")
    
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
        
        # 加载模型到GPU（使用fp16加速）
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )
        
        print(f"✅ 模型已加载到 {device_name}")
        
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
            r=config['lora_rank'],  # LoRA秩：控制可训练参数量
            lora_alpha=config['lora_rank'] * 2,  # LoRA缩放系数：通常为rank的2倍
            lora_dropout=0.05,  # Dropout防止过拟合
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 应用到所有线性层
            bias="none",
        )
        
        # 应用LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # 打印可训练参数数量
        
        # ========== 步骤3：加载和准备数据集 ==========
        print("\n[3/5] 加载数据集...")
        split_ratio = config.get('split_ratio', 0.1)
        train_dataset, eval_dataset = load_and_prepare_dataset(
            config['train_file'],
            split_ratio=split_ratio
        )
        
        print(f"✅ 训练样本: {len(train_dataset)}")
        print(f"✅ 验证样本: {len(eval_dataset)}")
        
        # 数据预处理
        print("   预处理数据...")
        tokenize_fn = lambda examples: tokenize_function(examples, tokenizer, config['max_length'])
        
        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset"
        )
        
        eval_dataset = eval_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
        
        # ========== 步骤4：配置Trainer ==========
        print("\n[4/5] 配置训练器...")
        
        # 对于短训练（1轮），保留所有checkpoint以完整记录训练过程
        # 对于长训练（多轮），只保留最近1个checkpoint节省磁盘
        save_total_limit = None if config['num_train_epochs'] == 1 else 1
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=config['learning_rate'],
            warmup_steps=100,  # 固定100步预热，与原始swift配置对齐
            weight_decay=0.1,  # 增强L2正则化，防止过拟合
            lr_scheduler_type="cosine",  # 余弦衰减，学习率更平滑
            logging_steps=config['save_steps'],
            save_steps=config['save_steps'],
            save_strategy="steps",
            save_total_limit=save_total_limit,  # 短训练保留全部，长训练只保留最近1个
            # 训练中评估配置（所有配置都有验证集）
            eval_strategy="steps",
            eval_steps=config['save_steps'],
            load_best_model_at_end=True,  # 训练结束时自动加载验证Loss最低的模型
            metric_for_best_model="loss",
            fp16=True,  # 半精度训练：节省显存，加速计算（A10支持）
            report_to=[],
            remove_unused_columns=False,
            # GPU优化参数
            dataloader_num_workers=4,  # 多进程并行加载数据
            dataloader_pin_memory=True,  # 固定内存页，加速CPU→GPU数据传输
            gradient_checkpointing=True,  # 梯度检查点：节省70%激活值显存，训练速度降低10-20%
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
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        train_result = trainer.train()
        
        print("\n✅ 微调完成！")
        
        # 保存最佳模型（已由load_best_model_at_end自动加载验证Loss最低的模型）
        best_model_dir = os.path.join(config_dir, "best_model")
        print(f"   保存最佳模型（验证Loss最低）到: {best_model_dir}")
        trainer.save_model(best_model_dir)
        trainer.save_state()
        checkpoint_path = best_model_dir
        
        # 提示：训练过程中的checkpoints仍保留在checkpoints目录
        print(f"   训练过程checkpoints: {output_dir}")
        print(f"   评估用最佳模型: {checkpoint_path}")
        
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
        description="本地模型监督微调（SFT - GPU）- 基于 PEFT + LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
预设配置说明（已优化，对齐原始swift配置）：
  ID 0: 过大学习率测试 - 观察训练不稳定/发散现象
  ID 1: 快速验证（推荐）- 首次运行，1分钟完成
  ID 2: 小数据集长训练 - 观察过拟合现象（50轮）
  ID 3: 大数据集标准训练 - 3轮训练，预期欠拟合（教学用）
  ID 4: 大数据集长训练 - 15轮训练，充分训练达到最佳效果

关键优化（对齐swift）：
  - Warmup: 固定100步（而非比例），Config 3预热占比大
  - 学习率衰减: Cosine余弦衰减（更平滑）
  - 正则化: Weight Decay = 0.1（增强版）

使用示例：
  python 5_model_1_sft.py --config-id 1  # 快速验证
  python 5_model_1_sft.py --config-id 3  # 标准训练（欠拟合示例）
  python 5_model_1_sft.py --config-id 4  # 充分训练（最佳效果）
        """
    )
    
    # 参数
    parser.add_argument("--config-id", type=int, required=True,
                       help="配置ID（0-4），每个ID对应一组预设的训练参数")
    parser.add_argument("--model-path", type=str, default="./model",
                       help="基座模型路径（默认: ./model）")
    
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
    
    # 运行微调
    checkpoint_path, loss_curve_path = run_fine_tune(
        config_id=args.config_id,
        model_path=args.model_path
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
