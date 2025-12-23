#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 模型合并工具：将 LoRA 权重合并到基座模型
================================================================================

【核心功能】
将 LoRA 微调后的适配器权重合并到基座模型，生成独立的完整模型。

【使用场景】
- 部署环境：无需再加载 LoRA 适配器，简化推理流程
- 分发模型：生成独立模型文件，便于分享和版本管理
- 性能优化：避免运行时动态加载适配器的开销

【技术原理】
LoRA（Low-Rank Adaptation）微调时，只训练低秩矩阵 A 和 B：
  - 原始权重矩阵: W ∈ R^(m×n)
  - LoRA 低秩分解: ΔW = B·A，其中 B ∈ R^(m×r), A ∈ R^(r×n)，r << min(m,n)
  - 推理时权重: W' = W + α·(B·A)
  
模型合并就是将 ΔW 直接加到基座模型的 W 上，生成新的权重 W'。
合并后的模型不再需要 LoRA 适配器，可以独立使用。

【命令行参数】
- --config-id: 指定微调配置ID（0-4），自动查找对应的checkpoint
- --checkpoint: 手动指定checkpoint路径（优先级高于config-id）
- --output-dir: 指定合并后模型的保存目录（可选）

【使用示例】
# 方式1：通过配置ID合并
python 5_model_3_merge.py --config-id 1

# 方式2：手动指定checkpoint路径
python 5_model_3_merge.py --checkpoint ./output/qwen25-sft/v_0/checkpoint-xxx

# 方式3：指定输出目录
python 5_model_3_merge.py --config-id 1 --output-dir ./output/my_merged_model

# 评估合并后的模型（使用独立的评估脚本）
python 5_model_2_eval.py --merged-model ./output/merged_config_1

【输出目录结构】
output/
└── qwen25-sft/
    └── merged_config_1/           # 合并后的完整模型
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── ...

【注意事项】
⚠️ 合并操作是不可逆的：合并后无法再拆分出 LoRA 权重
⚠️ 存储空间：合并后的模型大小约为基座模型大小（~3GB）
⚠️ 性能对比：合并前后模型性能应该完全一致（理论上）

================================================================================
作者: AI Assistant
日期: 2024
框架: ms-swift 3.x
================================================================================
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def find_checkpoint_by_config_id(config_id):
    """
    根据配置ID查找最新的checkpoint。
    
    【参数】
    - config_id: 配置ID（0-4）
    
    【返回】
    - checkpoint_path: 找到的checkpoint路径，或 None
    """
    base_dir = Path(f"./output/config_{config_id}/checkpoints")
    
    # 查找所有checkpoint目录
    checkpoint_dirs = []
    if base_dir.exists():
        for item in base_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoint_dirs[0])


def merge_lora_weights(checkpoint_path, output_dir=None, base_model_path="./model"):
    """
    将 LoRA 权重合并到基座模型。
    
    【功能】
    使用 PEFT 的 merge_and_unload 方法将 LoRA 权重合并到基座模型。
    合并后的模型可以直接部署，无需再加载 LoRA 适配器。
    
    【参数】
    - checkpoint_path: LoRA checkpoint 路径
    - output_dir: 合并后模型的保存目录（可选）
    - base_model_path: 基座模型路径
    
    【返回】
    - output_dir: 合并后模型的路径（成功时），或 None（失败时）
    """
    print("\n" + "="*80)
    print("🔀 模型合并：将 LoRA 权重合并到基座模型")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：Checkpoint 不存在于 {checkpoint_path}")
        return None
    
    # 如果未指定输出目录，自动生成
    if output_dir is None:
        # 从checkpoint路径中提取config_id
        checkpoint_path_obj = Path(checkpoint_path)
        
        # 尝试从路径中提取config_id
        path_parts = checkpoint_path.split('/')
        config_id = None
        for part in path_parts:
            if part.startswith('config_'):
                config_id = part.replace('config_', '')
                break
        
        if config_id:
            output_dir = f"./output/merged_config_{config_id}"
        else:
            output_dir = f"./output/merged_model"
    
    print(f"📝 输入Checkpoint: {checkpoint_path}")
    print(f"📝 基座模型: {base_model_path}")
    print(f"📝 输出目录: {output_dir}")
    print(f"\n🚀 开始合并...")
    
    try:
        # 步骤1: 加载基座模型
        print("   [1/4] 加载基座模型...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # 先加载到CPU，合并后再处理
            trust_remote_code=True,
        )
        
        # 步骤2: 加载tokenizer
        print("   [2/4] 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        # 步骤3: 加载LoRA权重并合并
        print("   [3/4] 加载LoRA权重并合并...")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_model = model.merge_and_unload()
        
        # 步骤4: 保存合并后的模型
        print("   [4/4] 保存合并后的模型...")
        os.makedirs(output_dir, exist_ok=True)
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ 模型合并成功！")
        print(f"   合并后的模型: {output_dir}")
        return output_dir
            
    except Exception as e:
        print(f"\n❌ 模型合并失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数：解析命令行参数并执行模型合并。
    """
    parser = argparse.ArgumentParser(
        description="模型合并工具 - 将 LoRA 权重合并到基座模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 通过配置ID合并（自动查找最新checkpoint）
  python 5_model_3_merge.py --config-id 1

  # 手动指定checkpoint路径
  python 5_model_3_merge.py --checkpoint ./output/qwen25-sft/v_0/checkpoint-100

  # 指定输出目录
  python 5_model_3_merge.py --config-id 1 --output-dir ./output/my_merged_model
        """
    )
    
    # checkpoint 来源参数（二选一）
    parser.add_argument("--config-id", type=int,
                       help="配置ID（0-4），自动查找对应的checkpoint")
    parser.add_argument("--checkpoint", type=str,
                       help="手动指定checkpoint路径（优先级高于config-id）")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str,
                       help="指定合并后模型的保存目录（可选，默认自动生成）")
    
    args = parser.parse_args()
    
    # ========== 确定checkpoint路径 ==========
    checkpoint_path = args.checkpoint
    
    if checkpoint_path is None:
        if args.config_id is None:
            print("\n❌ 错误：必须指定 --config-id 或 --checkpoint")
            print("使用 -h 查看帮助信息")
            return
        
        # 通过config_id查找checkpoint
        print(f"\n🔍 正在查找配置 {args.config_id} 的最新 checkpoint...")
        checkpoint_path = find_checkpoint_by_config_id(args.config_id)
        
        if checkpoint_path is None:
            print(f"❌ 错误：未找到配置 {args.config_id} 的 checkpoint")
            print(f"请确认是否已完成训练：python 5_model_1_sft.py --config-id {args.config_id}")
            return
        
        print(f"✅ 找到 checkpoint: {checkpoint_path}")
    else:
        # 验证手动指定的checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ 错误：指定的 checkpoint 不存在: {checkpoint_path}")
            return
        print(f"\n✅ 使用指定的 checkpoint: {checkpoint_path}")
    
    # ========== 执行模型合并 ==========
    merged_model_path = merge_lora_weights(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir
    )
    
    if merged_model_path is None:
        print("\n❌ 模型合并失败")
        return
    
    # ========== 输出总结 ==========
    print("\n" + "🎉 " + "="*76 + " 🎉")
    print("    模型合并完成")
    print("🎉 " + "="*76 + " 🎉")
    print(f"\n✅ 合并后的模型: {merged_model_path}")
    
    print(f"\n💡 下一步：评估合并后的模型")
    print(f"   python 5_model_2_eval.py --merged-model '{merged_model_path}'")
    
    print("\n" + "="*80)
    print("🎉 流程结束")
    print("="*80)


if __name__ == "__main__":
    main()

