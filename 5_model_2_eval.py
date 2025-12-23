# ==============================================================================
# 说明：模型评测脚本 - 评估基座模型和微调后模型的性能
# ==============================================================================
# 本脚本用于评估不同模型在数学问答任务上的准确率，支持：
# 1. 基座模型评估：测试原始 Qwen2.5-1.5B-Instruct 模型的性能
# 2. LoRA 模型评估：加载 LoRA checkpoint，测试微调后的性能
# 3. 合并模型评估：测试合并了 LoRA 权重的完整模型
# 4. 对比分析：对比微调前后的性能提升
#
# 【评测流程】
# 1. 加载基座模型
# 2. 在测试集上运行基座模型推理，计算基准准确率
# 3. 加载微调后模型（LoRA 或合并模型）
# 4. 在测试集上运行微调后模型推理，计算微调后准确率
# 5. 对比分析性能提升
#
# 【评分规则】
# - 完全正确：答案完整匹配（包括格式），+1 分
# - 部分正确：核心数字匹配但格式有误，+0.5 分
# - 错误：答案不匹配，0 分
#
# 【配置ID说明】
# 本脚本支持评估5种预设配置对应的微调模型：
#
# ┌────┬─────────────────────┬─────────┬──────┬───────┬─────────────┬───────┬────────┬─────────┬─────────────────────────┐
# │ ID │ 名称                │ 学习率  │ Rank │ Epoch │ 数据集      │ Batch │ 累积步数│有效Batch│ 说明                    │
# ├────┼─────────────────────┼─────────┼──────┼───────┼─────────────┼───────┼────────┼─────────┼─────────────────────────┤
# │ 0  │ 过大学习率测试      │ 0.1     │ 4    │ 1     │ train_100   │ 8     │ 2      │ 16      │ 观察训练不稳定/发散现象 │
# │ 1  │ 快速验证（推荐）✨  │ 5e-5    │ 4    │ 1     │ train_100   │ 2     │ 8      │ 16      │ 内存优化，3-5分钟完成   │
# │ 2  │ 小数据集长训练      │ 5e-5    │ 4    │ 50    │ train_100   │ 4     │ 4      │ 16      │ 观察过拟合现象          │
# │ 3  │ 大数据集标准训练    │ 5e-5    │ 8    │ 3     │ train_1k    │ 4     │ 4      │ 16      │ 性能/时间平衡，推荐     │
# │ 4  │ 大数据集长训练      │ 5e-5    │ 8    │ 15    │ train_1k    │ 4     │ 4      │ 16      │ 追求最佳性能，耗时较长  │
# └────┴─────────────────────┴─────────┴──────┴───────┴─────────────┴───────┴────────┴─────────┴─────────────────────────┘
#
# 【说明】
# - 有效Batch = Batch × 累积步数（梯度累积步数）
# - 所有配置已启用内存优化（Gradient Checkpointing + 禁用训练中评估）
# - 训练完成后需用本脚本单独评估
#
# 【使用方法】
# ```bash
# # 评估配置1的LoRA模型（自动查找checkpoint）
# python 5_model_2_eval.py --config-id 1
#
# # 评估配置3的LoRA模型
# python 5_model_2_eval.py --config-id 3
#
# # 评估合并后的完整模型
# python 5_model_2_eval.py --merged-model ./output/config_3/merged_model
#
# # 手动指定LoRA checkpoint路径
# python 5_model_2_eval.py --checkpoint ./output/config_1/checkpoint
#
# # 只评估基座模型（不加载微调checkpoint）
# python 5_model_2_eval.py --baseline-only
#
# # 限制测试样本数
# python 5_model_2_eval.py --config-id 1 --max-samples 10
# ```
#
# 【相关脚本】
# - `5_model_1_sft.py`: 训练和微调脚本
# - `chatbot/sft_utils.py`: 共用的工具函数模块
# ==============================================================================

import os
import json
import argparse
import torch
import glob
from peft import PeftModel
from chatbot.sft_utils import (
    initialize_model,
    run_model_evaluation
)


def evaluate_finetuned_model(
    base_llm, 
    tokenizer, 
    template, 
    checkpoint_path, 
    test_file="./resources/test.jsonl",
    max_samples=None
):
    """
    加载微调后的LoRA模型并在测试集上评估。
    
    【功能】
    1. 从 checkpoint 加载 LoRA 权重
    2. 将 LoRA 权重应用到基座模型
    3. 使用统一的评估函数计算准确率
    
    【评分规则】
    使用 chatbot.sft_utils.run_model_evaluation 的统一评分：
    - 完全正确：+1.0 分
    - 部分正确：+0.5 分
    - 错误：0 分
    
    【参数】
    - base_llm: 基座模型
    - tokenizer: Tokenizer
    - template: 格式化模板
    - checkpoint_path: LoRA checkpoint 路径
    - test_file: 测试集文件
    - max_samples: 最多测试的样本数
    
    【返回】
    - accuracy: 准确率（0-100）
    - finetuned_model: 加载了 LoRA 权重的模型
    """
    print("\n" + "="*80)
    print("📊 加载 LoRA 模型")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误：Checkpoint 不存在于 {checkpoint_path}")
        return 0.0, None
    
    print(f"🔍 正在加载 LoRA 权重: {checkpoint_path}")
    
    # 加载 LoRA 权重
    try:
        finetuned_model = PeftModel.from_pretrained(base_llm, checkpoint_path)
        finetuned_model.eval()  # 设置为评估模式
        print("✅ LoRA 权重加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")
        return 0.0, None
    
    # 使用统一的评估函数
    accuracy, _ = run_model_evaluation(
        finetuned_model,
        tokenizer,
        template,
        test_file=test_file,
        max_samples=max_samples,
        support_partial=True,
        model_name="LoRA微调模型"
    )
    
    return accuracy, finetuned_model


def evaluate_merged_model(
    merged_model_path,
    test_file="./resources/test.jsonl",
    max_samples=None
):
    """
    评估合并后的完整模型。
    
    【功能】
    加载合并了 LoRA 权重的完整模型，使用统一的评估函数进行评估。
    
    【评分规则】
    使用 chatbot.sft_utils.run_model_evaluation 的统一评分：
    - 完全正确：+1.0 分
    - 部分正确：+0.5 分
    - 错误：0 分
    
    【参数】
    - merged_model_path: 合并后模型的路径
    - test_file: 测试集文件
    - max_samples: 最多测试的样本数
    
    【返回】
    - accuracy: 准确率（0-100）
    """
    print("\n" + "="*80)
    print("📊 加载合并模型")
    print("="*80)
    
    if not os.path.exists(merged_model_path):
        print(f"❌ 错误：合并模型不存在于 {merged_model_path}")
        return 0.0
    
    print(f"🔍 正在加载合并后的模型: {merged_model_path}")
    
    # 加载合并后的模型
    try:
        llm, tokenizer, template, device = initialize_model(model_path=merged_model_path)
        if llm is None:
            return 0.0
        print("✅ 合并模型加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {str(e)}")
        return 0.0
    
    # 使用统一的评估函数
    accuracy, _ = run_model_evaluation(
        llm,
        tokenizer,
        template,
        test_file=test_file,
        max_samples=max_samples,
        support_partial=True,
        model_name="合并模型"
    )
    
    return accuracy


def compare_results(baseline_accuracy, finetuned_accuracy, model_type="LoRA"):
    """
    对比微调前后的性能。
    
    【参数】
    - baseline_accuracy: 基座模型准确率
    - finetuned_accuracy: 微调后模型准确率
    - model_type: 微调模型类型（"LoRA" 或 "Merged"）
    """
    print("\n" + "🏆 " + "="*76 + " 🏆")
    print("    微调效果对比分析")
    print("🏆 " + "="*76 + " 🏆")
    
    print(f"\n📊 性能对比:")
    print(f"   - 微调前（基座模型）准确率: {baseline_accuracy:.1f}%")
    print(f"   - 微调后（{model_type}模型）准确率: {finetuned_accuracy:.1f}%")
    
    if finetuned_accuracy > baseline_accuracy:
        improvement = finetuned_accuracy - baseline_accuracy
        improvement_rate = (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        print(f"   - ✅ 性能提升: +{improvement:.1f}% (相对提升 {improvement_rate:.1f}%)")
    elif finetuned_accuracy < baseline_accuracy:
        degradation = baseline_accuracy - finetuned_accuracy
        print(f"   - ⚠️  性能下降: -{degradation:.1f}%")
    else:
        print(f"   - ➖ 性能持平")
    
    print("\n💡 分析与建议:")
    
    if finetuned_accuracy > baseline_accuracy:
        print("   ✅ 微调成功！模型性能得到提升。")
        if finetuned_accuracy < 80:
            print("   📈 进一步优化建议：")
            print("      - 增加训练数据量（使用 train_1k.jsonl）")
            print("      - 增大 LoRA rank（从 4 增加到 8 或 16）")
            print("      - 增加训练轮数（num_train_epochs）")
            print("      - 调整学习率（建议范围 1e-5 到 1e-4）")
    elif finetuned_accuracy < baseline_accuracy:
        print("   ⚠️  微调效果不理想，可能原因：")
        print("      1. 训练数据量不足或质量较差")
        print("      2. 学习率过大导致过拟合")
        print("      3. LoRA rank 过小，表达能力不足")
        print("      4. 训练轮数不够或过多")
        print("   🔧 调优建议：")
        print("      - 检查训练数据质量和数量")
        print("      - 降低学习率（如从 5e-5 降到 1e-5）")
        print("      - 增大 LoRA rank（如从 4 增到 8）")
        print("      - 调整训练轮数（观察 Loss 曲线确定最佳值）")
    else:
        print("   ➖ 微调效果不明显，建议：")
        print("      - 增加训练数据量")
        print("      - 调整超参数（学习率、rank、epoch）")
        print("      - 检查数据分布是否与测试集一致")


def main():
    """
    主函数：执行完整的评测流程。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="评估基座模型和微调后模型的性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 评估配置1的LoRA模型
  python 5_model_2_eval.py --config-id 1

  # 评估合并后的完整模型
  python 5_model_2_eval.py --merged-model ./output/config_3/merged_model

  # 手动指定LoRA checkpoint
  python 5_model_2_eval.py --checkpoint ./output/config_1/checkpoint

  # 只评估基座模型
  python 5_model_2_eval.py --baseline-only --max-samples 10
        """
    )
    parser.add_argument("--config-id", type=int, default=None,
                       help="配置ID（自动查找对应的 checkpoint）")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="LoRA checkpoint 路径（手动指定）")
    parser.add_argument("--merged-model", type=str, default=None,
                       help="合并后的完整模型路径")
    parser.add_argument("--baseline-only", action="store_true",
                       help="只评估基座模型，不评估微调后模型")
    parser.add_argument("--test-file", type=str, default="./resources/test.jsonl",
                       help="测试集文件路径")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最多测试的样本数（默认全部）")
    parser.add_argument("--model-path", type=str, default="./model",
                       help="基座模型路径（默认: ./model）")
    args = parser.parse_args()
    
    print("\n" + "📊 " + "="*76 + " 📊")
    print("    模型评测流程")
    print("    基于 Qwen2.5-1.5B-Instruct + LoRA")
    print("📊 " + "="*76 + " 📊")
    
    # ========== 步骤 1：初始化基座模型（如果不是只评估合并模型）==========
    if args.merged_model is None:
        llm, tokenizer, template, device = initialize_model(model_path=args.model_path)
        if llm is None:
            print("\n❌ 模型初始化失败，程序退出")
            return
        
        # ========== 步骤 2：基座模型评估 ==========
        print("\n" + "="*80)
        print("📊 步骤 1：基座模型评估（微调前）")
        print("="*80)
        baseline_accuracy, _ = run_model_evaluation(
            llm,
            tokenizer,
            template,
            test_file=args.test_file,
            max_samples=args.max_samples,
            support_partial=True,
            model_name="基座模型"
        )
    else:
        # 如果只评估合并模型，跳过基座模型评估
        baseline_accuracy = 0.0
        llm, tokenizer, template = None, None, None
    
    # 如果只评估基座模型，直接返回
    if args.baseline_only:
        print("\n" + "="*80)
        print("🎉 评测完毕！")
        print("="*80)
        return
    
    # ========== 步骤 3：确定评估目标 ==========
    if args.merged_model:
        # 评估合并后的模型
        print("\n" + "="*80)
        print("📊 步骤 2：合并模型评估")
        print("="*80)
        finetuned_accuracy = evaluate_merged_model(
            merged_model_path=args.merged_model,
            test_file=args.test_file,
            max_samples=args.max_samples
        )
        model_type = "Merged"
        model_path_display = args.merged_model
    else:
        # 评估 LoRA 模型
        checkpoint_path = args.checkpoint
        
        if checkpoint_path is None:
            if args.config_id is None:
                # 自动查找最新的 checkpoint（所有配置）
                all_checkpoints = glob.glob("./output/config_*/checkpoints/checkpoint-*")
                if not all_checkpoints:
                    print("\n⚠️  未找到任何 checkpoint")
                    print("💡 请先运行: python 5_model_1_sft.py --config-id <id>")
                    return
                
                checkpoint_path = max(all_checkpoints, key=os.path.getmtime)
                print(f"\n🔍 未指定配置ID，自动选择最新 checkpoint: {checkpoint_path}")
            else:
                # 根据配置ID查找最新的 checkpoint
                checkpoint_base = f"./output/config_{args.config_id}/checkpoints"
                if not os.path.exists(checkpoint_base):
                    print(f"\n⚠️  未找到配置 {args.config_id} 的 checkpoint 目录: {checkpoint_base}")
                    print(f"💡 请先运行: python 5_model_1_sft.py --config-id {args.config_id}")
                    return
                
                # 查找最新的checkpoint
                checkpoints = glob.glob(f"{checkpoint_base}/checkpoint-*")
                if not checkpoints:
                    print(f"\n⚠️  配置 {args.config_id} 没有任何 checkpoint")
                    print(f"💡 请先运行: python 5_model_1_sft.py --config-id {args.config_id}")
                    return
                
                checkpoint_path = max(checkpoints, key=os.path.getmtime)
                print(f"\n🔍 配置 {args.config_id} 的最新 checkpoint: {checkpoint_path}")
        else:
            if not os.path.exists(checkpoint_path):
                print(f"\n❌ 错误：指定的 checkpoint 不存在: {checkpoint_path}")
                return
            print(f"\n🔍 使用指定的 checkpoint: {checkpoint_path}")
        
        # ========== 步骤 4：LoRA 模型评估 ==========
        print("\n" + "="*80)
        print("📊 步骤 2：微调后模型评估")
        print("="*80)
        finetuned_accuracy, finetuned_model = evaluate_finetuned_model(
            llm,
            tokenizer,
            template,
            checkpoint_path,
            test_file=args.test_file,
            max_samples=args.max_samples
        )
        model_type = "LoRA"
        model_path_display = checkpoint_path
    
    # ========== 步骤 5：对比分析 ==========
    if finetuned_accuracy > 0:
        if args.merged_model is None:  # 只有评估LoRA模型时才进行对比
            compare_results(baseline_accuracy, finetuned_accuracy, model_type)
        else:
            print(f"\n✅ 合并模型评估完成，准确率: {finetuned_accuracy:.1f}%")
    
    print("\n" + "="*80)
    print("🎉 评测完毕！")
    print("="*80)
    
    # 打印相关文件位置
    print(f"\n📁 相关文件:")
    print(f"   - 评估模型: {model_path_display}")
    
    # 查找对应的 Loss 曲线
    if args.config_id is not None:
        loss_curve_path = f"./output/config_{args.config_id}_training_loss.png"
        if os.path.exists(loss_curve_path):
            print(f"   - Loss 曲线: {loss_curve_path}")


if __name__ == "__main__":
    main()
