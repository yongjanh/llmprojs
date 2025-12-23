"""
RAG 系统评估工具模块。

提供基于 Ragas 框架的 RAG 系统评估功能，包括：
- 评估结果计算
- 评估结果展示
"""

import pandas as pd
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, answer_correctness


def evaluate_result(question, response, ground_truth):
    """
    使用 Ragas 评估回答质量。
    
    【评估维度】
    - Answer Correctness: 回答正确性（语义 + 事实）
    - Context Recall: 检索召回率
    - Context Precision: 检索精确度（排序质量）
    
    Args:
        question: 用户问题
        response: 查询响应对象
        ground_truth: 标准答案
        
    Returns:
        pd.DataFrame: 包含各项评估指标的结果表
    """
    if hasattr(response, 'response_txt'):
        answer = response.response_txt
    else:
        answer = str(response)
    
    if hasattr(response, 'source_nodes'):
        context = [source_node.get_content() for source_node in response.source_nodes]
    else:
        context = [""]

    data_samples = {
        'question': [question],
        'answer': [answer],
        'ground_truth': [ground_truth],
        'contexts': [context],
    }
    dataset = Dataset.from_dict(data_samples)

    score = evaluate(
        dataset=dataset,
        metrics=[answer_correctness, context_recall, context_precision],
        llm=Tongyi(model_name="qwen-plus"),
        embeddings=DashScopeEmbeddings(model="text-embedding-v3")
    )
    return score.to_pandas()


def show_evaluation_result(result_df, title="评估结果"):
    """
    统一格式化打印评估结果。
    
    【展示策略】
    只显示核心指标，过滤掉冗长的 context 和 answer 列。
    
    Args:
        result_df: Ragas 评估结果 DataFrame
        title: 结果标题
    """
    print(f"\n📊 {title}:")
    print("-" * 60)
    
    metrics_cols = ['answer_correctness', 'context_recall', 'context_precision']
    cols_to_show = [col for col in metrics_cols if col in result_df.columns]
    
    if cols_to_show:
        print(result_df[cols_to_show].to_string(index=False))
    else:
        print(result_df.to_string(index=False))
    print("-" * 60)

