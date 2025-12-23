"""
通用工具函数模块。

提供常用的数学和文本处理工具，包括：
- 向量相似度计算
- Embedding 模型对比
"""

import numpy as np


def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度。
    
    【公式】
    cos(θ) = (A · B) / (||A|| * ||B||)
    
    Args:
        a: 向量 A
        b: 向量 B
        
    Returns:
        float: 余弦相似度（-1 到 1）
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compare_embeddings(query, chunks, embedding_models):
    """
    比较不同嵌入模型的文本相似度。
    
    【用途】
    直观展示不同 Embedding 模型对同一文本的语义理解差异。
    
    Args:
        query: 查询文本
        chunks: 待比较的文本片段列表
        embedding_models: 嵌入模型字典 {模型名: 模型对象}
    """
    print(f"查询: {query}")
    for i, chunk in enumerate(chunks, 1):
        print(f"文本 {i}: {chunk}")

    for model_name, model in embedding_models.items():
        print(f"\n{'='*20} {model_name} {'='*20}")
        query_embedding = (model.get_query_embedding(query) if hasattr(model, 'get_query_embedding')
                         else model.get_text_embedding(query))

        for i, chunk in enumerate(chunks, 1):
            chunk_embedding = model.get_text_embedding(chunk)
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            print(f"查询与文本 {i} 的相似度: {similarity:.4f}")

