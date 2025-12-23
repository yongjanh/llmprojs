# ==============================================================================
# 说明：本文件用于演示 RAG 系统的全链路性能优化策略
# ==============================================================================
# 从数据准备到生成输出，系统化地展示 RAG 优化的 8 个关键环节。
#
# 【核心目标】
# 通过具体优化手段，解决 RAG 系统中常见的"检索不准"、"回答不佳"等问题，
# 并使用 Ragas 进行量化验证。
#
# 【优化路线图】
# 1. 数据层：扩大 Top-K、重建索引（Step 1-2）
# 2. 解析层：PDF → Markdown 结构化（Step 3）
# 3. 切片层：5 种切片策略对比（Step 4）
# 4. 嵌入层：Embedding 模型对比（Step 5）
# 5. 存储层：向量数据库选型（Step 6，理论）
# 6. 检索层：问题改写、多步查询、HyDE、Rerank（Step 7）
# 7. 生成层：Temperature、Penalty、Seed 调优（Step 8）
#
# 【关键原理说明】
# 
# 1. 文档准备阶段原理
#    - 意图空间（用户问的问题）vs 知识空间（已有文档内容）
#    - 重叠部分：可以依靠知识库回答问题，需持续提升内容质量
#    - 未覆盖的意图空间：幻觉居多，需要补充缺漏知识
#    - 未被利用的知识空间：可能产生检索干扰，需优化召回算法或剔除噪音
#
# 2. 文档解析与切片原理
#    - 文档解析：传统 PDF 解析常丢失表格、标题层级。使用 PyMuPDF4LLM 或 
#      DashScopeParse 转为 Markdown 可保留结构
#    - 文档切片：主题接近的内容应聚合。Token 切片容易切断语义；Markdown 切片
#      能按章节保持语义完整；Semantic 切片能智能检测语义边界
#
# 3. 切片策略对比
#    ┌────────────────────┬──────────────┬──────────────┬──────────────┐
#    │ 切片器             │ 切分依据     │ 优势         │ 劣势         │
#    ├────────────────────┼──────────────┼──────────────┼──────────────┤
#    │ TokenTextSplitter  │ Token 数量   │ 简单快速     │ 易切断语义   │
#    │ SentenceSplitter   │ 句子边界     │ 保持句完整   │ 忽略章节结构 │
#    │ MarkdownNodeParser │ Markdown结构 │ 保留层级关系 │ 依赖格式质量 │
#    │ SentenceWindow     │ 滑动窗口     │ 上下文丰富   │ 冗余信息多   │
#    │ SemanticSplitter   │ 语义相似度   │ 智能边界检测 │ 计算成本高   │
#    └────────────────────┴──────────────┴──────────────┴──────────────┘
#
# 4. 向量数据库选型原理
#    - 核心逻辑：向量数据库是 RAG 系统的"记忆中枢"。不同数据库在"存储效率"、
#      "查询性能"、"扩展性"等方面各有优劣
#    - 本例使用 LlamaIndex 默认的内存 VectorStore，实际生产中应根据负载选择
#      Milvus/DashVector/Pgvector
#
# 5. 检索层优化原理
#    - 检索前：通过问题改写、多步查询、HyDE 等手段提升召回率
#    - 检索中：通过标签增强、混合检索等手段提升准确率
#    - 检索后：通过重排序（Rerank）提升 Top-N 质量
#    - Rerank 原理：向量检索（Bi-Encoder）速度快但精度一般，适合粗排；
#      重排序（Cross-Encoder）计算量大但精度极高，适合精排
#
# 6. 生成层优化原理
#    - Temperature（温度）：控制输出随机性。低温（0.1）适合事实问答，
#      高温（0.7+）适合创意生成
#    - Presence Penalty（存在惩罚）：正值鼓励模型谈论新话题，防止重复
#    - Seed（随机种子）：固定 Seed 可让模型输出具有可复现性（Deterministic）
# ==============================================================================

from tqdm.cli import tqdm as tqdm_cli
import tqdm.auto
tqdm.auto.tqdm = tqdm_cli

# 导入所需的依赖包
import os
import logging
import pandas as pd
import numpy as np
from openai import OpenAI
from IPython.display import display

# LlamaIndex Core
from llama_index.core import (
    Settings, 
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    Document, 
    PromptTemplate
)
from llama_index.core.node_parser import (
    SentenceSplitter,
    MarkdownNodeParser,
    TokenTextSplitter,
    SentenceWindowNodeParser,
    SemanticSplitterNodeParser
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SimilarityPostprocessor
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform, HyDEQueryTransform
from llama_index.core.query_engine import MultiStepQueryEngine, TransformQueryEngine

# LlamaIndex Integrations
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

# Evaluation (核心评估功能已移至 chatbot/eval.py)

# Local Utils
from config.load_key import load_key
from chatbot import rag
from chatbot.eval import evaluate_result, show_evaluation_result
from chatbot.document import file_to_md_local, md_polisher
from chatbot.utils import cosine_similarity, compare_embeddings

# 设置 pandas 显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

# 设置日志级别
logging.basicConfig(level=logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 加载API密钥
load_key()
print(f'''你配置的 API Key 是：{os.environ["DASHSCOPE_API_KEY"][:5]+"*"*5}''')

# 配置 LlamaIndex
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)


# ==============================================================================
#                               工具函数定义区
# ==============================================================================

def ask(question, query_engine):
    """
    执行问答并打印结果。
    
    【功能】
    1. 尝试更新 Prompt 模板（如果支持）
    2. 执行查询并流式打印回答
    3. 展示检索到的参考文档及其得分
    
    Args:
        question: 用户问题
        query_engine: LlamaIndex 查询引擎
        
    Returns:
        response: 查询响应对象
    """
    # 尝试更新 prompt template，如果不适用则忽略
    try:
        rag.update_prompt_template(query_engine=query_engine)
    except:
        pass

    print('=' * 50)
    print(f'🤔 问题：{question}')
    print('=' * 50 + '\n')

    response = query_engine.query(question)

    print('🤖 回答：')
    if hasattr(response, 'print_response_stream') and callable(response.print_response_stream):
        response.print_response_stream()
    else:
        print(str(response))

    print('\n' + '-' * 50)
    print('📚 参考文档 (Top 检索结果)：\n')
    if hasattr(response, 'source_nodes'):
        for i, source_node in enumerate(response.source_nodes, start=1):
            score_str = f"{source_node.score:.4f}" if source_node.score is not None else "N/A"
            print(f'文档 {i} (Score: {score_str}):')
            # 截取前200字符避免刷屏
            content_preview = source_node.get_content()[:200].replace('\n', ' ')
            print(f"{content_preview}...")
            print()
    print('-' * 50)

    return response


# evaluate_result, show_evaluation_result: 已移至 chatbot/eval.py
# file_to_md_local, md_polisher: 已移至 chatbot/document.py


def evaluate_splitter(splitter, documents, question, ground_truth, splitter_name, node_postprocessors=None):
    """
    评测不同文档切片方法的效果。
    
    【评测流程】
    1. 使用指定切片器处理文档
    2. 构建向量索引
    3. 执行查询并打印回答
    4. 使用 Ragas 评估质量
    
    Args:
        splitter: 切片器对象
        documents: 文档列表
        question: 测试问题
        ground_truth: 标准答案
        splitter_name: 切片器名称（用于显示）
        node_postprocessors: 后处理器列表（可选）
    """
    print(f"\n{'='*50}")
    print(f"🔍 正在使用 {splitter_name} 方法进行切片测试...")
    print(f"{'='*50}\n")

    print("📑 正在处理文档与构建索引...")
    nodes = splitter.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        streaming=True,
        node_postprocessors=node_postprocessors 
    )

    response = query_engine.query(question)
    
    print(f"\n🤖 {splitter_name} 模型回答:")
    response.print_response_stream()
    
    show_evaluation_result(evaluate_result(question, response, ground_truth), f"{splitter_name} 评估结果")


# cosine_similarity, compare_embeddings: 已移至 chatbot/utils.py


def compare_embedding_models(documents, question, ground_truth, sentence_splitter):
    """
    比较不同嵌入模型在 RAG 中的表现。
    
    【对比维度】
    - 召回的文档片段
    - 生成的回答质量
    - Ragas 评估得分
    
    Args:
        documents: 文档列表
        question: 测试问题
        ground_truth: 标准答案
        sentence_splitter: 切片器
    """
    print("📑 正在处理文档...")
    nodes = sentence_splitter.get_nodes_from_documents(documents)

    embedding_models = {
        "text-embedding-v2": DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2),
        "text-embedding-v3": DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3)
    }

    for model_name, embed_model in embedding_models.items():
        print(f"\n{'='*50}")
        print(f"🔍 正在测试 {model_name}...")
        print(f"{'='*50}")

        index = VectorStoreIndex(nodes, embed_model=embed_model)
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=5)

        print(f"\n❓ 测试问题: {question}")
        print("\n🤖 模型回答:")
        response = query_engine.query(question)
        response.print_response_stream()

        print(f"\n📚 召回的参考片段:")
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n文档片段 {i}:")
            print("-" * 40)
            print(node)

        show_evaluation_result(evaluate_result(question, response, ground_truth), f"{model_name} 评估结果")


# --- Step 7 专用工具函数 ---

query_gen_str = """\
系统角色设定:
你是一个专业的问题改写助手。你的任务是将用户的原始问题扩充为一个更完整、更全面的问题。

规则：
1. 将可能的歧义、相关概念和上下文信息整合到一个完整的问题中
2. 使用括号对歧义概念进行补充说明
3. 添加关键的限定词和修饰语
4. 确保改写后的问题清晰且语义完整
5. 对于模糊概念，在括号中列举主要可能性

原始问题:
{query}

请生成一个综合的改写问题，确保：
- 包含原始问题的核心意图
- 涵盖可能的歧义解释
- 使用清晰的逻辑关系词连接不同方面
- 必要时使用括号补充说明

输出格式：
[综合改写] - 改写后的问题
"""
query_gen_prompt = PromptTemplate(query_gen_str)


def generate_queries(query: str):
    """
    使用 LLM 扩写问题。
    
    【应用场景】
    用户问题往往过于简短或模糊，通过 LLM 扩写可以：
    - 补充上下文信息
    - 澄清歧义
    - 提升检索召回率
    
    Args:
        query: 原始问题
        
    Returns:
        str: 扩写后的问题
    """
    response = Settings.llm.predict(
        query_gen_prompt, query=query
    )
    return response


def extract_tags(text):
    """
    提取文本标签（使用 OpenAI Compatible API）。
    
    【支持的标签类型】
    - 人名
    - 部门名称
    - 职位名称
    - 技术领域
    - 产品名称
    
    【应用场景】
    在混合检索中，标签可以作为精确匹配的过滤条件，
    与向量检索（模糊匹配）形成互补。
    
    Args:
        text: 待提取标签的文本
        
    Returns:
        str: JSON 格式的标签列表
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    system_message = """你是一个标签提取专家。请从文本中提取结构化信息，并按要求输出标签。
---
【支持的标签类型】
- 人名
- 部门名称
- 职位名称
- 技术领域
- 产品名称
---
【输出要求】
1. 请用 JSON 格式输出，如：[{"key": "部门名称", "value": "教研部"}]
2. 如果某类标签未识别到，则不输出该类
"""
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': text}
        ],
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


# ==============================================================================
#                               主程序执行区
# ==============================================================================

# 定义测试问题与标准答案
question = '张伟是哪个部门的'
ground_truth = '''公司有三名张伟，分别是：
- 教研部的张伟：职位是教研专员，邮箱 zhangwei@educompany.com。
- 课程开发部的张伟：职位是课程开发专员，邮箱 zhangwei01@educompany.com。
- IT部的张伟：职位是IT专员，邮箱 zhangwei036@educompany.com。
'''


def demo_step1_baseline():
    """
    Step 1: 基准测试 (Baseline)。
    
    【目的】
    使用旧索引 + 扩大 Top-K 进行基准测试，建立优化前的性能基线。
    
    【核心操作】
    - 加载已有索引
    - 设置 similarity_top_k=10（扩大召回窗口）
    - 执行查询并评估
    """
    print("\n" + "#" * 30 + " Step 1: 基准测试 (Top-K=10) " + "#" * 30 + "\n")
    try:
        index = rag.load_index()
        query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
        response = ask(question, query_engine)
        show_evaluation_result(evaluate_result(question, response, ground_truth), "Step 1 评估结果")
    except Exception as e:
        print(f"Step 1 跳过 (索引可能不存在): {e}")


def demo_step2_reindex():
    """
    Step 2: 数据层优化 - 重建索引。
    
    【目的】
    当知识库陈旧或缺失关键文档时，需要重新加载文档并重建索引。
    
    【核心操作】
    - 加载新文档目录 (./docs/ragdoc1)
    - 使用 VectorStoreIndex.from_documents() 重建索引
    - 对比优化前后的效果
    
    Returns:
        tuple: (文档列表, 新索引)
    """
    print("\n" + "#" * 30 + " Step 2: 数据层优化 (重建索引) " + "#" * 30 + "\n")
    print('📂 正在加载新文档 (./docs/ragdoc1)...')
    documents_step2 = SimpleDirectoryReader('./docs/ragdoc1').load_data()

    print('🛠️ 正在重建索引...')
    index_step2 = VectorStoreIndex.from_documents(documents_step2)

    query_engine = index_step2.as_query_engine(streaming=True, similarity_top_k=5)
    response = ask(question, query_engine)
    show_evaluation_result(evaluate_result(question, response, ground_truth), "Step 2 评估结果")

    return documents_step2, index_step2


def demo_step3_parsing():
    """
    Step 3: 解析层优化 - PDF 转 Markdown。
    
    【目的】
    传统 PDF 解析常丢失表格、标题层级等结构信息。
    使用 PyMuPDF4LLM 转为 Markdown 可保留文档结构。
    
    【原理】
    PDF 文件本质是"排版指令"，直接提取文本会丢失结构信息（如表格边界、标题层级）。
    结构化解析策略：
    - PyMuPDF4LLM: 本地解析，轻量级，速度快，适合简单文档
    - DashScopeParse: 云端解析，重量级，质量高，适合复杂文档（表格、公式）
    - MinerU: 本地解析，重量级，需要 GPU，质量最高
    
    【为什么需要 Markdown？】
    - 保留标题层级（# ## ###），便于按章节切分
    - 保留表格结构，防止信息丢失
    - 保留列表格式，便于信息提取
    
    【润色的作用】
    PDF 转 Markdown 可能存在的问题：
    - 目录层级错乱
    - 表格上下行不对齐
    - 上下文断裂
    使用 LLM 润色可以修复这些格式问题。
    
    【核心操作】
    1. 使用 PyMuPDF4LLM 本地解析 PDF
    2. 使用 LLM 润色 Markdown 内容（修复格式错误）
    3. 保存润色后的文档到本地
    """
    print("\n" + "#" * 30 + " Step 3: 解析层优化 (PDF -> Markdown) " + "#" * 30 + "\n")
    pdf_path = './docs/内容公司各部门职责与关键角色联系信息汇总.pdf'
    md_content = file_to_md_local(pdf_path)
    print(f"\n📄 解析结果预览 (前300字符):\n{md_content[:300]}...")

    md_polished = md_polisher(md_content)
    if not md_polished: 
        md_polished = md_content 

    output_path = './docs/optimized_doc.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_polished)
    print(f"\n💾 润色后的文档已保存至: {output_path}")
    print("-" * 50)


def demo_step4_chunking(documents):
    """
    Step 4: 切片层优化 - 切片器大比拼。
    
    【目的】
    对比 5 种切片策略的效果。切片质量直接影响检索精度。
    
    【核心原理】
    文档切片的核心目标：在"信息完整性"和"检索粒度"之间取得平衡。
    - 切太大：一个 chunk 包含多个主题，语义混杂，检索不精准
    - 切太小：上下文不足，信息碎片化，理解困难
    
    【5 种切片策略详解】
    
    1. TokenTextSplitter（按 Token 数切分）
       - 原理：按固定 Token 数（如 512）强制切分
       - 优势：简单快速，适合英文
       - 劣势：可能在句子中间切断，语义不完整
       - 适用：原型验证、对精度要求不高的场景
    
    2. SentenceSplitter（按句子切分）
       - 原理：按句子边界（句号、问号、感叹号）切分，累积到指定 Token 数
       - 优势：保持句子完整性，语义连贯
       - 劣势：忽略段落和章节结构，可能将不相关的句子放在一起
       - 适用：通用场景，平衡了速度和质量
    
    3. MarkdownNodeParser（按 Markdown 结构切分）
       - 原理：按 Markdown 标题层级（#、##、###）切分
       - 优势：保留文档结构，同一章节的内容聚合在一起
       - 劣势：依赖文档格式质量，纯文本文档无法使用
       - 适用：结构化文档（技术文档、报告）
    
    4. SentenceWindowNodeParser（滑动窗口）
       - 原理：以一个句子为核心，前后各扩展 N 个句子作为上下文窗口
       - 优势：上下文丰富，每个 chunk 都有足够的前后文信息
       - 劣势：存储冗余（相邻 chunk 重叠度高），检索时需要后处理
       - 适用：需要丰富上下文的场景（如多跳推理）
    
    5. SemanticSplitterNodeParser（语义切分）
       - 原理：计算相邻句子的语义相似度，当相似度突降时切分
       - 优势：智能检测主题边界，每个 chunk 内部语义高度一致
       - 劣势：计算成本高（需要 Embedding），速度慢
       - 适用：对质量要求极高的场景（精准问答、法律文档）
    
    【评估维度】
    - Answer Correctness: 回答正确性
    - Context Recall: 检索召回率
    - Context Precision: 检索精确度（排序质量）
    
    Args:
        documents: 文档列表
    """
    print("\n" + "#" * 30 + " Step 4: 切片层优化 (Splitter Comparison) " + "#" * 30 + "\n")

    # 1. TokenTextSplitter
    token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=20)
    evaluate_splitter(token_splitter, documents, question, ground_truth, "TokenSplitter")

    # 2. SentenceSplitter
    sentence_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    evaluate_splitter(sentence_splitter, documents, question, ground_truth, "SentenceSplitter")

    # 3. MarkdownNodeParser
    markdown_splitter = MarkdownNodeParser()
    evaluate_splitter(markdown_splitter, documents, question, ground_truth, "MarkdownNodeParser")

    # 4. SentenceWindowNodeParser
    sentence_window_splitter = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    evaluate_splitter(
        sentence_window_splitter, 
        documents, 
        question, 
        ground_truth, 
        "SentenceWindow",
        node_postprocessors=[MetadataReplacementPostProcessor(target_metadata_key="window")]
    )

    # 5. SemanticSplitterNodeParser
    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model
    )
    evaluate_splitter(semantic_splitter, documents, question, ground_truth, "SemanticSplitter")


def demo_step5_embedding(documents):
    """
    Step 5: 嵌入层优化 - Embedding 模型对比。
    
    【目的】
    Embedding 模型是 RAG 系统的"语义基石"，直接决定了相似度计算的准确性。
    对比 text-embedding-v2 与 v3 在语义区分度上的差异。
    
    【核心原理】
    Embedding 模型将文本映射到高维向量空间（如 1536 维），语义相近的文本在向量空间中
    距离也近。向量检索本质是在高维空间中寻找"最近邻"。
    
    【Embedding 质量的三个维度】
    1. 语义区分度：相关文本距离近，无关文本距离远
    2. 向量维度：维度越高表达能力越强，但计算成本也越高
    3. 领域适配性：通用模型 vs 领域模型（如法律、医疗）
    
    【余弦相似度原理】
    - 公式：cos(θ) = (A · B) / (||A|| * ||B||)
    - 取值范围：[-1, 1]
      * 1.0：完全相同方向（语义完全一致）
      * 0.0：垂直（无关）
      * -1.0：完全相反方向（语义相反）
    - 优势：不受向量长度影响，只关注方向
    
    【V2 vs V3 对比】
    ┌──────────────┬────────────────┬────────────────┐
    │ 维度         │ V2             │ V3             │
    ├──────────────┼────────────────┼────────────────┤
    │ 向量维度     │ 1536           │ 1536           │
    │ 训练数据     │ 较少           │ 更多           │
    │ 语义区分度   │ 中等           │ 更高           │
    │ 多语言能力   │ 基础           │ 增强           │
    │ 长文本处理   │ 基础           │ 增强           │
    │ 适用场景     │ 通用           │ 高精度场景     │
    └──────────────┴────────────────┴────────────────┘
    
    【为什么 Embedding 模型很重要？】
    RAG 系统的召回率上限由 Embedding 模型决定：
    - 好的 Embedding：相关文档排在前面，Top-K 就能召回
    - 差的 Embedding：相关文档排在后面，即使 Top-K 很大也召回不了
    
    【对比角度】
    1. 向量运算原理演示（余弦相似度）
    2. 文本相似度对比（同一文本在不同模型下的表现）
    3. RAG 实战对比（完整的检索 + 生成流程）
    
    Args:
        documents: 文档列表
    """
    print("\n" + "#" * 30 + " Step 5: 嵌入层优化 (Embedding Comparison) " + "#" * 30 + "\n")

    # 1. 向量运算原理演示
    print("\n>>> 角度 1: 向量运算原理演示")
    a = np.array([0.2, 0.8])
    b = np.array([0.3, 0.7])
    c = np.array([0.8, 0.2])
    print(f"向量 A: {a}")
    print(f"向量 B: {b}")
    print(f"向量 C: {c}")
    print(f"A 与 B 的余弦相似度: {cosine_similarity(a, b):.4f}")
    print(f"B 与 C 的余弦相似度: {cosine_similarity(b, c):.4f}")
    print("-" * 30)

    # 2. 实战：不同模型的文本相似度
    print("\n>>> 角度 2: 不同模型的文本相似度")
    query_text = "张伟是哪个部门的"
    chunks = [
        "核，提供⾏政管理与协调⽀持，优化⾏政⼯作流程。 ⾏政部 秦⻜ 蔡静 G705 034 ⾏政 ⾏政专员 13800000034 qinf@educompany.com 维护公司档案与信息系统，负责公司通知及公告的发布",
        "组织公司活动的前期准备与后期评估，确保公司各项⼯作的顺利进⾏。 IT部 张伟 ⻢云 H802 036 IT⽀撑 IT专员 13800000036 zhangwei036@educompany.com 进⾏公司⽹络及硬件设备的配置"
    ]
    embedding_models_dict = {
        "text-embedding-v2": DashScopeEmbedding(model_name="text-embedding-v2"),
        "text-embedding-v3": DashScopeEmbedding(model_name="text-embedding-v3")
    }
    compare_embeddings(query_text, chunks, embedding_models_dict)

    # 3. 实战：RAG 效果对比
    sentence_splitter_for_embed = SentenceSplitter(chunk_size=1000, chunk_overlap=200)
    compare_embedding_models(
        documents=documents,
        question=question,
        ground_truth=ground_truth,
        sentence_splitter=sentence_splitter_for_embed
    )


def demo_step6_vector_db():
    """
    Step 6: 存储层优化 - 向量数据库选型。
    
    【目的】
    向量数据库是 RAG 系统的"记忆中枢"。不同数据库在存储效率、查询性能、
    扩展性等方面各有优劣。
    
    【选型建议】
    ┌──────────────┬────────────┬──────────────┬──────────────┐
    │ 数据库类型   │ 代表产品   │ 适用场景     │ 核心优势     │
    ├──────────────┼────────────┼──────────────┼──────────────┤
    │ 内存型       │ Faiss      │ 原型验证     │ 速度极快     │
    │ 嵌入式       │ Chroma     │ 单机应用     │ 部署简单     │
    │ 专用向量库   │ Milvus     │ 生产环境     │ 功能完整     │
    │ 传统DB扩展   │ Pgvector   │ 混合场景     │ 兼容现有架构 │
    │ 云原生       │ DashVector │ 快速上线     │ 免运维       │
    └──────────────┴────────────┴──────────────┴──────────────┘
    
    【本例使用】
    LlamaIndex 默认的内存 VectorStore（适合学习和原型）。
    """
    print("\n" + "#" * 30 + " Step 6: 存储层优化 (Vector DB Selection) " + "#" * 30 + "\n")
    print("📚 本步骤为理论说明，详见函数 docstring 中的选型建议。\n")
    print("💡 生产环境推荐：")
    print("  - 小规模（< 10万文档）：Chroma、LanceDB")
    print("  - 中规模（10万-1000万）：Milvus、Qdrant")
    print("  - 大规模（> 1000万）：云原生方案（DashVector、Pinecone）")
    print("  - 已有 PostgreSQL：Pgvector 扩展")


def demo_step7_retrieval(base_index):
    """
    Step 7: 检索层优化 - 进阶检索策略。
    
    【目的】
    单纯的 Top-K 向量检索往往不够精准。本步骤介绍 RAG 检索链路中
    "检索前"、"检索中"、"检索后"的三阶段优化策略。
    
    【核心原理】
    RAG 检索链路可以分为三个阶段，每个阶段都有对应的优化策略。
    
    【检索前（Pre-Retrieval）- 优化查询】
    
    1. 问题改写 (Query Rewriting)
       - 原理：用户问题往往过于简短或模糊（如"张伟是谁？"），直接检索效果不佳。
         通过 LLM 将问题扩写为更完整、更适合检索的形式。
       - 示例：
         * 原始："张伟是谁？"
         * 改写："张伟是哪个部门的员工？他的职位是什么？联系方式是什么？"
       - 效果：扩写后的问题包含更多关键词，提升召回率
    
    2. 多步查询 (Multi-step Query)
       - 原理：复杂问题（如"张伟和李四分别是哪个部门的？"）包含多个子问题。
         StepDecomposeQueryTransform 会将复杂问题分解为多个简单子问题序列，
         逐步检索并汇总答案。
       - 流程：
         * 步骤1：识别问题中的多个主体（张伟、李四）
         * 步骤2：为每个主体生成子问题
         * 步骤3：逐步检索并合并结果
       - 效果：避免遗漏信息，提升多跳问题的回答质量
    
    3. HyDE (Hypothetical Document Embeddings - 假设性文档嵌入)
       - 原理：有时问题（Query）和文档（Document）在语义空间中距离较远。
         * 问题很短："张伟是谁？"（几个字）
         * 文档很长："张伟，教研部专员，负责课程开发..."（几十字）
         HyDE 策略是先让 LLM 生成一个"假设性答案"，这个答案虽然可能包含幻觉，
         但在语义上与真实文档非常接近。然后用这个"假设答案"去检索。
       - 示例：
         * 问题："张伟是谁？"
         * 假设答案："张伟是教研部的一名专员，主要负责课程内容的开发和审核工作..."
         * 用假设答案的 Embedding 去检索 → 更容易召回相关文档
       - 效果：缩小问题-文档的语义距离，提升召回率
    
    【检索中（Retrieval）- 优化召回】
    
    4. 标签增强 (Tag Extraction + Metadata Filtering)
       - 原理：纯向量检索是"模糊匹配"，标签是"精确匹配"。
         提取实体标签（如人名、部门名）作为元数据过滤条件，可以实现"精确制导"。
       - 流程：
         * 步骤1：从问题中提取标签（如"张伟"、"教研部"）
         * 步骤2：从文档中提取标签并存储为元数据
         * 步骤3：检索时同时匹配向量相似度和标签精确度
       - 效果：过滤掉不相关文档，提升精确度
    
    【检索后（Post-Retrieval）- 优化排序】
    
    5. 重排序 (Rerank)
       - 原理：
         * Bi-Encoder（向量检索）：将 Query 和 Doc 分别编码为向量，计算余弦相似度。
           优势：速度快（百万文档毫秒级），适合粗排。
           劣势：Query 和 Doc 独立编码，无法捕捉深层交互。
         * Cross-Encoder（重排序）：将 Query 和 Doc 拼接后一起编码，捕捉深层语义交互。
           优势：精度极高，适合精排。
           劣势：计算量大（每对 Query-Doc 都要重新编码）。
       - 最佳实践：
         * 第一阶段：用 Bi-Encoder 粗排，从 100万 召回 Top-20
         * 第二阶段：用 Cross-Encoder 精排，从 Top-20 筛选 Top-3
       - 效果：在速度和精度间取得最佳平衡
    
    【优化策略总结】
    1. 问题改写: 扩写模糊问题，提升召回率
    2. 多步查询: 分解复杂问题，逐步检索
    3. HyDE: 生成假设性答案，缩小问题-文档的语义距离
    4. 标签增强: 提取实体标签，精确制导
    5. 重排序: Cross-Encoder 精排，提升 Top-N 质量
    
    Args:
        base_index: 向量索引（Step 2 创建的高质量索引）
    """
    print("\n" + "#" * 30 + " Step 7: 检索层优化 (Advanced Retrieval) " + "#" * 30 + "\n")

    # --- 7.1 问题改写 (Query Rewriting) ---
    print("\n>>> 7.1 问题改写 (Query Rewriting)")
    print(f"原始问题: {question}")
    rewritten_query = generate_queries(question)
    print(f"改写后问题: {rewritten_query}")

    query_engine = base_index.as_query_engine(similarity_top_k=5)
    print("\n[使用改写后的问题进行检索]")
    response = ask(rewritten_query, query_engine)
    show_evaluation_result(evaluate_result(rewritten_query, response, ground_truth), "7.1 问题改写评估")

    # --- 7.2 多步查询 (Multi-step Query) ---
    print("\n>>> 7.2 多步查询 (Multi-step Query)")
    step_decompose_transform = StepDecomposeQueryTransform(verbose=True)
    query_engine_multistep = MultiStepQueryEngine(
        query_engine=base_index.as_query_engine(similarity_top_k=5),
        query_transform=step_decompose_transform,
        index_summary="公司人员信息，包含姓名、部门、职位、邮箱等"
    )
    print(f"用户问题: {question}")
    print("🤖 AI正在进行多步查询分解...")
    response = ask(question, query_engine_multistep)
    show_evaluation_result(evaluate_result(question, response, ground_truth), "7.2 多步查询评估")

    # --- 7.3 HyDE (Hypothetical Document Embeddings) ---
    print("\n>>> 7.3 HyDE 假设性文档检索")
    hyde = HyDEQueryTransform(include_original=True)
    query_engine_hyde = TransformQueryEngine(
        query_engine=base_index.as_query_engine(similarity_top_k=5),
        query_transform=hyde
    )
    print("🤖 AI正在生成假设性文档并检索...")
    query_bundle = hyde(question)
    print(f"👻 生成的假设性文档:\n{query_bundle.embedding_strs[0][:200]}...\n")

    response = ask(question, query_engine_hyde)
    show_evaluation_result(evaluate_result(question, response, ground_truth), "7.3 HyDE 评估")

    # --- 7.4 标签增强 (Tag Extraction) - Demo ---
    print("\n>>> 7.4 标签增强 (Tag Extraction Demo)")
    sample_text = "张伟是IT部的技术骨干，负责公司网络安全。"
    print(f"分析文本: {sample_text}")
    print(f"提取标签: {extract_tags(sample_text)}")

    # --- 7.5 重排序 (Rerank) ---
    print("\n>>> 7.5 重排序 (Reranking)")
    query_engine_rerank = base_index.as_query_engine(
        similarity_top_k=20,  # 扩大召回
        node_postprocessors=[
            DashScopeRerank(top_n=3, model="gte-rerank"),
            SimilarityPostprocessor(similarity_cutoff=0.2)
        ]
    )
    response = ask(question, query_engine_rerank)
    show_evaluation_result(evaluate_result(question, response, ground_truth), "7.5 重排序评估")


def demo_step8_generation(base_index):
    """
    Step 8: 生成层优化 - LLM 参数与角色调优。
    
    【目的】
    RAG 的最后一公里是生成。针对不同场景，需要调整 LLM 的核心参数。
    
    【核心原理】
    LLM 生成过程本质是"概率采样"：模型预测下一个 token 的概率分布，然后按概率采样。
    不同参数控制采样策略，从而影响输出风格。
    
    【核心参数详解】
    
    1. Temperature (温度) - 控制输出随机性
       - 原理：温度越低，高概率 token 被选中的可能性越大；温度越高，低概率 token
         也有机会被选中。
       - 公式：P_i = exp(logit_i / T) / Σ exp(logit_j / T)
       - 效果：
         * T = 0.1 (低温)：几乎总是选择最高概率的 token → 输出稳定、严谨、可预测
         * T = 0.7 (中温)：平衡了稳定性和创造性
         * T = 1.5 (高温)：低概率 token 也有机会 → 输出发散、有创意、不可预测
       - 适用场景：
         * 事实问答、客服对话、代码生成 → 低温（0.1-0.3）
         * 文案创作、头脑风暴、故事续写 → 高温（0.7-1.0）
    
    2. Presence Penalty (存在惩罚) - 防止重复
       - 原理：对已经出现过的 token 施加惩罚，降低其被再次选中的概率。
       - 取值范围：[-2.0, 2.0]
         * 正值：鼓励模型谈论新话题，避免重复（如 0.5-1.0）
         * 负值：允许模型重复使用相同的词（如引用原文）
         * 0：不施加惩罚
       - 效果：
         * Penalty = 0.0：允许重复，适合需要严格引用原文的场景
         * Penalty = 0.5：适度惩罚，避免陈词滥调
         * Penalty = 1.0：强力惩罚，输出更具多样性但可能不够连贯
       - 适用场景：
         * 事实问答、文档摘要 → 低惩罚（0.0），允许引用原文
         * 创意写作、文案生成 → 高惩罚（0.5-1.0），避免重复
    
    3. Seed (随机种子) - 控制可复现性
       - 原理：固定随机种子后，相同的输入会产生相同的输出（Deterministic）。
       - 效果：
         * 设置 Seed = 42：每次运行结果完全一致，便于调试和评估
         * 不设置 Seed：每次运行结果不同，更具随机性
       - 适用场景：
         * A/B 测试、效果评估 → 设置固定 Seed
         * 生产环境、需要多样性 → 不设置 Seed
    
    4. Max Tokens (最大输出长度)
       - 原理：限制生成的最大 token 数。
       - 效果：
         * 512 tokens：适合简短问答
         * 1024-2048 tokens：适合长文本生成
       - 注意：过短可能导致回答不完整，过长消耗更多资源
    
    【场景对比】
    - 场景 1: 严谨模式（事实问答）
      * Temperature = 0.1 (低温，确保事实准确)
      * Presence Penalty = 0.0 (允许引用原文)
      * Seed = 42 (可复现)
      * Max Tokens = 512 (简短回答)
    
    - 场景 2: 创意模式（文案生成）
      * Temperature = 0.8 (高温，激发创造力)
      * Presence Penalty = 0.5 (避免陈词滥调)
      * Seed = None (允许随机发挥)
      * Max Tokens = 1024 (长文本)
    
    Args:
        base_index: 向量索引
    """
    print("\n" + "#" * 30 + " Step 8: 生成层优化 (LLM Tuning) " + "#" * 30 + "\n")

    # 场景 1: 严谨的事实问答
    print(">>> 场景 1: 严谨模式 (Temp=0.1, Seed=42, Penalty=0.0)")
    llm_factual = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True,
        temperature=0.1,      # 低温度，确保事实准确
        max_tokens=512,
        presence_penalty=0.0, # 不强行惩罚重复，允许引用原文
        seed=42               # 固定随机种子，确保结果可复现
    )
    # 临时替换 Settings
    original_llm = Settings.llm
    Settings.llm = llm_factual
    ask(question, base_index.as_query_engine(similarity_top_k=3))

    # 场景 2: 发散的创意生成
    print("\n>>> 场景 2: 创意模式 (Temp=0.8, Penalty=0.5)")
    llm_creative = OpenAILike(
        model="qwen-plus",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        is_chat_model=True,
        temperature=0.8,      # 高温度，激发创造力
        max_tokens=1024,
        presence_penalty=0.5, # 鼓励尝试新词汇，避免陈词滥调
        # seed 不设置，允许随机发挥
    )
    Settings.llm = llm_creative
    ask("请为公司年会写一个创意开场白", base_index.as_query_engine(similarity_top_k=3))

    # 恢复默认
    Settings.llm = original_llm


def main():
    """主程序：执行所有 RAG 优化步骤"""
    
    print("\n" + "="*60)
    print("  RAG 系统全链路性能优化实验")
    print("="*60)
    
    try:
        # Step 1: 基准测试
        demo_step1_baseline()
        
        # Step 2: 数据层优化（重建索引）
        documents_step2, index_step2 = demo_step2_reindex()
        
        # Step 3: 解析层优化（PDF -> Markdown）
        demo_step3_parsing()
        
        # Step 4: 切片层优化（切片器对比）
        demo_step4_chunking(documents_step2)
        
        # Step 5: 嵌入层优化（Embedding 模型对比）
        demo_step5_embedding(documents_step2)
        
        # Step 6: 存储层优化（理论说明）
        demo_step6_vector_db()
        
        # Step 7: 检索层优化（进阶检索策略）
        demo_step7_retrieval(index_step2)
        
        # Step 8: 生成层优化（LLM 参数调优）
        demo_step8_generation(index_step2)
        
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("  所有优化步骤演示完成！")
    print("="*60)
    print("\n【总结】")
    print("通过 8 个优化步骤，我们系统化地提升了 RAG 系统的质量。")
    print("\n【关键要点】")
    print("  ✓ 数据层: 扩大 Top-K、重建索引")
    print("  ✓ 解析层: PDF → Markdown 保留结构")
    print("  ✓ 切片层: 语义切片 > Token 切片")
    print("  ✓ 嵌入层: V3 模型 > V2 模型")
    print("  ✓ 检索层: 问题改写 + Rerank 提升精度")
    print("  ✓ 生成层: 事实问答用低温度，创意生成用高温度")
    print("\n【下一步】")
    print("  → 根据具体业务场景，选择合适的优化策略")
    print("  → 建立评估数据集，持续监控优化效果")
    print("  → 将优化策略集成到生产环境\n")


if __name__ == "__main__":
    main()
