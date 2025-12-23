"""
文档处理工具模块。

提供文档解析和处理功能，包括：
- PDF 转 Markdown
- Markdown 内容润色
"""

import pymupdf4llm
from dashscope import Generation


def file_to_md_local(file_path):
    """
    使用 PyMuPDF4LLM 本地解析 PDF 为 Markdown。
    
    【优势】
    - 速度快，无需网络调用
    - 能保留表格、标题等结构信息
    - 轻量级，适合小规模文档
    
    Args:
        file_path: PDF 文件路径
        
    Returns:
        str: Markdown 格式文本
    """
    print(f"🚀 正在使用 PyMuPDF4LLM 本地解析: {file_path}")
    md_text = pymupdf4llm.to_markdown(file_path)
    return md_text


def md_polisher(data):
    """
    使用 LLM 润色 Markdown 内容。
    
    【功能】
    修复 PDF 转 Markdown 过程中的常见问题：
    - 目录层级错误
    - 表格格式混乱
    - 上下文不连贯
    
    Args:
        data: 原始 Markdown 文本
        
    Returns:
        str: 润色后的 Markdown 文本
    """
    if not data: 
        return ""
    
    messages = [
        {'role': 'user', 'content': '下面这段文本是由pdf转为markdown的，格式和内容可能存在一些问题，需要你帮我优化下：\n1、目录层级，如果目录层级顺序不对请以markdown形式补全或修改；\n2、内容错误，如果存在上下文不一致的情况，请你修改下；\n3、如果有表格，注意上下行不一致的情况；\n4、输出文本整体应该与输入没有较大差异，不要自己制造内容，我是需要对原文进行润色；\n4、输出格式要求：markdown文本，你的所有回答都应该放在一个markdown文件里面。\n特别注意：只输出转换后的 markdown 内容本身，不输出任何其他信息。\n需要处理的内容是：' + data[:2000]} 
    ]
    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        result_format='message',
        stream=True,
        incremental_output=True
    )
    result = ""
    print("\n📝 正在润色 Markdown 文本...")
    print("-" * 50)
    for chunk in response:
        content = chunk.output.choices[0].message.content
        print(content, end='')
        result += content
    print("\n" + "-" * 50)
    return result

