# _*_ coding: utf-8 _*_
# @Time    : 2026/4/2 17:17
# @Author  : ClarkeChan
# @File    : vector_store.py
# @Project : rag-doc-qa

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

CHROMA_DIR = "chroma_db"


def build_vector_store(chunks):
    """从文本块构建向量数据库"""
    # 如果已有数据库，先删除重建
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"向量数据库构建完成: {len(chunks)} 个文本块已存储")
    return vectorstore


def load_vector_store():
    """加载已有的向量数据库"""
    if not os.path.exists(CHROMA_DIR):
        print("向量数据库不存在，请先构建")
        return None

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print("向量数据库加载完成")
    return vectorstore


def search(vectorstore, query, top_k=3):
    """检索最相关的文档块，返回内容+来源+相似度分数"""
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    for i, (doc, score) in enumerate(results):
        print(f"\n--- 检索结果 {i + 1} (相似度: {score:.4f}) ---")
        print(f"来源: {doc.metadata.get('source', '未知')}")
        print(f"内容: {doc.page_content[:150]}...")

    return results