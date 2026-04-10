# _*_ coding: utf-8 _*_
# @Time    : 2026/4/9 8:50
# @Author  : ClarkeChan
# @File    : test_embedding.py
# @Project : rag-doc-qa

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5"
)

text = "螺纹孔异物缺陷检测标准"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")
print(f"前5个数值: {vector[:5]}")