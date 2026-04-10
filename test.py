# _*_ coding: utf-8 _*_
# @Time    : 2026/4/9 10:48
# @Author  : ClarkeChan
# @File    : test
# @Project : rag-doc-qa

from doc_loader import load_directory, split_documents

# 加载文档
docs = load_directory("data")

# 切片
chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

# 查看前3个切片
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 第{i+1}块 ---")
    print(f"来源: {chunk.metadata.get('source', '未知')}")
    print(f"内容: {chunk.page_content[:100]}...")