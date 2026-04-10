# _*_ coding: utf-8 _*_
# @Time    : 2026/4/9 15:58
# @Author  : ClarkeChan
# @File    : test_rag
# @Project : rag-doc-qa

from doc_loader import load_directory, split_documents
from vector_store import build_vector_store
from rag_chain import ask

# ===== 第一步：构建向量数据库（只需运行一次）=====
print("=" * 50)
print("第一步：构建向量数据库")
print("=" * 50)
docs = load_directory("data")
chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
build_vector_store(chunks)

# ===== 第二步：测试问答 =====
questions = [
    "螺纹孔异物怎么判定？",
    "AI检测系统故障了怎么办？",
    "SN-2产品检测要注意什么？",
]

for model in ["deepseek", "glm-4.5-air"]:
    print(f"\n{'=' * 50}")
    print(f"模型: {model}")
    print(f"{'=' * 50}")

    for q in questions:
        print(f"\n问题: {q}")
        result = ask(q, model_name=model)
        print(f"回答: {result['answer']}")
        print(f"引用来源: {result['sources']}")
        print("-" * 40)