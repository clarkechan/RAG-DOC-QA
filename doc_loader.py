# _*_ coding: utf-8 _*_
# @Time    : 2026/4/2 17:16
# @Author  : ClarkeChan
# @File    : doc_loader.py
# @Project : rag-doc-qa

import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_single_file(file_path):
    """根据文件类型自动选择加载器"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        print(f"不支持的文件类型: {ext}")
        return []

    return loader.load()


def load_directory(dir_path):
    """加载目录下所有支持的文档"""
    all_docs = []
    supported = [".pdf", ".txt", ".docx", ".doc"]

    for filename in os.listdir(dir_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in supported:
            file_path = os.path.join(dir_path, filename)
            docs = load_single_file(file_path)
            print(f"加载 {filename}: {len(docs)} 页/段")
            all_docs.extend(docs)

    print(f"\n共加载 {len(all_docs)} 个文档段")
    return all_docs


def split_documents(docs, chunk_size=500, chunk_overlap=100):
    """文档切片"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " "]  # 中文优先分割符
    )
    chunks = splitter.split_documents(docs)
    print(f"切片完成: {len(docs)} 个文档 → {len(chunks)} 个文本块")
    print(f"参数: chunk_size={chunk_size}, overlap={chunk_overlap}")
    return chunks