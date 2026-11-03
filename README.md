# 🏭 工业文档智能问答系统（RAG-DOC-QA）

> 上传 PDF / Word / TXT 文档，即可基于文档内容进行多轮问答。回答会标注引用来源，并支持 DeepSeek 与 GLM 多模型对比。

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Chroma](https://img.shields.io/badge/VectorDB-Chroma-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

---

## ✨ 功能

| 功能 | 说明 |
| --- | --- |
| 📄 多格式文档 | 支持 PDF / DOCX / TXT 上传，自动选择加载器 |
| 🔍 语义检索 | 中文 Embedding `BAAI/bge-large-zh-v1.5` + Chroma 向量库 |
| 📎 引用溯源 | 回答中标注 `[引用N]`，并展示来源文件与相似度分数 |
| 💬 多轮对话 | 自动维护对话历史，支持追问 |
| ⚖️ 多模型对比 | 同一问题同时交给 DeepSeek / GLM-4.5-Air / GLM-4.6V 回答，并排对比 |
| 🎚️ 检索调参 | 界面上可实时调整 `top_k` |

## 🏗️ 工作流程

```
文档上传 (PDF / DOCX / TXT)
   │  doc_loader.py
   ▼
文本切片  chunk_size=500, overlap=100，按中文标点优先分割
   │
   ▼
向量化  bge-large-zh-v1.5  →  Chroma 持久化 (chroma_db/)
   │  vector_store.py
   ▼
用户提问 → 检索 top_k 文本块 → 拼接对话历史 → LLM 生成带引用的回答
                                               rag_chain.py
```

## 📁 项目结构

```
RAG-DOC-QA/
├── app.py              # Streamlit 界面（入口）
├── doc_loader.py       # 文档加载与切片
├── vector_store.py     # 向量库构建 / 加载 / 检索
├── rag_chain.py        # RAGAssistant：多轮问答、模型切换、多模型对比
├── config_example.py   # 模型与 API Key 配置模板
├── requirements.txt
├── data/               # 示例文档（模拟的工业质量标准）
└── test*.py            # 各模块的分步测试脚本
```

## 🚀 快速开始

**1. 安装依赖**

```bash
git clone https://github.com/clarkechan/RAG-DOC-QA.git
cd RAG-DOC-QA
python -m venv .venv
.venv\Scripts\activate        # Windows（macOS/Linux：source .venv/bin/activate）
pip install -r requirements.txt
```

**2. 配置 API Key**

```bash
cp config_example.py config.py
```

在 `config.py` 中填入 `DEEPSEEK_API_KEY` 和 `GLM_API_KEY`（只用其中一个模型的话，填一个即可）。

**3. 启动**

```bash
streamlit run app.py
```

打开页面后，在左侧点击 **「📂 使用默认文档构建知识库」**，或者上传自己的文档，然后就可以开始提问。

> 首次运行会从 Hugging Face 下载 Embedding 模型（约 1.3 GB），需要等待一段时间。

## 💡 示例问题

基于 `data/` 中的示例文档：

- 螺纹孔异物怎么判定？
- 如果是 B 级缺陷，具体怎么处理？（追问）
- AI 检测系统故障了怎么办？
- 产线出现 Level 1 异常怎么办？

## 🧪 分步测试脚本

| 脚本 | 用途 |
| --- | --- |
| `test.py` | 查看文档切片结果 |
| `test_embedding.py` | 验证 Embedding 模型与向量维度 |
| `test_rag.py` | 构建向量库，并用两个模型回答固定问题 |
| `test_advanced.py` | 演示多轮对话、模型切换、多模型对比、`top_k` 对比 |

## 🔧 技术栈

LangChain · Chroma · HuggingFace Embeddings (`bge-large-zh-v1.5`) · DeepSeek / 智谱 GLM（OpenAI 兼容接口）· Streamlit

## 👤 作者

**Clarke Chan（陈星宇）** · [Blog](https://mrcxy.com) · [GitHub](https://github.com/clarkechan)
