import streamlit as st
import os
from doc_loader import load_single_file, split_documents
from vector_store import build_vector_store, load_vector_store, embeddings
from rag_chain import RAGAssistant, compare_models

# ===== 页面配置 =====
st.set_page_config(
    page_title="工业文档智能问答系统",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 工业文档智能问答系统")
st.caption("基于 RAG 架构 | 支持 DeepSeek / GLM 多模型对比")

# ===== 侧边栏 =====
with st.sidebar:
    st.header("⚙️ 设置")

    # 模型选择
    model_name = st.selectbox(
        "选择模型",
        ["deepseek", "glm-4.5-air", "glm-4.6v"],
        index=0
    )

    # 检索参数
    top_k = st.slider("检索文档数量 (top_k)", 1, 5, 3)

    st.divider()

    # 文档上传
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "上传文档（支持 PDF / TXT / DOCX）",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("📥 处理并构建知识库"):
        with st.spinner("正在处理文档..."):
            all_docs = []

            # 保存上传文件到临时目录
            upload_dir = "uploaded_docs"
            os.makedirs(upload_dir, exist_ok=True)

            for file in uploaded_files:
                file_path = os.path.join(upload_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                docs = load_single_file(file_path)
                all_docs.extend(docs)
                st.sidebar.success(f"✅ {file.name}: {len(docs)} 页/段")

            # 切片 + 构建向量数据库
            chunks = split_documents(all_docs)
            build_vector_store(chunks)

            st.sidebar.success(f"🎉 知识库构建完成！共 {len(chunks)} 个文本块")

    # 使用默认文档
    if st.button("📂 使用默认文档构建知识库"):
        if os.path.exists("data"):
            with st.spinner("正在处理默认文档..."):
                from doc_loader import load_directory

                docs = load_directory("data")
                chunks = split_documents(docs)
                build_vector_store(chunks)
                st.sidebar.success(f"🎉 知识库构建完成！共 {len(chunks)} 个文本块")
        else:
            st.sidebar.error("data/ 目录不存在")

    st.divider()

    # 知识库状态
    st.header("📊 系统状态")
    vectorstore = load_vector_store()
    if vectorstore:
        collection = vectorstore._collection
        st.success(f"知识库已就绪: {collection.count()} 个文本块")
    else:
        st.warning("知识库未构建，请先上传文档")

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.session_state.assistant = None
        st.rerun()

# ===== 初始化会话状态 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state or st.session_state.get("current_model") != model_name:
    st.session_state.assistant = RAGAssistant(model_name=model_name, top_k=top_k)
    st.session_state.current_model = model_name

# 更新参数
st.session_state.assistant.model_name = model_name
st.session_state.assistant.top_k = top_k

# ===== 主界面：两栏布局 =====
col_chat, col_compare = st.tabs(["💬 智能问答", "⚖️ 模型对比"])

# ----- Tab 1: 问答对话 -----
with col_chat:
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📎 引用来源"):
                    for source, score in msg["sources"]:
                        st.text(f"• {source} (相似度: {score})")

    # 用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 检查知识库
        if not load_vector_store():
            st.error("请先在左侧构建知识库")
        else:
            # 显示用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 生成回答
            with st.chat_message("assistant"):
                with st.spinner(f"正在使用 {model_name} 思考..."):
                    result = st.session_state.assistant.ask(prompt)

                st.markdown(result["answer"])

                # 显示引用来源
                with st.expander("📎 引用来源"):
                    for source, score in result["sources"]:
                        st.text(f"• {source} (相似度: {score})")

                st.caption(f"模型: {result['model']} | top_k: {top_k}")

            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })

# ----- Tab 2: 模型对比 -----
with col_compare:
    st.subheader("同一问题，多模型回答对比")

    compare_query = st.text_input("输入对比问题", placeholder="例如：AI检测系统故障了怎么办？")

    compare_models_list = st.multiselect(
        "选择对比模型",
        ["deepseek", "glm-4.5-air", "glm-4.6v"],
        default=["deepseek", "glm-4.5-air"]
    )

    if st.button("🔍 开始对比") and compare_query:
        if not load_vector_store():
            st.error("请先在左侧构建知识库")
        else:
            cols = st.columns(len(compare_models_list))

            for i, m in enumerate(compare_models_list):
                with cols[i]:
                    st.markdown(f"**{m}**")
                    with st.spinner(f"{m} 生成中..."):
                        assistant_temp = RAGAssistant(model_name=m, top_k=top_k)
                        result = assistant_temp.ask(compare_query)
                    st.markdown(result["answer"])
                    with st.expander("引用来源"):
                        for source, score in result["sources"]:
                            st.text(f"• {source} ({score})")