from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from config import get_llm
from vector_store import load_vector_store


def format_docs_with_source(docs_with_scores):
    """格式化检索结果，附带来源信息"""
    formatted = []
    for i, (doc, score) in enumerate(docs_with_scores):
        source = doc.metadata.get("source", "未知")
        formatted.append(f"[引用{i + 1}] 来源: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def format_chat_history(history):
    """格式化对话历史"""
    if not history:
        return "无"
    lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            lines.append(f"用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"助手: {msg.content[:200]}...")
    return "\n".join(lines)


# 支持多轮对话的Prompt
RAG_PROMPT = ChatPromptTemplate.from_template("""你是一个工业质量领域的专家助手。请根据参考文档和对话历史回答用户的问题。

要求：
1. 只基于参考文档中的信息回答，不要编造
2. 在回答中标注引用来源，例如 [引用1]
3. 如果文档中没有相关信息，明确告知用户
4. 如果用户的问题是对上一轮回答的追问，结合对话历史理解上下文

对话历史：
{history}

参考文档：
{context}

用户问题：{question}

回答：""")


class RAGAssistant:
    """支持多轮对话、模型切换、检索调参的RAG助手"""

    def __init__(self, model_name="deepseek", top_k=3, chunk_size=500, chunk_overlap=100):
        self.model_name = model_name
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chat_history = []  # 对话历史
        self.vectorstore = load_vector_store()

    def ask(self, query):
        """问答，自动维护对话历史"""
        if not self.vectorstore:
            return {"answer": "向量数据库未构建", "sources": [], "model": self.model_name}

        # 1. 检索
        results = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        context = format_docs_with_source(results)
        history_text = format_chat_history(self.chat_history)

        # 2. 生成
        llm = get_llm(self.model_name)
        chain = RAG_PROMPT | llm
        response = chain.invoke({
            "context": context,
            "history": history_text,
            "question": query
        })

        # 3. 更新对话历史
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response.content))

        return {
            "answer": response.content,
            "sources": [(doc.metadata.get("source", "未知"), round(score, 4))
                        for doc, score in results],
            "model": self.model_name
        }

    def switch_model(self, model_name):
        """切换模型"""
        self.model_name = model_name
        print(f"已切换到模型: {model_name}")

    def set_top_k(self, top_k):
        """调整检索数量"""
        self.top_k = top_k
        print(f"检索数量已调整为: {top_k}")

    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        print("对话历史已清空")

    def get_history(self):
        """查看对话历史"""
        return self.chat_history


def compare_models(query, models=None, top_k=3):
    """多模型对比同一问题"""
    if models is None:
        models = ["deepseek", "glm-4.5-air"]

    vectorstore = load_vector_store()
    if not vectorstore:
        print("向量数据库未构建")
        return

    results = vectorstore.similarity_search_with_score(query, k=top_k)
    context = format_docs_with_source(results)

    print(f"问题: {query}")
    print(f"检索到 {len(results)} 个相关文档块")
    print("=" * 60)

    responses = {}
    for model_name in models:
        llm = get_llm(model_name)
        chain = RAG_PROMPT | llm
        response = chain.invoke({
            "context": context,
            "history": "无",
            "question": query
        })
        responses[model_name] = response.content

        print(f"\n{'─' * 60}")
        print(f"模型: {model_name}")
        print(f"{'─' * 60}")
        print(response.content)

    return responses