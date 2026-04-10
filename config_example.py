# _*_ coding: utf-8 _*_
# @Time    : 2026/4/2 17:16
# @Author  : ClarkeChan
# @File    : config.py
# @Project : rag-doc-qa

from langchain_openai import ChatOpenAI

DEEPSEEK_API_KEY = "your-key"
GLM_API_KEY = "your-key"

# 模型配置
MODELS = {
    "glm-4.5-air": {
        "api_key": GLM_API_KEY,
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.5-air"
    },
    "glm-4.6v": {
        "api_key": GLM_API_KEY,
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.6v"
    },
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    }
}

def get_llm(model_name="deepseek"):
    cfg = MODELS[model_name]
    return ChatOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=0
    )

# for name in ["deepseek", "glm-4.5-air", "glm-4.6v"]:
#     print(f"\n--- {name} ---")
#     llm = get_llm(name)
#     response = llm.invoke("用一句话介绍RAG技术")
#     print(response.content)
