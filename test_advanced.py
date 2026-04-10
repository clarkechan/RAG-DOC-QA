# _*_ coding: utf-8 _*_
# @Time    : 2026/4/10 9:24
# @Author  : ClarkeChan
# @File    : test_advanced.py
# @Project : rag-doc-qa

from rag_chain import RAGAssistant, compare_models

# ===== 功能1：多轮对话 =====
print("=" * 60)
print("功能1：多轮对话")
print("=" * 60)

assistant = RAGAssistant(model_name="deepseek")

# 第1轮
r1 = assistant.ask("螺纹孔异物怎么判定？")
print(f"问: 螺纹孔异物怎么判定？")
print(f"答: {r1['answer']}\n")

# 第2轮（追问，需要理解上下文）
r2 = assistant.ask("如果是B级缺陷，具体怎么处理？")
print(f"问: 如果是B级缺陷，具体怎么处理？")
print(f"答: {r2['answer']}\n")

# 第3轮（继续追问）
r3 = assistant.ask("AI置信度低于阈值时呢？")
print(f"问: AI置信度低于阈值时呢？")
print(f"答: {r3['answer']}\n")


# ===== 功能2：模型切换 =====
print("\n" + "=" * 60)
print("功能2：模型切换")
print("=" * 60)

assistant.clear_history()
assistant.switch_model("glm-4.5-air")
r4 = assistant.ask("产线出现Level 1异常怎么办？")
print(f"GLM回答: {r4['answer']}\n")

assistant.switch_model("deepseek")
r5 = assistant.ask("产线出现Level 1异常怎么办？")
print(f"DeepSeek回答: {r5['answer']}\n")


# ===== 功能3：模型对比 =====
print("\n" + "=" * 60)
print("功能3：同一问题多模型对比")
print("=" * 60)

compare_models(
    "5Why分析法怎么用？举个例子",
    models=["deepseek", "glm-4.5-air"]
)


# ===== 功能4：检索参数调整 =====
print("\n" + "=" * 60)
print("功能4：top_k对比")
print("=" * 60)

for k in [1, 3, 5]:
    assistant_k = RAGAssistant(model_name="deepseek", top_k=k)
    r = assistant_k.ask("AI系统故障怎么处理？")
    print(f"\ntop_k={k} → 引用来源数: {len(r['sources'])}")
    print(f"来源: {r['sources']}")
    print(f"回答前100字: {r['answer'][:100]}...")