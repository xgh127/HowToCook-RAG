"""
RAG 全自动测评脚本（兼容你的菜谱RAG系统）
无需GT | 无需标注 | 一键测评 | 自动出分 | 自动保存报告
"""
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from openai import OpenAI  # Kimi 兼容 OpenAI 接口

# 加载你的系统
sys.path.append(str(Path(__file__).parent.parent))  # 确保能导入上层目录的模块
from dotenv import load_dotenv
from config import RAGConfig
from main import RecipeRAGSystem  # 直接调用你的系统！

# 加载环境变量
load_dotenv()
moonshot_api_key = os.getenv("MOONSHOT_API_KEY")

# ----------------------
# RAGAS 测评依赖
# ----------------------
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # 事实忠实度（不胡说八道）
    answer_relevancy,    # 回答相关性
    context_precision,   # 检索精准度
    context_recall,      # 检索完整度
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

# 加载环境变量
load_dotenv()
moonshot_api_key = os.getenv("MOONSHOT_API_KEY")

# ======================
# 🔥 关键：连接 Kimi 模型
# ======================
client = OpenAI(
    api_key=moonshot_api_key,
    base_url="https://api.moonshot.cn/v1"  # Kimi 官方接口
)

# ======================
# 测试集（可自由扩充）
# ======================
TEST_QUERIES = [
    "宫保鸡丁怎么做？",
    "推荐适合健身吃的家常菜",
    "水煮鱼需要哪些食材？",
    "简单的素菜有哪些？",
    "咖喱炒蟹的制作步骤是什么？",
]

# ======================
# LLM-as-Judge 评分 Prompt（论文标准）
# ======================
SCORING_PROMPT = """
你是一位专业的RAG系统评估专家，请按照标准对模型回答进行评分。

评分标准（1-5分，5分为最好）：
1. 事实忠实度 (Faithfulness)：回答是否完全基于上下文，无编造。
2. 回答相关性 (Answer Relevancy)：回答是否切题，不答非所问。
3. 上下文精准度 (Context Precision)：检索文档是否高度相关。
4. 上下文完整度 (Context Recall)：上下文是否包含回答所需全部信息。

用户问题：{question}
上下文：{context}
模型回答：{answer}

请只返回JSON格式，不要额外解释：
{{
    "faithfulness": 分数,
    "answer_relevancy": 分数,
    "context_precision": 分数,
    "context_recall": 分数
}}
"""

# ======================
# 🔥 Kimi 打分函数
# ======================
def kimi_judge(question, answer, context):
    prompt = SCORING_PROMPT.format(
        question=question,
        answer=answer,
        context="\n---\n".join(context)
    )

    response = client.chat.completions.create(
        model="moonshot-v1-8k",  # Kimi 官方模型
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}  # 强制JSON输出
    )

    return json.loads(response.choices[0].message.content)

# ======================
# 测评主流程
# ======================
def run_evaluation(rag_system):
    print("🚀 启动 Kimi 全自动 RAG 测评...\n")
    results = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] 测试问题：{query}")
        
        # 调用你的 RAG 系统
        answer = rag_system.ask_question(query)
        
        # 获取检索上下文
        rewritten = rag_system.generation_module.query_rewrite(query)
        chunks = rag_system.retrieval_module.hybrid_search(rewritten, top_k=3)
        contexts = [c.page_content for c in chunks]

        # Kimi 评分
        score = kimi_judge(query, answer, contexts)
        results.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "scores": score
        })

        print(f"✅ 评分完成：{score}\n")

    # 保存报告
    save_report(results)

# ======================
# 保存报告
# ======================
def save_report(results):
    # 自动创建 output 目录
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rag_eval_kimi_{now}.json"
    save_path = output_dir / filename

    avg_scores = {
        "faithfulness": 0, "answer_relevancy": 0,
        "context_precision": 0, "context_recall": 0
    }
    for r in results:
        for k in avg_scores:
            avg_scores[k] += r["scores"][k]
    for k in avg_scores:
        avg_scores[k] = round(avg_scores[k] / len(results), 2)

    report = {
        "timestamp": now,
        "average_scores": avg_scores,
        "details": results
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📊 平均分：{avg_scores}")
    print(f"✅ 报告已保存到：\n{save_path}")

# ======================
# 运行
# ======================
if __name__ == "__main__":
    config = RAGConfig(moonshot_api_key=moonshot_api_key)
    rag = RecipeRAGSystem(config=config)
    rag.initialize_system()
    rag.build_knowledge_base()
    run_evaluation(rag)