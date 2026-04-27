"""
Layer 3: 端到端质量评估
指标：Faithfulness, AnswerRelevance, ContextUtilization
使用LLM-as-Judge，低成本版本
"""
import json
import sys
import os
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from main import RecipeRAGSystem, RAGConfig

load_dotenv()


class End2EndEvaluator:
    """端到端评估器 - 使用简单LLM打分而非RAGAS"""

    # 评分Prompt模板
    FAITHFULNESS_PROMPT = """你是一位严格的烹饪知识审核员。请判断系统回答是否忠实于提供的食谱信息。

评分标准：
5分：回答完全基于食谱信息，没有任何编造或 extrapolation
4分：回答基本基于食谱，有少量合理推断但不影响准确性
3分：回答部分基于食谱，有部分内容无法从食谱中验证
2分：回答有明显内容与食谱矛盾
1分：回答完全脱离食谱，大量编造

用户问题：{question}

食谱信息：
{context}

系统回答：{answer}

请只输出一个1-5的整数分数，不要任何解释："""

    USEFULNESS_PROMPT = """你是一位用户体验设计师。请判断系统回答对用户问题的帮助程度。

评分标准：
5分：回答完全解决了用户问题，信息完整、实用、可操作
4分：回答基本解决了问题，缺少少量细节
3分：回答部分解决了问题，有信息缺失
2分：回答与问题相关但几乎没有实用信息
1分：回答完全无关或无法使用

用户问题：{question}
系统回答：{answer}

请只输出一个1-5的整数分数，不要任何解释："""

    STRUCTURE_PROMPT = """你是一位内容编辑。请判断系统回答的结构清晰度。

评分标准：
5分：结构清晰，有明确标题、分步骤、重点突出
4分：结构较好，有分段但标题不够明确
3分：结构一般，内容混在一起但还能读
2分：结构混乱，难以找到关键信息
1分：完全没有结构，一团乱麻

系统回答：{answer}

请只输出一个1-5的整数分数，不要任何解释："""

    def __init__(self, rag_system: RecipeRAGSystem):
        self.rag = rag_system
        self.llm = rag_system.generation_module.llm

    def evaluate_single(self, question: str, answer: str, context_docs: List) -> Dict:
        """评估单条回答"""
        context = self._build_context(context_docs)

        # 三个维度打分
        faithfulness = self._score(
            self.FAITHFULNESS_PROMPT.format(
                question=question, context=context, answer=answer
            )
        )

        usefulness = self._score(
            self.USEFULNESS_PROMPT.format(
                question=question, answer=answer
            )
        )

        structure = self._score(
            self.STRUCTURE_PROMPT.format(answer=answer)
        )

        return {
            'faithfulness': faithfulness,
            'usefulness': usefulness,
            'structure': structure,
            'overall': round((faithfulness + usefulness + structure) / 3, 2)
        }

    def _score(self, prompt: str) -> int:
        """调用LLM打分"""
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # 提取数字
            import re
            numbers = re.findall(r'\d+', content)
            if numbers:
                score = int(numbers[0])
                return max(1, min(5, score))  # 限制在1-5
            return 3  # 默认中等
        except Exception as e:
            print(f"打分失败: {e}")
            return 3

    def _build_context(self, docs) -> str:
        """构建上下文字符串"""
        parts = []
        for i, doc in enumerate(docs[:3], 1):
            parts.append(f"[食谱{i}] {doc.metadata.get('dish_name', '未知')}\n{doc.page_content[:500]}")
        return "\n\n".join(parts)

    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """批量评估"""
        results = []

        print(f"\n{'='*60}")
        print(f"🎯 端到端质量评估 - 共 {len(test_cases)} 条")
        print(f"{'='*60}\n")

        for i, case in enumerate(test_cases, 1):
            question = case['question']
            print(f"[{i}/{len(test_cases)}] 评估: {question[:40]}...")

            # 生成回答
            try:
                answer = self.rag.ask_question(question)
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
                continue

            # 获取检索到的文档（用于faithfulness评估）
            try:
                rewritten = self.rag.generation_module.query_rewrite(question)
                chunks = self.rag.retrieval_module.hybrid_search(rewritten)
                parent_docs = self.rag.data_module.get_parent_documents(chunks)
            except:
                parent_docs = []

            # 评估
            scores = self.evaluate_single(question, answer, parent_docs)

            result = {
                'question': question,
                'answer': answer,
                'scores': scores
            }
            results.append(result)

            print(f"  ✅ 忠实度:{scores['faithfulness']} 实用性:{scores['usefulness']} "
                  f"结构:{scores['structure']} 综合:{scores['overall']}")

        # 汇总
        summary = {
            'faithfulness_avg': round(sum(r['scores']['faithfulness'] for r in results) / len(results), 2),
            'usefulness_avg': round(sum(r['scores']['usefulness'] for r in results) / len(results), 2),
            'structure_avg': round(sum(r['scores']['structure'] for r in results) / len(results), 2),
            'overall_avg': round(sum(r['scores']['overall'] for r in results) / len(results), 2),
            'details': results
        }

        return summary

    def print_summary(self, summary: Dict):
        """打印摘要"""
        print(f"\n{'='*60}")
        print("📊 端到端评估摘要")
        print(f"{'='*60}")
        print(f"忠实度 (Faithfulness): {summary['faithfulness_avg']}/5.0")
        print(f"实用性 (Usefulness):   {summary['usefulness_avg']}/5.0")
        print(f"结构度 (Structure):    {summary['structure_avg']}/5.0")
        print(f"综合评分:              {summary['overall_avg']}/5.0")

        # 找出最好和最差的
        details = summary['details']
        best = max(details, key=lambda x: x['scores']['overall'])
        worst = min(details, key=lambda x: x['scores']['overall'])

        print(f"\n🏆 最佳回答 [{best['scores']['overall']}分]:")
        print(f"  问题: {best['question']}")
        print(f"  回答: {best['answer'][:100]}...")

        print(f"\n💩 最差回答 [{worst['scores']['overall']}分]:")
        print(f"  问题: {worst['question']}")
        print(f"  回答: {worst['answer'][:100]}...")


def run_end2end_eval():
    """运行端到端评估"""
    api_key = os.getenv("MOONSHOT_API_KEY")

    # 初始化RAG
    print("🚀 初始化RAG系统...")
    config = RAGConfig(moonshot_api_key=api_key)
    rag = RecipeRAGSystem(config=config)
    rag.initialize_system()
    rag.build_knowledge_base()

    # 测试用例
    test_cases = [
        {"question": "宫保鸡丁怎么做？"},
        {"question": "推荐几个简单的素菜"},
        {"question": "水煮鱼需要哪些食材？"},
        {"question": "怎么做蛋糕？"},
        {"question": "适合健身吃的家常菜推荐"},
    ]

    # 评估
    evaluator = End2EndEvaluator(rag)
    summary = evaluator.evaluate_batch(test_cases)
    evaluator.print_summary(summary)

    # 保存
    output_path = Path(__file__).parent / "end2end_eval_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {output_path}")

    return summary


if __name__ == "__main__":
    run_end2end_eval()
