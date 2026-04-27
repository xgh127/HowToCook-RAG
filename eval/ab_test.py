"""
A/B测试框架 - 对比两个RAG配置的效果
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))
from main import RecipeRAGSystem, RAGConfig


class ABTestFramework:
    """A/B测试框架"""

    JUDGE_PROMPT = """你是一位公正的评估专家。请对比两个版本的回答，判断哪个更好。

用户问题：{question}

【版本A】
{answer_a}

【版本B】
{answer_b}

评估维度：
1. 准确性：信息是否正确、是否与食谱一致
2. 完整性：是否回答了用户的全部问题
3. 实用性：用户能否按照回答实际操作
4. 清晰度：结构是否清晰、易于理解

请输出JSON格式：
{{
    "winner": "A" or "B" or "tie",
    "reason": "简要说明理由",
    "a_score": 1-5,
    "b_score": 1-5
}}
"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def run_test(self, config_a: Dict, config_b: Dict, test_queries: List[str]) -> Dict:
        """
        运行A/B测试

        Args:
            config_a: 版本A的配置参数
            config_b: 版本B的配置参数
            test_queries: 测试问题列表
        """
        print(f"\n{'='*60}")
        print("🆚 A/B测试")
        print(f"{'='*60}")
        print(f"版本A配置: {config_a}")
        print(f"版本B配置: {config_b}")
        print(f"测试用例数: {len(test_queries)}\n")

        # 初始化两个系统
        print("初始化版本A...")
        rag_a = self._create_rag(config_a)

        print("初始化版本B...")
        rag_b = self._create_rag(config_b)

        # 运行测试
        results = []
        wins_a = 0
        wins_b = 0
        ties = 0

        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}/{len(test_queries)}] 测试: {query}")

            # 生成回答
            answer_a = rag_a.ask_question(query)
            answer_b = rag_b.ask_question(query)

            # 判断胜负
            judgment = self._judge(query, answer_a, answer_b)

            result = {
                'query': query,
                'answer_a': answer_a,
                'answer_b': answer_b,
                'judgment': judgment
            }
            results.append(result)

            if judgment['winner'] == 'A':
                wins_a += 1
                print(f"  🏆 版本A胜 [{judgment['a_score']} vs {judgment['b_score']}]")
            elif judgment['winner'] == 'B':
                wins_b += 1
                print(f"  🏆 版本B胜 [{judgment['b_score']} vs {judgment['a_score']}]")
            else:
                ties += 1
                print(f"  🤝 平局 [{judgment['a_score']} vs {judgment['b_score']}]")

        # 汇总
        total = len(test_queries)
        summary = {
            'config_a': config_a,
            'config_b': config_b,
            'total_queries': total,
            'wins_a': wins_a,
            'wins_b': wins_b,
            'ties': ties,
            'win_rate_a': round(wins_a / total, 3),
            'win_rate_b': round(wins_b / total, 3),
            'details': results
        }

        return summary

    def _create_rag(self, config_dict: Dict) -> RecipeRAGSystem:
        """创建RAG系统"""
        config = RAGConfig(
            moonshot_api_key=self.api_key,
            **{k: v for k, v in config_dict.items() if k != 'name'}
        )
        rag = RecipeRAGSystem(config=config)
        rag.initialize_system()
        rag.build_knowledge_base()
        return rag

    def _judge(self, question: str, answer_a: str, answer_b: str) -> Dict:
        """调用LLM判断胜负"""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.cn/v1",
            model="kimi-k2-0711-preview",
            temperature=0.1
        )

        prompt = self.JUDGE_PROMPT.format(
            question=question,
            answer_a=answer_a[:1000],
            answer_b=answer_b[:1000]
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 尝试解析JSON
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                judgment = json.loads(json_match.group())
                return {
                    'winner': judgment.get('winner', 'tie'),
                    'reason': judgment.get('reason', ''),
                    'a_score': judgment.get('a_score', 3),
                    'b_score': judgment.get('b_score', 3)
                }
        except Exception as e:
            print(f"判断失败: {e}")

        # 默认平局
        return {'winner': 'tie', 'reason': '判断失败', 'a_score': 3, 'b_score': 3}

    def print_summary(self, summary: Dict):
        """打印摘要"""
        print(f"\n{'='*60}")
        print("📊 A/B测试结果")
        print(f"{'='*60}")
        print(f"总测试数: {summary['total_queries']}")
        print(f"版本A胜: {summary['wins_a']} ({summary['win_rate_a']:.1%})")
        print(f"版本B胜: {summary['wins_b']} ({summary['win_rate_b']:.1%})")
        print(f"平局: {summary['ties']}")

        if summary['win_rate_a'] > summary['win_rate_b']:
            print(f"\n🏆 结论: 版本A更优")
        elif summary['win_rate_b'] > summary['win_rate_a']:
            print(f"\n🏆 结论: 版本B更优")
        else:
            print(f"\n🤝 结论: 两者相当")


def run_ab_test():
    """运行A/B测试示例"""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY")

    # 定义要对比的配置
    config_a = {
        'name': '当前配置',
        'top_k': 3,
        'temperature': 0.1
    }

    config_b = {
        'name': '扩大召回',
        'top_k': 5,
        'temperature': 0.1
    }

    # 测试问题
    test_queries = [
        "宫保鸡丁怎么做？",
        "推荐几个简单的素菜",
        "水煮鱼需要哪些食材？",
        "怎么做蛋糕？",
        "适合健身吃的家常菜",
    ]

    # 运行测试
    framework = ABTestFramework(api_key)
    results = framework.run_test(config_a, config_b, test_queries)
    framework.print_summary(results)

    # 保存
    output_path = Path(__file__).parent / "ab_test_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {output_path}")

    return results


if __name__ == "__main__":
    run_ab_test()
