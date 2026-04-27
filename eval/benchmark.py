"""
检索优化效果对比评估

对比：
1. baseline（无过滤优化）：跳过过滤条件提取，直接hybrid_search
2. optimized（有过滤优化）：提取过滤条件，使用metadata_filtered_search

输出：对比指标，展示优化效果
"""
import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from main import RecipeRAGSystem, RAGConfig


class BenchmarkEvaluator:
    """对比评估器：baseline vs optimized"""

    def __init__(self, rag_system: RecipeRAGSystem, test_set_path: str):
        self.rag = rag_system
        with open(test_set_path, 'r', encoding='utf-8') as f:
            self.test_set = json.load(f)

    def evaluate_baseline(self, k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Baseline评估：跳过过滤条件提取，直接hybrid_search（模拟优化前的逻辑）
        """
        print("\n" + "="*60)
        print("📊 BASELINE评估（无过滤优化）")
        print("="*60)

        results = {
            'overall': {},
            'by_query_type': {},
            'per_query': []
        }

        for item in self.test_set:
            query = item['query']
            relevant = set(item['relevant_dishes'])
            query_type = item.get('query_type', 'unknown')

            # Baseline：直接hybrid_search，不提取过滤条件
            try:
                chunks = self.rag.retrieval_module.hybrid_search(query, top_k=max(k_values))
                parent_docs = self.rag.data_module.get_parent_documents(chunks)
                retrieved_dishes = [d.metadata.get('dish_name', '') for d in parent_docs]
            except Exception as e:
                print(f"❌ 检索失败 [{item['id']}]: {e}")
                continue

            # Baseline也提取filters用于List查询的filter_accuracy计算
            # 但不使用filters进行检索（模拟优化前的逻辑）
            baseline_filters = self.rag._extract_filters_from_query(query)
            query_result = self._compute_metrics(
                item, retrieved_dishes, relevant, k_values, filters=baseline_filters
            )
            results['per_query'].append(query_result)

            if query_type not in results['by_query_type']:
                results['by_query_type'][query_type] = []
            results['by_query_type'][query_type].append(query_result)

        self._compute_overall(results, k_values)
        return results

    def evaluate_optimized(self, k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        Optimized评估：提取过滤条件，使用metadata_filtered_search（当前优化后的逻辑）
        """
        print("\n" + "="*60)
        print("📊 OPTIMIZED评估（有过滤优化）")
        print("="*60)

        results = {
            'overall': {},
            'by_query_type': {},
            'per_query': []
        }

        for item in self.test_set:
            query = item['query']
            relevant = set(item['relevant_dishes'])
            query_type = item.get('query_type', 'unknown')

            # Optimized：提取过滤条件，使用metadata_filtered_search
            try:
                filters = self.rag._extract_filters_from_query(query)

                # [2026-04-28 00:57] 修复：Detail查询不应用metadata过滤
                if query_type == 'detail':
                    chunks = self.rag.retrieval_module.hybrid_search(query, top_k=max(k_values))
                elif filters:
                    chunks = self.rag.retrieval_module.metadata_filtered_search(
                        query, filters, top_k=max(k_values)
                    )
                else:
                    chunks = self.rag.retrieval_module.hybrid_search(query, top_k=max(k_values))

                parent_docs = self.rag.data_module.get_parent_documents(chunks)
                retrieved_dishes = [d.metadata.get('dish_name', '') for d in parent_docs]
            except Exception as e:
                print(f"❌ 检索失败 [{item['id']}]: {e}")
                continue

            query_result = self._compute_metrics(
                item, retrieved_dishes, relevant, k_values, filters=filters
            )
            results['per_query'].append(query_result)

            if query_type not in results['by_query_type']:
                results['by_query_type'][query_type] = []
            results['by_query_type'][query_type].append(query_result)

        self._compute_overall(results, k_values)
        return results

    def _compute_metrics(self, item, retrieved_dishes, relevant, k_values, filters=None):
        """计算单条查询的指标"""
        query_type = item.get('query_type', 'unknown')

        query_result = {
            'id': item['id'],
            'query': item['query'],
            'query_type': query_type,
            'relevant_dishes': list(relevant),
            'retrieved_dishes': retrieved_dishes[:max(k_values)],
        }

        if query_type == 'list':
            # List查询：过滤准确率
            # 归一化filters：将category_list/difficulty_list转换为标准的category/difficulty（列表值）
            normalized_filters = {}
            if filters:
                for key, value in filters.items():
                    if key == 'category_list':
                        normalized_filters['category'] = value if isinstance(value, list) else [value]
                    elif key == 'difficulty_list':
                        normalized_filters['difficulty'] = value if isinstance(value, list) else [value]
                    elif key not in ['cuisine']:
                        normalized_filters[key] = value

            filter_correct = 0
            if normalized_filters:
                for dish in retrieved_dishes[:3]:
                    for doc in self.rag.data_module.documents:
                        if doc.metadata.get('dish_name') == dish:
                            match = True
                            for key, value in normalized_filters.items():
                                if key not in doc.metadata:
                                    match = False
                                    break
                                # 支持列表OR匹配和单值精确匹配
                                if isinstance(value, list):
                                    if doc.metadata[key] not in value:
                                        match = False
                                        break
                                else:
                                    if doc.metadata[key] != value:
                                        match = False
                                        break
                            if match:
                                filter_correct += 1
                            break

            filter_accuracy = round(filter_correct / 3, 3) if retrieved_dishes[:3] else 0
            query_result['filter_accuracy@3'] = filter_accuracy

            # Category Accuracy
            expected_categories = set(item.get('relevant_categories', []))
            cat_correct = 0
            for dish in retrieved_dishes[:3]:
                for doc in self.rag.data_module.documents:
                    if doc.metadata.get('dish_name') == dish:
                        if doc.metadata.get('category') in expected_categories:
                            cat_correct += 1
                        break
            query_result['category_accuracy@3'] = round(cat_correct / 3, 3) if retrieved_dishes[:3] else 0

            for k in k_values:
                query_result[f'recall@{k}'] = filter_accuracy
                query_result[f'precision@{k}'] = filter_accuracy

            query_result['mrr'] = 1.0 if filter_accuracy > 0 else 0
        else:
            # Detail查询：Recall@K
            for k in k_values:
                retrieved_k = set(retrieved_dishes[:k])
                recall = len(relevant & retrieved_k) / len(relevant) if relevant else 0
                query_result[f'recall@{k}'] = round(recall, 3)

            for k in k_values:
                retrieved_k = retrieved_dishes[:k]
                if retrieved_k:
                    precision = len([d for d in retrieved_k if d in relevant]) / len(retrieved_k)
                else:
                    precision = 0
                query_result[f'precision@{k}'] = round(precision, 3)

            mrr = 0
            for i, dish in enumerate(retrieved_dishes):
                if dish in relevant:
                    mrr = 1.0 / (i + 1)
                    break
            query_result['mrr'] = round(mrr, 3)

        return query_result

    def _compute_overall(self, results: Dict, k_values: List[int]):
        """计算总体指标"""
        queries = results['per_query']
        n = len(queries)
        if n == 0:
            return

        overall = {}
        for k in k_values:
            overall[f'recall@{k}'] = round(
                sum(q[f'recall@{k}'] for q in queries) / n, 3
            )
            overall[f'precision@{k}'] = round(
                sum(q[f'precision@{k}'] for q in queries) / n, 3
            )
        overall['mrr'] = round(sum(q['mrr'] for q in queries) / n, 3)
        results['overall'] = overall

    def print_comparison(self, baseline: Dict, optimized: Dict):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📈 优化效果对比")
        print("="*60)

        print(f"\n{'指标':<20} {'Baseline':<12} {'Optimized':<12} {'提升':<12}")
        print("-" * 60)

        for metric in ['recall@3', 'recall@5', 'recall@10', 'precision@3', 'mrr']:
            b_val = baseline['overall'].get(metric, 0)
            o_val = optimized['overall'].get(metric, 0)
            delta = o_val - b_val
            delta_pct = (delta / b_val * 100) if b_val > 0 else 0

            print(f"{metric:<20} {b_val:<12.3f} {o_val:<12.3f} {delta:+.3f} ({delta_pct:+.1f}%)")

        # 按查询类型对比
        print("\n" + "="*60)
        print("📊 按查询类型对比")
        print("="*60)

        for qtype in ['detail', 'list']:
            print(f"\n【{qtype.upper()}查询】")
            print(f"{'指标':<20} {'Baseline':<12} {'Optimized':<12} {'提升':<12}")
            print("-" * 60)

            b_queries = baseline['by_query_type'].get(qtype, [])
            o_queries = optimized['by_query_type'].get(qtype, [])

            if not b_queries or not o_queries:
                continue

            for metric in ['recall@3', 'recall@5', 'mrr']:
                b_val = sum(q.get(metric, 0) for q in b_queries) / len(b_queries)
                o_val = sum(q.get(metric, 0) for q in o_queries) / len(o_queries)
                delta = o_val - b_val

                print(f"{metric:<20} {b_val:<12.3f} {o_val:<12.3f} {delta:+.3f}")

            # List查询额外显示过滤准确率
            if qtype == 'list':
                b_fa = sum(q.get('filter_accuracy@3', 0) for q in b_queries) / len(b_queries)
                o_fa = sum(q.get('filter_accuracy@3', 0) for q in o_queries) / len(o_queries)
                print(f"{'filter_accuracy@3':<20} {b_fa:<12.3f} {o_fa:<12.3f} {o_fa - b_fa:+.3f}")

    def export_comparison(self, baseline: Dict, optimized: Dict, output_path: str):
        """导出对比结果"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'baseline': baseline,
            'optimized': optimized,
            'improvement': {}
        }

        for metric in ['recall@3', 'recall@5', 'recall@10', 'precision@3', 'mrr']:
            b_val = baseline['overall'].get(metric, 0)
            o_val = optimized['overall'].get(metric, 0)
            comparison['improvement'][metric] = {
                'baseline': b_val,
                'optimized': o_val,
                'absolute_delta': round(o_val - b_val, 3),
                'relative_delta_pct': round((o_val - b_val) / b_val * 100, 1) if b_val > 0 else 0
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"\n💾 对比结果已保存: {output_path}")


def run_benchmark():
    """运行对比评估"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY")

    # 初始化RAG系统
    print("🚀 初始化RAG系统...")
    config = RAGConfig(moonshot_api_key=api_key)
    rag = RecipeRAGSystem(config=config)
    rag.initialize_system()
    rag.build_knowledge_base()

    # 运行对比评估
    test_path = Path(__file__).parent.parent / "tests" / "retrieval_test_set.json"
    evaluator = BenchmarkEvaluator(rag, str(test_path))

    baseline_results = evaluator.evaluate_baseline()
    optimized_results = evaluator.evaluate_optimized()

    evaluator.print_comparison(baseline_results, optimized_results)

    # 导出结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).parent.parent / "output" / f"{timestamp}_benchmark_result.json"
    evaluator.export_comparison(baseline_results, optimized_results, str(output_path))

    return baseline_results, optimized_results


if __name__ == "__main__":
    run_benchmark()
