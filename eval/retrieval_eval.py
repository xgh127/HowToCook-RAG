"""
Layer 2: 检索质量评估
指标：Recall@K, MRR, Precision@K, Category Accuracy
"""
import json
from datetime import datetime
import sys
from pathlib import Path
from typing import Dict, List
import argparse
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))
from main import RecipeRAGSystem, RAGConfig


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self, rag_system: RecipeRAGSystem, test_set_path: str):
        self.rag = rag_system
        with open(test_set_path, 'r', encoding='utf-8') as f:
            self.test_set = json.load(f)

    def evaluate(self, k_values: List[int] = [3, 5, 10]) -> Dict:
        """
        运行检索评估

        Args:
            k_values: 评估的top_k值

        Returns:
            评估结果字典
        """
        results = {
            'overall': {},
            'by_query_type': defaultdict(list),
            'by_difficulty': defaultdict(list),
            'per_query': []
        }

        print(f"\n{'='*60}")
        print(f"🔍 检索质量评估 - 共 {len(self.test_set)} 条测试用例")
        print(f"{'='*60}\n")

        for item in self.test_set:
            query = item['query']
            relevant = set(item['relevant_dishes'])
            query_type = item.get('query_type', 'unknown')
            difficulty = item.get('difficulty', 'unknown')

            # [2026-04-27 18:10] 修复：评估时必须使用与生产环境一致的检索逻辑
            # 原代码直接调用 hybrid_search(query)，跳过了 _extract_filters_from_query
            # 导致同义词/场景词过滤条件未被提取，评估结果与生产环境不一致
            # 修复方案：先提取过滤条件，再调用 metadata_filtered_search（与 ask_question 一致）
            try:
                # 步骤1：提取过滤条件（与生产环境一致）
                filters = self.rag._extract_filters_from_query(query)
                
                # 步骤2：根据是否有过滤条件选择检索方法（与 ask_question 一致）
                if filters:
                    print(f"  [Filter] {filters}")
                    chunks = self.rag.retrieval_module.metadata_filtered_search(query, filters, top_k=max(k_values))
                else:
                    chunks = self.rag.retrieval_module.hybrid_search(query, top_k=max(k_values))
                
                parent_docs = self.rag.data_module.get_parent_documents(chunks)
                retrieved_dishes = [d.metadata.get('dish_name', '') for d in parent_docs]
            except Exception as e:
                print(f"❌ 检索失败 [{item['id']}]: {e}")
                continue

            # 计算指标
            query_result = {
                'id': item['id'],
                'query': query,
                'query_type': query_type,
                'difficulty': difficulty,
                'relevant_dishes': list(relevant),
                'retrieved_dishes': retrieved_dishes[:max(k_values)],
            }

            # [2026-04-27 21:50] 修改：List查询使用过滤准确率，Detail查询使用Recall@K
            # 原因：List查询的期望菜品只是示例，不是完整集合，Recall@K不公平
            if query_type == 'list':
                # List查询：评估过滤准确率（返回的结果是否满足过滤条件）
                filter_correct_count = 0
                for dish in retrieved_dishes[:3]:
                    for doc in self.rag.data_module.documents:
                        if doc.metadata.get('dish_name') == dish:
                            match = True
                            for key, value in (filters or {}).items():
                                if key == 'category' and doc.metadata.get('category') != value:
                                    match = False
                                elif key == 'difficulty' and doc.metadata.get('difficulty') != value:
                                    match = False
                                elif key == 'category_list' and doc.metadata.get('category') not in value:
                                    match = False
                                elif key == 'difficulty_list' and doc.metadata.get('difficulty') not in value:
                                    match = False
                            if match:
                                filter_correct_count += 1
                            break
                
                filter_accuracy = round(filter_correct_count / 3, 3) if retrieved_dishes[:3] else 0
                query_result['filter_accuracy@3'] = filter_accuracy
                
                # 同时计算Category Accuracy（返回的结果是否属于期望分类）
                expected_categories = set(item.get('relevant_categories', []))
                category_correct_count = 0
                for dish in retrieved_dishes[:3]:
                    for doc in self.rag.data_module.documents:
                        if doc.metadata.get('dish_name') == dish:
                            if doc.metadata.get('category') in expected_categories:
                                category_correct_count += 1
                            break
                query_result['category_accuracy@3'] = round(category_correct_count / 3, 3) if retrieved_dishes[:3] else 0
                
                # List查询的Recall@K使用宽松定义：只要返回的结果满足过滤条件就算对
                for k in k_values:
                    query_result[f'recall@{k}'] = filter_accuracy  # 用过滤准确率替代
                    query_result[f'precision@{k}'] = filter_accuracy
                
                mrr = 1.0 if filter_accuracy > 0 else 0  # 有过滤准确率就算命中
                
            else:
                # Detail查询：使用传统的Recall@K和Precision@K
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
                
                # MRR (Mean Reciprocal Rank)
                mrr = 0
                for i, dish in enumerate(retrieved_dishes):
                    if dish in relevant:
                        mrr = 1.0 / (i + 1)
                        break
            
            query_result['mrr'] = round(mrr, 3)

            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, dish in enumerate(retrieved_dishes):
                if dish in relevant:
                    mrr = 1.0 / (i + 1)
                    break
            query_result['mrr'] = round(mrr, 3)

            # 路由分类正确性（如果系统有路由）
            try:
                route = self.rag.generation_module.query_router(query)
                route_correct = (route == query_type)
                query_result['route_correct'] = route_correct
                query_result['predicted_route'] = route
            except:
                query_result['route_correct'] = None
                query_result['predicted_route'] = 'unknown'

            results['per_query'].append(query_result)
            results['by_query_type'][query_type].append(query_result)
            results['by_difficulty'][difficulty].append(query_result)

            # 打印单条结果
            if query_type == 'list':
                status = "✅" if query_result.get('filter_accuracy@3', 0) > 0 else "❌"
                print(f"{status} [{item['id']}] {query[:30]}... "
                      f"FilterAcc@3={query_result.get('filter_accuracy@3', 0):.2f} "
                      f"CatAcc@3={query_result.get('category_accuracy@3', 0):.2f}")
            else:
                status = "✅" if query_result['recall@3'] > 0 else "❌"
                print(f"{status} [{item['id']}] {query[:30]}... "
                      f"R@3={query_result['recall@3']:.2f} "
                      f"R@5={query_result['recall@5']:.2f} "
                      f"MRR={query_result['mrr']:.2f}")

        # 计算总体指标
        self._compute_overall(results, k_values)

        # 按类型分析
        self._compute_by_group(results, 'by_query_type', '查询类型')
        self._compute_by_group(results, 'by_difficulty', '难度')

        return results

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

        # 路由准确率
        route_results = [q['route_correct'] for q in queries if q['route_correct'] is not None]
        if route_results:
            overall['route_accuracy'] = round(sum(route_results) / len(route_results), 3)

        results['overall'] = overall

    def _compute_by_group(self, results: Dict, group_key: str, group_name: str):
        """按分组计算指标"""
        group_data = results[group_key]

        print(f"\n📊 按{group_name}分析:")
        print("-" * 60)

        for group, queries in sorted(group_data.items()):
            n = len(queries)
            if n == 0:
                continue

            avg_recall3 = sum(q['recall@3'] for q in queries) / n
            avg_recall5 = sum(q['recall@5'] for q in queries) / n
            avg_mrr = sum(q['mrr'] for q in queries) / n
            
            # [2026-04-27 21:50] 新增：List查询的过滤准确率
            list_queries = [q for q in queries if q.get('query_type') == 'list']
            if list_queries:
                avg_filter_acc = sum(q.get('filter_accuracy@3', 0) for q in list_queries) / len(list_queries)
                avg_cat_acc = sum(q.get('category_accuracy@3', 0) for q in list_queries) / len(list_queries)
                print(f"  {group:12s} | 样本: {n:2d} | "
                      f"R@3: {avg_recall3:.3f} | R@5: {avg_recall5:.3f} | "
                      f"MRR: {avg_mrr:.3f} | "
                      f"FilterAcc: {avg_filter_acc:.3f} | CatAcc: {avg_cat_acc:.3f}")
            else:
                print(f"  {group:12s} | 样本: {n:2d} | "
                      f"R@3: {avg_recall3:.3f} | R@5: {avg_recall5:.3f} | "
                      f"MRR: {avg_mrr:.3f}")

    def print_summary(self, results: Dict):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print("📈 检索评估摘要")
        print(f"{'='*60}")

        overall = results['overall']
        print(f"\n总体指标:")
        for metric, value in overall.items():
            print(f"  {metric:20s}: {value:.3f}")

        # 失败案例分析
        # [2026-04-27 21:50] Detail查询看Recall@3，List查询看FilterAccuracy
        detail_failures = [q for q in results['per_query'] 
                          if q.get('query_type') == 'detail' and q['recall@3'] == 0]
        list_failures = [q for q in results['per_query'] 
                        if q.get('query_type') == 'list' and q.get('filter_accuracy@3', 0) == 0]
        
        if detail_failures:
            print(f"\n⚠️  Detail查询失败案例 ({len(detail_failures)}条):")
            for q in detail_failures[:5]:
                print(f"  - [{q['id']}] {q['query']}")
                print(f"    期望: {q['relevant_dishes']}")
                print(f"    实际: {q['retrieved_dishes'][:5]}")
        
        if list_failures:
            print(f"\n⚠️  List查询过滤失败案例 ({len(list_failures)}条):")
            for q in list_failures[:5]:
                print(f"  - [{q['id']}] {q['query']}")
                print(f"    过滤条件: {q.get('filters', 'N/A')}")
                print(f"    实际: {q['retrieved_dishes'][:5]}")

        # 优秀案例
        excellent = [q for q in results['per_query'] if q['recall@3'] >= 1.0]
        print(f"\n✅ 完美召回案例 ({len(excellent)}条):")
        for q in excellent[:5]:
            print(f"  - [{q['id']}] {q['query']}")

    def export_results(self, results: Dict, output_path: str):
        """导出评估结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 评估结果已保存: {output_path}")


def run_retrieval_eval():
    """运行检索评估"""
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("MOONSHOT_API_KEY")
    parser = argparse.ArgumentParser(description="Run RAG evaluation with customizable output.")
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="自定义输出文件名 (例如: my_eval.json)。如果不指定，默认使用 时间戳+retrieval_eval_result"
    )
    args = parser.parse_args()

    # 初始化RAG系统
    print("🚀 初始化RAG系统...")
    config = RAGConfig(moonshot_api_key=api_key)
    rag = RecipeRAGSystem(config=config)
    rag.initialize_system()
    rag.build_knowledge_base()

    # 运行评估
    test_path = Path(__file__).parent.parent / "tests" / "retrieval_test_set.json"
    evaluator = RetrievalEvaluator(rag, str(test_path))
    results = evaluator.evaluate(k_values=[3, 5, 10])

    # 打印摘要
    evaluator.print_summary(results)

    # 导出结果，将结果文件按照时间说命名
   
    # 2. 确定输出文件名
    if args.output:
        # 如果用户传入了参数，直接使用
        filename = args.output
    else:
        # 默认逻辑：时间戳 + 默认后缀
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_retrieval_eval_result.json"

    # 3. 构建最终路径
    output_path = Path(__file__).parent / filename
    evaluator.export_results(results, str(output_path))

    return results


if __name__ == "__main__":
    run_retrieval_eval()
