"""
检索优化模块
"""

import logging
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import jieba

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和元数据过滤"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置基础检索器"""
        logger.info("正在设置基础检索器...")

        self.base_vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        def chinese_preprocess(text: str) -> list:
            return list(jieba.cut(text))

        self.bm25_retriever = BM25Retriever.from_documents(
            documents=self.chunks,
            k=5,
            preprocess_func=chinese_preprocess
        )

        logger.info("基础检索器设置完成")

    # ==================== 元数据过滤工具方法 ====================

    def _filter_documents(self, docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """
        元数据过滤工具方法

        支持：
        - 单值精确匹配: {'category': '素菜'}
        - 列表OR匹配: {'category': ['素菜', '水产']} 或 {'category_list': ['素菜', '水产']}
        - 多字段AND匹配: {'category': '素菜', 'difficulty': '简单'}
        """
        if not filters:
            return docs

        # 统一处理：将 category_list/difficulty_list 转换为标准列表格式
        normalized_filters = {}
        for key, value in filters.items():
            if key in ['category_list', 'difficulty_list']:
                # 映射到对应的单字段，值为列表（OR语义）
                target_key = key.replace('_list', '')
                normalized_filters[target_key] = value if isinstance(value, list) else [value]
            elif key not in ['cuisine']:  # 跳过辅助字段
                normalized_filters[key] = value

        filtered = []
        for doc in docs:
            match = True
            for key, value in normalized_filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                # 支持列表/单值匹配（列表表示OR逻辑）
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                else:
                    if doc.metadata[key] != value:
                        match = False
                        break
            if match:
                filtered.append(doc)
        return filtered

    # ==================== 向量检索（带元数据过滤） ====================
    # 【2026-04-27 23:07】修改：使用FAISS原生filter参数 + 调大fetch_k
    # 原因：FAISS原生不支持元数据过滤，采用"后过滤"策略——先检索更多候选，再过滤

    def _filtered_vector_search(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        带元数据过滤的向量检索

        策略：
        1. 无过滤条件 → 直接使用FAISS检索
        2. 有过滤条件 → 调大fetch_k先检索更多候选，再用FAISS filter过滤

        【修改点】fetch_k = min(总文档数, max(200, k * 5))
        - 1977个chunk时，k=5则fetch_k=200，确保过滤后有足够候选
        """
        if not filters:
            search_kwargs = {"k": k}
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            return retriever.invoke(query)

        # 【修改点】调大fetch_k，确保过滤后有足够候选
        total_docs = len(self.chunks)
        fetch_k = min(total_docs, max(200, k * 5))

        # 将category_list/difficulty_list转换为FAISS可理解的filter格式
        faiss_filter = {}
        for key, value in filters.items():
            if key in ['category_list', 'difficulty_list']:
                # FAISS filter支持列表OR匹配
                faiss_filter[key.replace('_list', '')] = value
            elif key not in ['cuisine']:
                faiss_filter[key] = value

        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=faiss_filter if faiss_filter else None,
            fetch_k=fetch_k
        )

        if len(results) < k:
            print(f"[警告] 过滤后结果不足: 期望{k}个，实际{len(results)}个。filters={filters}")

        return results

    # ==================== BM25检索（带元数据过滤） ====================
    # 【2026-04-27 23:07】修改：候选集不大时直接用关键词匹配排序，避免重建BM25索引

    def _filtered_bm25_search(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        带元数据过滤的BM25检索

        策略：
        1. 无过滤条件 → 直接使用BM25检索
        2. 有过滤条件 → 先过滤候选集，再在候选集内检索
           - 候选集≤100：直接用关键词匹配排序（避免重建索引开销）
           - 候选集>100：重建BM25索引后检索
        """
        if not filters:
            self.bm25_retriever.k = k
            return self.bm25_retriever.invoke(query)

        # 阶段1：用过滤条件筛选候选chunks（统一处理category_list/difficulty_list）
        candidate_chunks = self._filter_documents(self.chunks, filters)

        if len(candidate_chunks) == 0:
            print(f"[警告] 没有满足过滤条件 {filters} 的文档")
            return []

        # 阶段2：在候选集内检索
        # 【修改点】候选集较小时，直接用关键词匹配排序
        if len(candidate_chunks) <= 100:
            query_terms = set(jieba.cut(query))
            scored = []
            for chunk in candidate_chunks:
                chunk_terms = set(jieba.cut(chunk.page_content))
                score = len(query_terms & chunk_terms)
                scored.append((score, chunk))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in scored[:k]]

        # 候选集较大时，重建BM25索引
        def chinese_preprocess(text: str) -> list:
            return list(jieba.cut(text))

        temp_bm25 = BM25Retriever.from_documents(
            documents=candidate_chunks,
            k=min(k, len(candidate_chunks)),
            preprocess_func=chinese_preprocess
        )

        return temp_bm25.invoke(query)

    # ==================== 混合检索（主入口） ====================
    # 【2026-04-27 18:00】重构：支持多字段OR检索（通过列表值实现）

    def hybrid_search(self, query: str, filters: Dict[str, Any] = None, top_k: int = 3) -> List[Document]:
        """
        混合检索主入口

        支持：
        - 无过滤：向量+BM25混合检索，RRF重排
        - 有过滤：先过滤候选集，再混合检索
          * 单值过滤：{'category': '素菜'}
          * 列表OR过滤：{'category': ['荤菜', '素菜', '汤品']}
          * 多字段AND：{'category': '素菜', 'difficulty': '简单'}
        """
        # 检查是否包含列表值（OR语义）
        has_list_filter = any(
            isinstance(v, list) for k, v in (filters or {}).items()
            if k not in ['cuisine']
        )

        # 如果有过滤条件且包含列表OR，使用多值过滤检索
        if filters and has_list_filter:
            return self._multi_value_search(query, filters, top_k)

        # 标准检索：向量+BM25混合，RRF重排
        vector_docs = self._filtered_vector_search(query, filters, k=5)
        bm25_docs = self._filtered_bm25_search(query, filters, k=5)
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    # ==================== 多值OR检索（统一处理多分类/多难度） ====================
    # 【2026-04-27 23:40】重构：统一_multi_category_search和_multi_difficulty_search为_multi_value_search
    # 原因：两者逻辑完全相同，都是"按条件过滤候选集→候选集不大直接返回/候选集大则分别检索"

    def _multi_value_search(self, query: str, filters: Dict[str, Any], top_k: int) -> List[Document]:
        """
        多值OR检索（统一处理多分类、多难度等任意字段的OR条件）

        策略：
        1. 用_filter_documents获取满足所有过滤条件的候选chunks（自动处理列表OR）
        2. 候选集≤200：直接返回候选集（不过度依赖语义相似度）
        3. 候选集>200：对每个列表值的组合分别检索，合并去重后RRF重排
        """
        # 获取所有列表字段（OR字段）
        list_fields = {k: v for k, v in filters.items()
                      if isinstance(v, list) and k not in ['cuisine']}
        print(f"[多值OR检索] 条件: {list_fields}")

        # 阶段1：获取候选集（_filter_documents自动处理OR逻辑）
        candidate_chunks = self._filter_documents(self.chunks, filters)
        print(f"[多值OR检索] 候选集大小: {len(candidate_chunks)}个chunks")

        # 【修改点】候选集不大时直接返回
        if len(candidate_chunks) <= 200:
            print(f"[多值OR检索] 候选集较小({len(candidate_chunks)}个)，直接返回候选集")
            return self._deduplicate_and_limit(candidate_chunks, top_k)

        # 候选集较大时，对每个列表值分别做精确过滤检索
        all_vector_docs = []
        all_bm25_docs = []

        # 生成所有组合（笛卡尔积）
        from itertools import product
        list_keys = list(list_fields.keys())
        list_values = list(list_fields.values())

        for combo in product(*list_values):
            # 构建单值过滤条件
            combo_filters = {k: v for k, v in filters.items() if not isinstance(v, list)}
            for key, value in zip(list_keys, combo):
                combo_filters[key] = value

            vector_docs = self._filtered_vector_search(query, combo_filters, k=5)
            bm25_docs = self._filtered_bm25_search(query, combo_filters, k=5)

            all_vector_docs.extend(vector_docs)
            all_bm25_docs.extend(bm25_docs)

        # 去重后RRF重排
        unique_vector_docs = self._deduplicate_docs(all_vector_docs)
        unique_bm25_docs = self._deduplicate_docs(all_bm25_docs)

        print(f"[多值OR检索] 向量检索: {len(unique_vector_docs)}个, BM25: {len(unique_bm25_docs)}个")

        reranked_docs = self._rrf_rerank(unique_vector_docs, unique_bm25_docs)
        return reranked_docs[:top_k]

    # ==================== 工具方法 ====================

    def _deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        """按内容去重"""
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        return unique_docs

    def _deduplicate_and_limit(self, chunks: List[Document], top_k: int) -> List[Document]:
        """按父文档去重并限制数量"""
        seen_parents = set()
        unique_docs = []
        for chunk in chunks:
            parent = chunk.metadata.get('parent_id', '')
            if parent and parent not in seen_parents:
                seen_parents.add(parent)
                unique_docs.append(chunk)
            elif not parent:
                unique_docs.append(chunk)
            if len(unique_docs) >= top_k:
                break
        return unique_docs

    # ==================== 公共接口 ====================

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """带元数据过滤的检索（直接调用混合检索）"""
        return self.hybrid_search(query, filters=filters, top_k=top_k)

    def test_retrievers(self, query: str):
        """测试检索器"""
        logger.info(f"测试检索器 - 查询: {query}")
        vector_results = self.base_vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        print("\n向量检索结果:", len(vector_results))
        print("\nBM25检索结果:", len(bm25_results))

    # ==================== RRF重排 ====================

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """RRF重排"""
        doc_scores = {}
        doc_objects = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata['rrf_score'] = final_score
            reranked_docs.append(doc)

        logger.info(f"RRF重排完成: 向量{len(vector_docs)}个, BM25{len(bm25_docs)}个, 最终{len(reranked_docs)}个")
        return reranked_docs
