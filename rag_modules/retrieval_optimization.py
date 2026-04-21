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
    """检索优化模块 - 负责混合检索和过滤（检索阶段直接过滤，优化结果数量）"""

    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore  # FAISS向量存储对象
        self.chunks = chunks  # 文档块列表，用于BM25检索
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置基础检索器（无过滤，过滤逻辑动态传入）"""
        logger.info("正在设置基础检索器...")

        # 基础向量检索器（过滤条件动态传入，不写死）
        self.base_vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # 中文预处理函数
        def chinese_preprocess(text: str) -> list:
            return list(jieba.cut(text))

        # 基础BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=self.chunks,
            k=5,
            preprocess_func=chinese_preprocess
        )

        logger.info("基础检索器设置完成")

    def _filter_documents(self, docs: List[Document], filters: Dict[str, Any]) -> List[Document]:
        """
        元数据过滤工具方法（复用逻辑，供BM25使用）
        """
        if not filters:
            return docs

        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                # 支持列表/单值匹配
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

    def _filtered_vector_search(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        带元数据过滤的向量检索（FAISS原生支持，检索阶段直接过滤）
        """
        search_kwargs = {"k": k}
        # 动态添加过滤条件
        if filters:
            search_kwargs["filter"] = filters

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        return retriever.invoke(query)

    def _filtered_bm25_search(self, query: str, filters: Dict[str, Any] = None, k: int = 5) -> List[Document]:
        """
        带元数据过滤的BM25检索（检索后立即过滤，等价于检索阶段过滤）
        """
        # 先检索
        docs = self.bm25_retriever.invoke(query)
        # 立即过滤（前置过滤，不进入后续流程）
        return self._filter_documents(docs, filters)

    def hybrid_search(self, query: str, filters: Dict[str, Any] = None, top_k: int = 3) -> List[Document]:
        """
        【优化】混合检索 - 检索阶段直接应用元数据过滤，再RRF重排
        Args:
            query: 查询文本
            filters: 元数据过滤条件（可选）
            top_k: 返回结果数量
        Returns:
            检索到的文档列表
        """
        # 关键修改：检索时直接过滤，而非事后过滤
        vector_docs = self._filtered_vector_search(query, filters, k=5)
        bm25_docs = self._filtered_bm25_search(query, filters, k=5)

        # RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """
        【优化】带元数据过滤的检索：直接调用带过滤的混合检索
        废弃原逻辑：先检索全量→再过滤
        """
        return self.hybrid_search(query, filters=filters, top_k=top_k)

    def test_retrievers(self, query: str):
        """测试检索器，输出检索结果"""
        logger.info(f"测试检索器 - 查询: {query}")
        vector_results = self.base_vector_retriever.invoke(query)
        bm25_results = self.bm25_retriever.invoke(query)
        print("\n向量检索结果:", len(vector_results))
        print("\nBM25检索结果:", len(bm25_results))

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """RRF重排（逻辑不变，仅接收已过滤的文档）"""
        doc_scores = {}
        doc_objects = {}

        # 向量检索分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # BM25检索分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            doc = doc_objects[doc_id]
            doc.metadata['rrf_score'] = final_score
            reranked_docs.append(doc)

        logger.info(f"RRF重排完成: 向量{len(vector_docs)}个, BM25{len(bm25_docs)}个, 最终{len(reranked_docs)}个")
        return reranked_docs