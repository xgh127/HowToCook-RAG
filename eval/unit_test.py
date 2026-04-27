"""
Layer 1: 单元测试
测试各个模块的基础功能
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import unittest
from rag_modules.data_preparation import DataPreparationModule
from rag_modules.index_construction import IndexConstructionModule
from rag_modules.retrieval_optimization import RetrievalOptimizationModule
from rag_modules.generation_integration import GenerationIntegrationModule
from langchain_core.documents import Document


class TestDataPreparation(unittest.TestCase):
    """测试数据准备模块"""

    @classmethod
    def setUpClass(cls):
        cls.module = DataPreparationModule("./data/cook")

    def test_load_documents(self):
        """测试文档加载"""
        docs = self.module.load_documents()
        self.assertGreater(len(docs), 0, "应该加载到文档")

        # 检查元数据
        for doc in docs[:5]:
            self.assertIn('dish_name', doc.metadata)
            self.assertIn('category', doc.metadata)
            self.assertIn('difficulty', doc.metadata)
            self.assertIn('parent_id', doc.metadata)

    def test_enhance_metadata(self):
        """测试元数据增强"""
        doc = Document(
            page_content="预估烹饪难度：★★★\n# 测试菜",
            metadata={"source": "data/cook/dishes/meat_dish/测试菜.md"}
        )
        self.module._enhance_metadata(doc)

        self.assertEqual(doc.metadata['category'], '荤菜')
        self.assertEqual(doc.metadata['dish_name'], '测试菜')
        self.assertEqual(doc.metadata['difficulty'], '中等')

    def test_chunk_documents(self):
        """测试文档分块"""
        self.module.load_documents()
        chunks = self.module.chunk_documents()

        self.assertGreater(len(chunks), len(self.module.documents),
                          "分块数应该大于文档数")

        # 检查父子关系
        for chunk in chunks[:10]:
            self.assertIn('parent_id', chunk.metadata)
            self.assertIn('chunk_id', chunk.metadata)

    def test_get_parent_documents(self):
        """测试父子文档映射"""
        self.module.load_documents()
        chunks = self.module.chunk_documents()

        if chunks:
            # 取前3个子块
            test_chunks = chunks[:3]
            parents = self.module.get_parent_documents(test_chunks)

            # 父文档应该去重
            parent_ids = [p.metadata['parent_id'] for p in parents]
            self.assertEqual(len(parent_ids), len(set(parent_ids)),
                           "父文档应该去重")

    def test_category_mapping(self):
        """测试分类映射完整性"""
        categories = DataPreparationModule.get_supported_categories()
        self.assertIn('荤菜', categories)
        self.assertIn('素菜', categories)
        self.assertIn('汤品', categories)


class TestQueryRouter(unittest.TestCase):
    """测试查询路由"""

    @classmethod
    def setUpClass(cls):
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("MOONSHOT_API_KEY")

        cls.module = GenerationIntegrationModule(
            model_name="kimi-k2-0711-preview",
            api_key=api_key
        )

    def test_list_queries(self):
        """测试列表类查询识别"""
        list_queries = [
            "推荐几个素菜",
            "有什么川菜",
            "给我3个简单的菜",
        ]
        for q in list_queries:
            route = self.module.query_router(q)
            self.assertEqual(route, 'list',
                           f"'{q}' 应该被识别为list类型")

    def test_detail_queries(self):
        """测试详细类查询识别"""
        detail_queries = [
            "宫保鸡丁怎么做",
            "红烧肉需要什么食材",
            "制作步骤",
        ]
        for q in detail_queries:
            route = self.module.query_router(q)
            self.assertEqual(route, 'detail',
                           f"'{q}' 应该被识别为detail类型")


class TestRetrieval(unittest.TestCase):
    """测试检索模块"""

    def test_rrf_rerank(self):
        """测试RRF重排"""
        # 创建模拟文档
        docs = [
            Document(page_content=f"doc_{i}", metadata={})
            for i in range(5)
        ]

        # 模拟向量检索结果（顺序：0,1,2,3,4）
        vector_docs = docs[:5]
        # 模拟BM25结果（顺序：4,3,2,1,0）
        bm25_docs = list(reversed(docs[:5]))

        # 创建检索模块实例进行测试
        # 这里简化测试，直接验证RRF逻辑
        doc_scores = {}
        k = 60

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # 所有文档都应该有分数
        self.assertEqual(len(doc_scores), 5)

        # doc_0在向量排第1，BM25排第5，总分应该较高
        # doc_4在向量排第5，BM25排第1，总分也应该较高
        score_0 = doc_scores[hash("doc_0")]
        score_4 = doc_scores[hash("doc_4")]
        score_2 = doc_scores[hash("doc_2")]

        # doc_0和doc_4应该比doc_2分数高（因为各有一个第1名）
        self.assertGreater(score_0, score_2)
        self.assertGreater(score_4, score_2)


class TestIndexConstruction(unittest.TestCase):
    """测试索引构建模块"""

    def test_embedding_initialization(self):
        """测试嵌入模型初始化"""
        module = IndexConstructionModule()
        self.assertIsNotNone(module.embeddings)

    def test_save_load_index(self):
        """测试索引保存和加载"""
        import tempfile
        import shutil

        # 创建临时目录
        temp_dir = tempfile.mkdtemp()

        try:
            # 创建索引
            module = IndexConstructionModule(index_save_path=temp_dir)

            # 用模拟数据构建索引
            docs = [
                Document(page_content="测试文档1", metadata={"test": 1}),
                Document(page_content="测试文档2", metadata={"test": 2}),
            ]
            module.build_vector_index(docs)
            module.save_index()

            # 加载索引
            loaded = module.load_index()
            self.assertIsNotNone(loaded)

            # 验证可以搜索
            results = loaded.similarity_search("测试", k=1)
            self.assertEqual(len(results), 1)

        finally:
            # 清理
            shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有单元测试"""
    print("="*60)
    print("🧪 运行单元测试")
    print("="*60)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryRouter))
    suite.addTests(loader.loadTestsFromTestCase(TestRetrieval))
    suite.addTests(loader.loadTestsFromTestCase(TestIndexConstruction))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 打印摘要
    print(f"\n{'='*60}")
    print("📊 测试结果摘要")
    print(f"{'='*60}")
    print(f"运行测试: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 存在失败的测试")

    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
