"""
数据准备模块
"""

import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    # 统一维护的分类与难度配置，供外部复用，避免关键词重复定义
    CATEGORY_MAPPING = {
        'meat_dish': '荤菜',
        'vegetable_dish': '素菜',
        'soup': '汤品',
        'dessert': '甜品',
        'breakfast': '早餐',
        'staple': '主食',
        'aquatic': '水产',
        'condiment': '调料',
        'drink': '饮品'
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ['非常简单', '简单', '中等', '困难', '非常困难']
    
    # [2026-04-27 17:50] 新增：分类同义词表，用于语义过滤匹配
    # 解决用户查询中的同义词/变体词无法匹配到标准分类的问题
    # 例如：用户说"素食"→匹配"素菜"，"海鲜"→匹配"水产"
    CATEGORY_SYNONYMS = {
        '素菜': ['素菜', '素食', '青菜', '蔬菜', '清淡', '斋菜', '素'],
        '荤菜': ['荤菜', '肉菜', '炒菜', '大肉', '主菜', '荤'],
        '汤品': ['汤品', '汤', '羹', '炖品', '煲汤', '炖汤'],
        '甜品': ['甜品', '甜点', '蛋糕', '布丁', '糖水', '烘焙', '糕点'],
        '饮品': ['饮品', '饮料', '奶茶', '果汁', '酒水', '茶', '喝的'],
        '水产': ['水产', '海鲜', '鱼', '虾', '蟹', '贝类', '海味'],
        '早餐': ['早餐', '早点', '早饭', '早茶', '早点'],
        '主食': ['主食', '米饭', '面食', '粉', '面条', '粥', '主食类'],
        '调料': ['调料', '酱料', '蘸料', '调味', '调味品'],
    }
    
    # [2026-04-27 17:50] 新增：场景词到过滤条件的映射
    # 解决"新手""快手""健身"等场景词无法映射到具体过滤条件的问题
    # 注意：difficulty用列表表示OR逻辑（满足任一即可）
    SCENE_TO_FILTER = {
        # --- 难度映射 ---
        '新手': {'difficulty': ['非常简单', '简单']},
        '快手': {'difficulty': ['非常简单', '简单']},
        '简单': {'difficulty': ['非常简单', '简单']},
        '容易': {'difficulty': ['非常简单', '简单']},
        '入门': {'difficulty': ['非常简单', '简单']},
        '零基础': {'difficulty': ['非常简单', '简单']},
        '复杂': {'difficulty': ['困难', '非常困难']},
        '硬菜': {'difficulty': ['困难', '非常困难']},
        '挑战': {'difficulty': ['困难', '非常困难']},
        '大菜': {'difficulty': ['困难', '非常困难']},
        '困难': {'difficulty': ['困难', '非常困难']},
        
        # --- 场景映射（分类+难度组合） ---
        '健身': {'category': ['素菜', '水产'], 'difficulty': ['非常简单', '简单', '中等']},
        '减肥': {'category': ['素菜', '汤品'], 'difficulty': ['非常简单', '简单']},
        '减脂': {'category': ['素菜', '水产'], 'difficulty': ['非常简单', '简单']},
        '低脂': {'category': ['素菜', '水产', '汤品'], 'difficulty': ['非常简单', '简单', '中等']},
        '下酒': {'category': ['荤菜', '素菜'], 'difficulty': ['简单', '中等']},
        '宴客': {'difficulty': ['中等', '困难']},
        '待客': {'difficulty': ['中等', '困难']},
        '聚餐': {'difficulty': ['中等', '困难']},
        '请客': {'difficulty': ['中等', '困难']},
        '便当': {'category': ['荤菜', '素菜'], 'difficulty': ['简单']},
        '带饭': {'category': ['荤菜', '素菜'], 'difficulty': ['简单']},
        '工作日': {'difficulty': ['非常简单', '简单']},
        '懒人': {'difficulty': ['非常简单', '简单']},
    }
    
    # [2026-04-27 17:50] 新增：菜系到分类的映射（特殊处理）
    # 菜系不直接等于分类，而是跨多个分类，需要OR检索
    # 例如：川菜包含荤菜、素菜、汤品等多个分类
    CUISINE_TO_CATEGORY = {
        '川菜': ['荤菜', '素菜', '汤品'],
        '湘菜': ['荤菜', '素菜', '汤品'],
        '粤菜': ['荤菜', '素菜', '汤品', '水产'],
        '鲁菜': ['荤菜', '素菜', '汤品'],
        '江浙菜': ['荤菜', '素菜', '水产', '汤品'],
        '东北菜': ['荤菜', '素菜', '汤品'],
        '家常菜': ['荤菜', '素菜', '汤品'],
    }
    
    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        
        Args:
            data_path: 数据文件夹路径
        """
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_path} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        data_path_obj = Path(self.data_path)# data/cook

        for md_file in data_path_obj.rglob("*.md"): # md_file示例：data/cook/dishes/aquatic/咖喱炒蟹.md
            try:
                # 直接读取文件内容，保持Markdown格式
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 为每个父文档分配确定性的唯一ID（基于数据根目录的相对路径）
                try:
                    data_root = Path(self.data_path).resolve()
                    relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                except Exception:
                    relative_path = Path(md_file).as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),# md_file示例：data/cook/dishes/aquatic/咖喱炒蟹.md
                        "parent_id": parent_id,
                        "doc_type": "parent"  # 标记为父文档
                    }
                )
                documents.append(doc)

            except Exception as e:
                logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        # 增强文档元数据
        for doc in documents:
            self._enhance_metadata(doc)
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def _enhance_metadata(self, doc: Document):
        """
        增强文档元数据
        
        Args:
            doc: 需要增强元数据的文档
        """
        file_path = Path(doc.metadata.get('source', '')) # 示例：data/cook/dishes/aquatic/咖喱炒蟹.md
        path_parts = file_path.parts # 示例：('data', 'cook', 'dishes', 'aquatic', '咖喱炒蟹.md')
        
        # 提取菜品分类
        doc.metadata['category'] = '其他' # 默认分类
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata['category'] = value
                break
        
        # 提取菜品名称,
        doc.metadata['dish_name'] = file_path.stem #示例：咖喱炒蟹，这里的stem会自动去掉.md后缀以及路径部分

        # 分析难度等级
        content = doc.page_content
        if '★★★★★' in content:
            doc.metadata['difficulty'] = '非常困难'
        elif '★★★★' in content:
            doc.metadata['difficulty'] = '困难'
        elif '★★★' in content:
            doc.metadata['difficulty'] = '中等'
        elif '★★' in content:
            doc.metadata['difficulty'] = '简单'
        elif '★' in content:
            doc.metadata['difficulty'] = '非常简单'
        else:
            doc.metadata['difficulty'] = '未知'

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        """对外提供支持的难度标签列表"""
        return cls.DIFFICULTY_LABELS
    
    # [2026-04-27 17:50] 新增：对外提供同义词表，供main.py使用
    @classmethod
    def get_category_synonyms(cls) -> Dict[str, List[str]]:
        """对外提供分类同义词表"""
        return cls.CATEGORY_SYNONYMS
    
    # [2026-04-27 17:50] 新增：对外提供场景映射表，供main.py使用
    @classmethod
    def get_scene_to_filter(cls) -> Dict[str, Dict[str, Any]]:
        """对外提供场景词到过滤条件的映射表"""
        return cls.SCENE_TO_FILTER
    
    # [2026-04-27 17:50] 新增：对外提供菜系映射表，供main.py使用
    @classmethod
    def get_cuisine_to_category(cls) -> Dict[str, List[str]]:
        """对外提供菜系到分类的映射表"""
        return cls.CUISINE_TO_CATEGORY
    
    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 菜品名称
            ("##", "二级标题"),   # 必备原料、计算、操作等
            ("###", "三级标题")   # 简易版本、复杂版本等
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个文档进行Markdown分割，示例：[Document(metadata={'主标题': '咖喱炒蟹的做法'}, page_content='# 咖喱炒蟹的做法  \n第一次吃咖喱炒蟹是在泰国的建兴酒家中餐厅，爆肉的螃蟹挂满有蟹黄味道的咖喱，味道真的绝，喜欢吃海鲜的程序员绝对不能错过。操作简单，对沿海的程序员非常友好。  \n预估烹饪难度：★★★★'), Document(metadata={'主标题': '咖喱炒蟹的做法', '二级标题': '必备原料和工具'}, page_content='## 必备原料和工具  \n- 青蟹（别称：肉蟹）\n- 咖喱块（推介乐惠蟹黄咖喱）\n- 洋葱\n- 椰浆\n- 鸡蛋\n- 生粉（别称：淀粉）\n- 大蒜'), Document(metadata={'主标题': '咖喱炒蟹的做法', '二级标题': '计算'}, page_content='## 计算  \n每次制作前需要确定计划做几份。一份正好够 1 个人食用  \n总量：  \n- 肉蟹 1 只（大约 300g） * 份数\n- 咖喱块 15g（一小块）*份数\n- 椰浆 100ml*份数\n- 鸡蛋 1 个 *份数\n- 洋葱 200g *份数\n- 大蒜 5 瓣 *份数'), Document(metadata={'主标题': '咖喱炒蟹的做法', '二级标题': '操作'}, page_content='## 操作  \n- 肉蟹掀盖后对半砍开，蟹钳用刀背轻轻拍裂，切口和蟹钳蘸一下生粉，不要太多。撒 5g 生粉到蟹盖中，盖住蟹黄，备用\n- 洋葱切成洋葱碎，备用\n- 大蒜切碎，备用\n- 烧一壶开水，备用\n- 起锅烧油，倒入约 20ml 食用油，等待 10 秒让油温升高\n- 将螃蟹切口朝下，轻轻放入锅中，煎 20 秒，这一步主要是封住蟹黄，蟹肉。然后翻面，每面煎 10 秒。煎完将螃蟹取出备用\n- 将螃蟹盖放入锅中，使用勺子舀起锅中热油泼到蟹盖中，煎封住蟹盖中的蟹黄，煎 20 秒后取出备用\n- 不用刷锅，再倒入 10ml 食用油，大火让油温升高至轻微冒烟，将大蒜末，洋葱碎倒入，炒 10 秒钟\n- 将咖喱块放入锅中炒化（10 秒），放入煎好的螃蟹，翻炒均匀\n- 倒入开水 300ml，焖煮 3 分钟。\n- 焖煮完后，倒入椰浆和蛋清，关火，关火后不断翻炒，一直到酱汁变浓稠。\n- 出锅'), Document(metadata={'主标题': '咖喱炒蟹的做法', '二级标题': '附加内容'}, page_content='## 附加内容  \n- 做法参考：[十几年澳门厨房佬教学挂汁的咖喱蟹怎么做](https://www.bilibili.com/video/BV1Nq4y1W7K9)  \n如果您遵循本指南的制作流程而发现有问题或可以改进的流程，请提出 Issue 或 Pull request 。')]
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]

                for i, chunk in enumerate(md_chunks):# 示例：i= 2 ，chunk=Document(metadata={'主标题': '咖喱炒蟹的做法', '二级标题': '计算'}, page_content='## 计算  \n每次制作前需要确定计划做几份。一份正好够 1 个人食用  \n总量：  \n- 肉蟹 1 只（大约 300g） * 份数\n- 咖喱块 15g（一小块）*份数\n- 椰浆 100ml*份数\n- 鸡蛋 1 个 *份数\n- 洋葱 200g *份数\n- 大蒜 5 瓣 *份数')
                    # 为子块分配唯一ID
                    child_id = str(uuid.uuid4())

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i      # 在父文档中的位置
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)# 这里的extend是将分割后的子块列表添加到总的chunk列表中，而不是将整个列表作为一个元素添加

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        """
        按分类过滤文档
        
        Args:
            category: 菜品分类
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('category') == category]
    
    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        """
        按难度过滤文档
        
        Args:
            difficulty: 难度等级
            
        Returns:
            过滤后的文档列表
        """
        return [doc for doc in self.documents if doc.metadata.get('difficulty') == difficulty]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        categories = {}
        difficulties = {}

        for doc in self.documents:
            # 统计分类
            category = doc.metadata.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1

            # 统计难度
            difficulty = doc.metadata.get('difficulty', '未知')
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'categories': categories,
            'difficulties': difficulties,
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'dish_name': doc.metadata.get('dish_name'),
                'category': doc.metadata.get('category'),
                'difficulty': doc.metadata.get('difficulty'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{dish_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs
