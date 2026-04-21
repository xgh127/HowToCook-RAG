"""
RAG 官方标准化评估脚本
适配你的 RecipeRAGSystem | 上下文=父文档 | GT基于HowToCook生成
严格对齐官方示例：EvaluationDataset + LangchainLLMWrapper + 官方指标
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import sys
# ======================
# 1. 导入你的RAG系统
# ======================
sys.path.append(str(Path(__file__).parent.parent))
from main import RecipeRAGSystem, RAGConfig

# ======================
# 2. RAGAS 官方导入
# ======================
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# 模型
from langchain_openai import ChatOpenAI

# ======================
# 3. 基础配置
# ======================
load_dotenv()
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

# 评估用LLM（Kimi）
llm = ChatOpenAI(
    api_key=MOONSHOT_API_KEY,
    base_url="https://api.moonshot.cn/v1",
    model="kimi-k2-0711-preview",
    temperature=0.1
)
evaluator_llm = LangchainLLMWrapper(llm)

# 测试配置
SAVE_PATH = Path(__file__).parent / "output" / "rag_official_eval_data.json"
SAVE_PATH.parent.mkdir(exist_ok=True)

# 测试问题 + HowToCook标准答案
sample_queries = [
    "宫保鸡丁怎么做？",
    "推荐适合健身吃的家常菜",
    "水煮鱼需要哪些食材？",
    "简单的素菜有哪些？",
    "咖喱炒蟹的制作步骤是什么？"
]

expected_responses = [
    "菜品介绍 \n 宫保鸡丁是经典老派川菜，有简易版和风味更浓郁的复杂版两种做法，默认一人份（两人也可食用）。\n\n## 🛒 所需食材（1–2 人份）\n### 必备原料 \n 主料：手枪腿 1 支（约 350g，也可用鸡胸肉）、大葱 1 根（约 180g）、熟花生 150g、干辣椒 / 二荆条 10g\n 基础调料：生抽 10g、料酒 15g、盐 2g、淀粉 25g、植物油 20g、白糖 2g\n 进阶调料：老抽 5g、香醋 5g、花椒 5g、鸡精 2g、芝麻油 10g\n 可选配料：莴笋 250g、豆瓣酱 10g、油泼辣子 5g\n\n## 👨‍🍳 制作步骤 \n### 简易版做法 \n1. 处理鸡肉：手枪腿去骨，用刀背拍松后切 1.5cm 见方肉丁，清水浸泡 10 分钟沥干 \n2. 制备辅料：葱绿加姜片、开水泡成葱姜水，葱白切粒，花生微波炉高火 5 分钟焙干 \n3. 腌制鸡丁：鸡丁加盐、老抽、料酒、15g 淀粉拌匀，分次加葱姜水至粘手，冷藏腌制 1 小时 \n4. 焙香香料：小火将干辣椒、花椒焙香捞出 \n5. 煎制鸡丁：大火热油至 7 成热，下鸡丁煎至发白翻面，翻炒均匀 \n6. 焖煮调味：加葱白粒和葱姜水，中小火焖 2 分钟，再放花生、香料、鸡精、香醋、白糖炒匀 \n7. 勾芡出锅：水淀粉勾芡收汁，关火淋芝麻油即可 \n\n### 复杂版做法（推荐）\n1. 预处理：莴笋切 1cm 小丁，二荆条切段，鸡肉处理、腌制同简易版 \n2. 炒花生：中火热油炒生花生至微焦捞出，留底油 \n3. 炒鸡丁：底油烧热下鸡丁，加豆瓣酱翻炒 1 分钟，再放莴笋丁炒 1 分钟 \n4. 焖煮：加葱白粒、葱姜水、二荆条段，中小火焖 2 分钟 \n5. 调味收汁：放花生、花椒、调料炒匀，水淀粉勾芡 \n6. 出锅：淋芝麻油和油泼辣子，翻炒 10 秒即可 \n\n## 💡 制作技巧 \n1. 辣椒可按口味调整，怕辣可去籽 \n2. 焖煮加水需用热水，避免鸡肉变柴",
    "为您推荐以下菜品：\n1. 西葫芦炒鸡蛋\n2. 瘦肉土豆片\n3. 素炒豆角",
    "水煮鱼需要巴沙鱼、时令蔬菜、红油豆瓣酱、藤椒油、大蒜、盐、糖等食材，是经典川味麻辣菜品。",
    "为您推荐：素炒豆角、清炒花菜、鸡蛋花",
    "🥘 菜品介绍 \n 咖喱炒蟹是经典泰式海鲜菜品，操作简单，单人份刚好够 1 人食用，烹饪难度为★★★★。\n\n## 🛒 所需食材（1 人份）\n 主料：青蟹（肉蟹）1 只（约 300g）、洋葱 200g、大蒜 5 瓣 \n 调料：咖喱块 15g、椰浆 100ml、鸡蛋 1 个、生粉（淀粉）适量、食用油 30ml\n\n## 👨‍🍳 制作步骤 \n1. 处理螃蟹：肉蟹掀盖后对半砍开，蟹钳用刀背轻轻拍裂，切口和蟹钳蘸一下生粉，撒 5g 生粉到蟹盖中盖住蟹黄，备用 \n2. 备料：洋葱切成洋葱碎，大蒜切碎，烧一壶开水备用 \n3. 煎蟹封黄：起锅烧油，倒入约 20ml 食用油，油温升高后将螃蟹切口朝下放入锅中煎 20 秒，翻面每面煎 10 秒后取出；蟹盖放入锅中，用热油泼淋煎 20 秒后取出 \n4. 炒香底料：锅中再倒入 10ml 食用油，大火烧至油温轻微冒烟，放入蒜末、洋葱碎翻炒 10 秒 \n5. 炒化咖喱：将咖喱块放入锅中炒化 10 秒，放入煎好的螃蟹翻炒均匀 \n6. 焖煮：倒入 300ml 开水，焖煮 3 分钟 \n7. 收汁挂糊：倒入椰浆和蛋清，关火后不断翻炒至酱汁变浓稠 \n8. 出锅"
]

# ======================
# 4. 初始化你的RAG系统
# ======================
def init_rag_system():
    config = RAGConfig(moonshot_api_key=MOONSHOT_API_KEY)
    rag = RecipeRAGSystem(config=config)
    rag.initialize_system()
    rag.build_knowledge_base()
    return rag

# ======================
# 5. 官方格式：生成评估数据（核心：上下文=父文档）
# ======================
def generate_official_dataset(rag):
    print("\n🚀 重新生成官方评估数据（上下文=完整父文档）...")
    dataset = []
    if(SAVE_PATH.exists()):
        print(f"⚠️ 评估数据已存在，直接加载：{SAVE_PATH}")
        with open(SAVE_PATH, "r", encoding="utf-8") as f:
            # 加载数据,组合成dataset格式
            loaded_data = json.load(f)
            dataset.extend(loaded_data)
        return dataset
    for query, reference in zip(sample_queries, expected_responses):
        # 1. 检索chunk → 自动获取父文档（你的系统原生逻辑）
        rewritten_query = rag.generation_module.query_rewrite(query)
        chunks = rag.retrieval_module.hybrid_search(rewritten_query)
        parent_docs = rag.data_module.get_parent_documents(chunks)
        
        # 2. 提取 父文档内容 作为上下文（严格匹配你的生成逻辑）
        retrieved_contexts = [doc.page_content for doc in parent_docs]
        
        # 3. 生成回答
        response = rag.ask_question(query)
        
        # 4. 官方格式拼接
        dataset.append({
            "user_input": query,
            "retrieved_contexts": retrieved_contexts,  # ✅ 父文档！
            "response": response,
            "reference": reference
        })

    # 保存数据（留档）
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    return dataset

# ======================
# 6. 官方评估（完全复刻你的示例代码）
# ======================
def run_official_evaluation():
    # 初始化系统
    rag = init_rag_system()
    
    # 生成数据
    dataset = generate_official_dataset(rag)
    
    # 加载到官方数据集
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    print(f"\n✅ 数据集构建完成：共 {len(evaluation_dataset)} 条样本")

    # 官方评估
    print("\n📊 开始 RAGAS 官方评估...")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),    # 上下文召回率
            Faithfulness(),         # 事实忠实度
            FactualCorrectness()    # 事实准确性
        ],
        llm=evaluator_llm
    )

    # 输出结果
    print("\n" + "="*60)
    print("🎉 RAG 官方评估结果")
    print("="*60)
    print(result)
    print("="*60)
    print(f"📁 评估数据已保存：{SAVE_PATH}")

# ======================
# 运行
# ======================
if __name__ == "__main__":
    run_official_evaluation()