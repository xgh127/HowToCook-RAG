# 评估体系

## 设计原则

原评估体系使用RAGAS框架，但存在以下问题：
1. **需要数据真值**：RAGAS需要理想的参考答案，但菜谱推荐无标准答案
2. **黑盒不可定位**：RAGAS给出分数但无法定位具体问题（是分块问题？检索问题？还是生成问题？）
3. **样本需求大**：需要大量测试用例才能稳定评估

**新评估体系采用三层架构：**

| 层级 | 评估内容 | 指标 | 无需真值 | 快速反馈 |
|------|---------|------|---------|---------|
| Layer 1 | 单元测试 | 分块/元数据/路由/BM25分词 | ✅ | ✅ |
| Layer 2 | 检索质量 | Recall@K, Filter Accuracy, Category Accuracy | ✅ | ✅ |
| Layer 3 | 端到端质量 | LLM 1-5分制打分 | ✅ | ✅ |

## Layer 2: 检索质量评估（核心）

### Detail查询（有明确答案）

- **指标：** Recall@K, Precision@K, MRR
- **示例：** "宫保鸡丁怎么做？" → 期望召回"宫保鸡丁"
- **评估方式：** 精确匹配标注的期望菜品

### List查询（无唯一答案）

- **原指标：** Recall@K（不合理，因为正确答案不唯一）
- **新指标：** Filter Accuracy@K + Category Accuracy@K
- **示例：** "推荐几个简单的素菜" → 检查返回的结果是否满足"简单"+"素菜"的条件

**为什么List查询不用Recall@K？**
- 知识库中有几十个"简单素菜"，测试集只标注了5个期望菜品
- 系统返回3个正确的简单素菜，但可能不在期望列表中 → Recall=0（不公平）
- 新指标关注"返回的结果是否满足过滤条件"，更合理

## 运行评估

```bash
# 运行完整评估
python eval/run_all_eval.py

# 单独运行检索评估
python eval/retrieval_eval.py

# 单独运行单元测试
python eval/unit_test.py

# 单独运行端到端评估
python eval/end2end_eval.py

# 运行对比评估（Baseline vs Optimized）
python eval/benchmark.py
```

## 测试数据集

`tests/retrieval_test_set.json`：30条测试用例

- **Detail查询**（17条）：有明确答案，如"宫保鸡丁怎么做"
- **List查询**（13条）：开放式推荐，如"推荐几个简单的素菜"

每条用例包含：
- `query`: 查询文本
- `relevant_dishes`: 期望召回的菜品列表
- `relevant_categories`: 期望的分类
- `query_type`: detail / list
- `difficulty`: easy / medium / hard

## 对比评估

### 2026.04.27 检索优化效果

| 指标 | Baseline | Optimized | 提升 |
|------|----------|-----------|------|
| recall@3 | 0.476 | 0.758 | +0.282 (+59.2%) |
| recall@5 | 0.487 | 0.758 | +0.271 (+55.6%) |
| recall@10 | 0.498 | 0.758 | +0.260 (+52.2%) |
| precision@3 | 0.278 | 0.424 | +0.146 (+52.5%) |
| mrr | 0.533 | 0.818 | +0.285 (+53.5%) |

**实验设计：**
- **Baseline**：跳过过滤条件提取，直接 `hybrid_search(query)`
- **Optimized**：提取过滤条件 → `metadata_filtered_search(query, filters)`

**关键发现：**
1. **Detail查询**（如"宫保鸡丁怎么做"）：优化前后持平，因为这类查询本身就有明确答案，语义检索就能召回
2. **List查询**（如"推荐几个简单的素菜"）：优化后显著提升，因为过滤条件确保返回结果满足"简单"+"素菜"的要求
