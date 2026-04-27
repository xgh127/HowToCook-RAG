# HowToCook-RAG 分层评估方案

## 评估体系架构

```
┌─────────────────────────────────────────┐
│  Layer 3: 端到端质量评估                 │
│  文件: end2end_eval.py                   │
│  指标: Faithfulness, Usefulness, Structure│
│  方法: LLM-as-Judge (低成本版)            │
│  频率: 每次重大迭代                       │
│  成本: 中等 (调用LLM打分)                 │
├─────────────────────────────────────────┤
│  Layer 2: 检索质量评估                   │
│  文件: retrieval_eval.py                 │
│  指标: Recall@K, Precision@K, MRR        │
│  方法: 人工标注query-doc相关性            │
│  频率: 每次调参后                        │
│  成本: 低 (自动化运行)                    │
├─────────────────────────────────────────┤
│  Layer 1: 单元测试                       │
│  文件: unit_test.py                      │
│  指标: 功能正确性                        │
│  方法: pytest/unittest                   │
│  频率: 每次代码改动                      │
│  成本: 极低                              │
└─────────────────────────────────────────┘
```

## 文件说明

| 文件 | 作用 | 运行方式 |
|------|------|----------|
| `tests/retrieval_test_set.json` | 30条检索测试用例 | 数据文件 |
| `eval/unit_test.py` | 单元测试 | `python eval/unit_test.py` |
| `eval/retrieval_eval.py` | 检索质量评估 | `python eval/retrieval_eval.py` |
| `eval/end2end_eval.py` | 端到端评估 | `python eval/end2end_eval.py` |
| `eval/ab_test.py` | A/B测试框架 | `python eval/ab_test.py` |
| `eval/run_all_eval.py` | 一键运行全部 | `python eval/run_all_eval.py` |

## 快速开始

### 1. 运行单元测试（Layer 1）

```bash
python eval/unit_test.py
```

测试内容：
- 文档加载和元数据提取
- Markdown分块逻辑
- 父子文档映射
- 查询路由分类
- RRF重排算法
- 索引保存/加载

### 2. 运行检索评估（Layer 2）

```bash
python eval/retrieval_eval.py
```

输出指标：
- `recall@3/5/10`: 召回率
- `precision@3/5/10`: 精确率
- `mrr`: 平均倒数排名
- `route_accuracy`: 路由分类准确率

还会输出：
- 按查询类型分析（list/detail）
- 按难度分析（easy/medium/hard）
- 失败案例分析

### 3. 运行端到端评估（Layer 3）

```bash
python eval/end2end_eval.py
```

评估维度：
- **Faithfulness (忠实度)**: 回答是否基于检索到的食谱
- **Usefulness (实用性)**: 回答是否解决了用户问题
- **Structure (结构度)**: 回答格式是否清晰

### 4. 运行A/B测试

```bash
python eval/ab_test.py
```

对比两个RAG配置的效果，自动判断哪个更好。

### 5. 一键运行全部

```bash
python eval/run_all_eval.py
```

按提示选择要运行的评估层级。

## 测试集构建

### 检索测试集格式

```json
{
  "id": "r001",
  "query": "宫保鸡丁怎么做？",
  "relevant_dishes": ["宫保鸡丁"],
  "relevant_categories": ["荤菜"],
  "query_type": "detail",
  "difficulty": "easy"
}
```

字段说明：
- `id`: 唯一标识
- `query`: 用户问题
- `relevant_dishes`: 应该召回的菜谱名称列表
- `relevant_categories`: 相关分类（可选）
- `query_type`: 查询类型（list/detail）
- `difficulty`: 难度标注（easy/medium/hard）

### 如何扩展测试集

1. 从 `data/cook/dishes/` 下选择热门菜谱
2. 为每个菜谱写2-3个query：
   - 具体做法："XX怎么做"
   - 食材询问："XX需要什么食材"
   - 技巧询问："XX有什么技巧"
3. 标注应该召回哪些菜
4. 添加到 `tests/retrieval_test_set.json`

## 评估指标解读

### 检索指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Recall@3 | > 0.7 | top3里找到相关菜的概率 |
| Recall@5 | > 0.85 | top5里找到相关菜的概率 |
| MRR | > 0.6 | 第一个相关菜的平均排名 |
| Route Accuracy | > 0.9 | 查询路由分类准确率 |
| Filter Accuracy@3 | > 0.8 | List查询top3结果满足过滤条件的比例 |

### 按查询类型评估

**Detail查询**（如"宫保鸡丁怎么做"）：
- 不过滤，直接hybrid_search
- 目标：recall@3 > 0.8, mrr > 0.9

**List查询**（如"推荐几个简单的素菜"）：
- 应用metadata过滤
- 目标：filter_accuracy@3 > 0.8, recall@3 > 0.8

### 端到端指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Faithfulness | > 4.0 | 回答忠实于食谱 |
| Usefulness | > 4.0 | 回答实用可操作 |
| Structure | > 3.5 | 格式清晰有结构 |
| Overall | > 4.0 | 综合评分 |

## 常见问题

**Q: 为什么recall@3比recall@5低很多？**
A: 说明相关文档排在3-5位，可以考虑：
- 调大RRF的k值（让排名更平均）
- 增加向量检索的权重
- 优化embedding模型

**Q: 端到端faithfulness低怎么办？**
A: 说明LLM在编造内容，可以：
- 增强prompt约束（要求严格基于上下文）
- 添加引用溯源（要求标注信息来源）
- 减少max_tokens（限制发挥空间）

**Q: 如何对比两个版本的效果？**
A: 使用A/B测试：
```python
config_a = {'top_k': 3}
config_b = {'top_k': 5}
results = run_ab_test(config_a, config_b, test_queries)
```

## 进阶：自定义评估

### 添加新的检索指标

在 `retrieval_eval.py` 的 `evaluate` 方法中：

```python
# 添加NDCG计算
def _ndcg(relevant, retrieved, k):
    dcg = 0
    for i, item in enumerate(retrieved[:k]):
        rel = 1 if item in relevant else 0
        dcg += rel / np.log2(i + 2)
    
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0
```

### 添加新的端到端维度

在 `end2end_eval.py` 中添加新的prompt：

```python
SAFETY_PROMPT = """评估回答是否包含危险建议...
"""
```

## 与CI/CD集成

```yaml
# .github/workflows/eval.yml
name: RAG Evaluation
on: [push]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Unit Tests
        run: python eval/unit_test.py
      - name: Run Retrieval Eval
        run: python eval/retrieval_eval.py
      - name: Check Metrics
        run: |
          recall=$(cat eval/retrieval_eval_result.json | jq '.overall.recall@3')
          if (( $(echo "$recall < 0.6" | bc -l) )); then
            echo "Recall@3 too low: $recall"
            exit 1
          fi
```
