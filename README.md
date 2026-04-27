# 智能食谱问答系统

基于 [HowToCook](https://github.com/Anduin2017/HowToCook) 菜谱数据构建的 RAG 问答系统。

**核心能力：**
- 询问具体菜品做法："宫保鸡丁怎么做？"
- 寻求菜品推荐："推荐几个简单的素菜"
- 获取食材信息："红烧肉需要什么食材？"

---

## 快速开始

```bash
# 1. 创建环境
conda create -n cook-rag-1 python=3.12.7
conda activate cook-rag-1

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
echo "MOONSHOT_API_KEY=your_key" > .env

# 4. 运行
python main.py
```

> 若无法访问 HuggingFace，使用 Ollama 本地嵌入模型：`ollama pull bge-m3:567m`，详见[部署说明](docs/deployment.md)。

---

## 项目架构

```
├── main.py                     # 主程序入口
├── config.py                   # 配置管理
├── rag_modules/               # 核心模块
│   ├── data_preparation.py    # 数据准备（父子文档分块）
│   ├── index_construction.py  # 索引构建（FAISS + 缓存）
│   ├── retrieval_optimization.py # 检索优化（混合检索 + 元数据过滤）
│   └── generation_integration.py # 生成集成（查询路由 + 重写）
├── eval/                      # 评估体系
│   ├── benchmark.py           # 检索优化对比评估
│   ├── retrieval_eval.py      # 检索质量评估
│   ├── end2end_eval.py        # 端到端评估
│   └── unit_test.py           # 单元测试
└── tests/                     # 测试数据集
    └── retrieval_test_set.json
```

---

## 更新日志

| 日期 | 更新内容 | 详情 |
|------|---------|------|
| 2026.04.28 | Detail查询过滤修复 | 查询类型感知过滤，Detail跳过metadata过滤 |
| 2026.04.27 | 检索系统全面升级 | [检索优化详情](docs/retrieval-optimization.md) |
| 2026.04.21 | 前置元数据过滤 | 将过滤条件提取移至检索之前 |
| 2026.04.20 | BM25 分词修复 | 添加 jieba 中文分词 |

---

## 评估体系

### 检索优化效果（2026.04.28 修复后）

对比实验：Baseline（无过滤优化）vs Optimized（查询类型感知过滤）

| 指标         | Baseline | Optimized | 提升           |
|--------------|----------|-----------|----------------|
| recall@3     | 0.631    | **0.842** | **+0.211 (+33.4%)** |
| recall@5     | 0.642    | **0.853** | **+0.211 (+32.9%)** |
| recall@10    | 0.653    | **0.864** | **+0.211 (+32.3%)** |
| precision@3  | 0.433    | **0.644** | **+0.211 (+48.7%)** |
| mrr          | 0.800    | **0.900** | **+0.100 (+12.5%)** |

**按查询类型细分：**

| 查询类型 | 指标 | Baseline | Optimized | 提升 |
|---------|------|----------|-----------|------|
| Detail | recall@3 | 0.839 | 0.839 | 保持 |
| **List** | **recall@3** | **0.359** | **0.846** | **+0.487 (+135.7%)** |
| **List** | **filter_accuracy@3** | **0.359** | **0.846** | **+0.487 (+135.7%)** |

运行对比评估：
```bash
python eval/benchmark.py
```

[→ 查看完整评估体系说明](docs/evaluation.md)

---

## 模块简介

### 1. 数据准备：父子文档分块

- **子块**（~200字）：按 Markdown 标题分块，用于精确检索
- **父文档**（完整菜谱）：检索到子块后，返回完整父文档用于生成
- 解决"按标题分块导致上下文缺失"问题

[→ 详情](docs/data-preparation.md)

### 2. 索引构建：FAISS + 缓存

- 向量索引：语义相似度检索
- BM25 索引：关键词匹配检索（jieba 中文分词）
- 索引缓存：首次构建后保存到本地，后续秒级加载

[→ 详情](docs/index-construction.md)

### 3. 检索优化：混合检索 + 元数据过滤

- **混合检索**：向量 + BM25，RRF 重排融合
- **查询类型感知过滤**：List查询应用过滤，Detail查询跳过过滤
- **三层过滤提取**：精确匹配 → 同义词扩展 → 场景词映射
- **多值 OR 检索**：支持 `category: ['荤菜', '素菜']` 等 OR 条件

[→ 详情](docs/retrieval-optimization.md)

### 4. 生成集成：查询路由 + 重写

- 智能路由：list / detail / general 三种查询类型
- 查询重写：模糊查询优化为更精确的检索词
- 多模式生成：列表模式 / 详细模式 / 基础模式

[→ 详情](docs/generation-integration.md)

---

## 未来方向

- [ ] 知识图谱：食材关联查询、食材组合推荐
- [ ] 多模态：菜品图片检索、视觉搜索
- [ ] 个性化：用户偏好学习、营养分析（MCP 工具集成）

---

## 致谢

- 教程：[Datawhale/all-in-rag](https://github.com/datawhalechina/all-in-rag)
- 数据：[Anduin2017/HowToCook](https://github.com/Anduin2017/HowToCook)
