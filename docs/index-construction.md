# 索引构建模块

## 混合索引

系统同时维护两种索引：

| 索引类型 | 原理 | 优势 | 适用场景 |
|---------|------|------|---------|
| 向量索引（FAISS） | 语义相似度 | 理解查询意图、同义词 | "简单易做的菜" |
| BM25索引 | 关键词匹配 | 精确匹配菜名、食材 | "宫保鸡丁" |

## 索引缓存

首次构建后将FAISS索引保存到本地，后续启动时直接加载：

- **首次构建**：~几分钟（取决于文档数量和嵌入模型）
- **后续加载**：~几秒钟

## 嵌入模型选择

### 方案1：HuggingFace（推荐，需能访问HuggingFace）

```python
from langchain_huggingface import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### 方案2：Ollama本地模型（无法访问HuggingFace时）

```bash
ollama pull bge-m3:567m
```

```python
from langchain_ollama import OllamaEmbeddings

self.embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
    base_url="http://localhost:11434"
)
```

> 注：`nomic-embed-text` 仅支持英文，`shaw/dmeta-embedding-zh` 有上下文窗口限制。bge-m3:567m 性能中等但稳定可用。

## BM25中文分词

使用 jieba 进行中文分词：

```python
import jieba

def chinese_preprocess(text: str) -> list:
    return list(jieba.cut(text))

bm25_retriever = BM25Retriever.from_documents(
    documents=chunks,
    k=5,
    preprocess_func=chinese_preprocess
)
```

## RRF重排

综合两种检索方式的排名信息：

```python
score = 1.0 / (k + rank + 1)  # k=60
```

RRF 可能并不是效果最好的重排方式，但是够用。如需更先进的重排方法（ColBERT、RankLLM 等）可自行尝试。
