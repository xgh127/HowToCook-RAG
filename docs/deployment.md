# 部署说明

## 环境准备

```bash
conda create -n cook-rag-1 python=3.12.7
conda activate cook-rag-1
pip install -r requirements.txt
```

## API Key 配置

```bash
echo "MOONSHOT_API_KEY=your_key" > .env
```

## 嵌入模型选择

### 方案1：HuggingFace（推荐）

若可正常访问 HuggingFace：

```python
# index_construction.py
from langchain_huggingface import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

更多模型：[HuggingFace Embedding Models](https://huggingface.co/models?filter=embedding)

### 方案2：Ollama本地模型

若无法访问 HuggingFace：

```bash
ollama pull bge-m3:567m
```

```python
# index_construction.py
from langchain_ollama import OllamaEmbeddings

self.embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
    base_url="http://localhost:11434"
)
```

**注意：**
- `nomic-embed-text` 仅支持英文，中文效果差
- `shaw/dmeta-embedding-zh` 有上下文窗口限制
- `bge-m3:567m` 性能中等但稳定可用

## 运行

```bash
python main.py
```
