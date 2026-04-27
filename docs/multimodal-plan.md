# HowToCook-RAG 多模态功能设计方案

## 目标
支持用户上传菜品图片，系统自动识别菜品名称，并返回对应的菜谱信息。

## 方案选型：VLM 图像识别 + 文本检索（推荐）

### 为什么选这个方案

| 方案 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **VLM + 文本检索** ⭐ | 用视觉大模型识别菜名 → 走现有RAG检索 | 实现简单、准确率高、无需重建索引 | 依赖VLM API | **推荐** |
| CLIP 图像向量检索 | 图片→向量，与文本向量对齐检索 | 端到端、无需识别菜名 | 需要菜品图片数据集训练、准确率受限 | 有大规模菜品图库时 |
| 传统CV分类 | 训练菜品分类模型 | 速度快、成本低 | 需要大量标注数据、扩展性差 | 固定菜品库且数据充足 |

**结论**：VLM + 文本检索是最佳路径。现有VLM（GPT-4V、Qwen-VL、Moonshot-VL等）识别菜品准确率已很高，且无需改动现有检索架构。

---

## 架构设计

```
用户上传图片
    │
    ▼
┌─────────────────┐
│  图像识别模块    │  ← 新增：VLM识别菜品名称
│  (ImageAnalyzer) │
└────────┬────────┘
         │ 返回: "这道菜是宫保鸡丁"
         ▼
┌─────────────────┐
│   查询重写模块   │  ← 现有：将识别结果转为检索query
│  (query_rewrite) │
└────────┬────────┘
         │ 返回: "宫保鸡丁的做法"
         ▼
┌─────────────────┐
│   检索模块       │  ← 现有：hybrid_search / metadata_filtered_search
│  (Retrieval)     │
└────────┬────────┘
         │ 返回: 相关菜谱文档
         ▼
┌─────────────────┐
│   生成模块       │  ← 现有：生成回答
│  (Generation)    │
└─────────────────┘
```

---

## 具体实现步骤

### Step 1: 新增图像识别模块

文件：`rag_modules/image_analyzer.py`

```python
"""
图像识别模块 - 使用VLM识别菜品图片
"""
import os
import base64
import logging
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_community.chat_models.moonshot import MoonshotChat

logger = logging.getLogger(__name__)


class ImageAnalyzerModule:
    """图像识别模块 - 识别菜品图片并返回菜名"""

    # 识别菜品的系统提示词
    DISH_RECOGNITION_PROMPT = """你是一位专业的菜品识别专家。
请仔细观察用户上传的图片，判断这是什么菜。

要求：
1. 只返回菜品名称，不要任何解释
2. 如果无法确定，返回"未知菜品"
3. 优先使用中文菜名
4. 如果是家常菜但有多种叫法，返回最常见的名称

示例输出格式：
- 宫保鸡丁
- 西红柿炒鸡蛋
- 红烧肉
- 未知菜品
"""

    def __init__(self, api_key: str, model_name: str = "kimi-k2-0711-preview"):
        self.api_key = api_key
        self.model_name = model_name
        self.vlm = MoonshotChat(
            model=model_name,
            temperature=0.1,
            max_tokens=100,
            moonshot_api_key=api_key
        )

    def analyze_image(self, image_path: str) -> str:
        """
        分析图片，返回识别的菜品名称

        Args:
            image_path: 图片文件路径

        Returns:
            菜品名称，如"宫保鸡丁"
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 读取图片并转为base64
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 构建多模态消息
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.DISH_RECOGNITION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        )

        # 调用VLM识别
        response = self.vlm.invoke([message])
        dish_name = response.content.strip()

        logger.info(f"图片识别结果: {dish_name}")
        return dish_name

    def analyze_image_bytes(self, image_bytes: bytes) -> str:
        """
        分析图片字节流，返回识别的菜品名称

        Args:
            image_bytes: 图片二进制数据

        Returns:
            菜品名称
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": self.DISH_RECOGNITION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        )

        response = self.vlm.invoke([message])
        dish_name = response.content.strip()

        logger.info(f"图片识别结果: {dish_name}")
        return dish_name
```

### Step 2: 修改主程序支持图片输入

文件：`main.py`

```python
# 新增导入
from rag_modules.image_analyzer import ImageAnalyzerModule

class RecipeRAGSystem:
    def __init__(self, config: RAGConfig = None):
        # ... 现有初始化 ...
        self.image_analyzer = None  # 新增

    def setup_image_analyzer(self):
        """初始化图像识别模块"""
        if self.config.moonshot_api_key:
            self.image_analyzer = ImageAnalyzerModule(
                api_key=self.config.moonshot_api_key,
                model_name=self.config.vlm_model  # 新增配置项
            )
            logger.info("图像识别模块初始化完成")

    def ask_with_image(self, image_path: str, question: str = "这道菜怎么做？") -> str:
        """
        上传图片并提问

        Args:
            image_path: 图片文件路径
            question: 用户问题（可选，默认"这道菜怎么做？"）

        Returns:
            回答文本
        """
        if not self.image_analyzer:
            return "图像识别模块未初始化，请检查API配置。"

        # 1. 识别菜品
        print(f"🔍 正在识别图片中的菜品...")
        dish_name = self.image_analyzer.analyze_image(image_path)

        if dish_name == "未知菜品":
            return "抱歉，无法识别图片中的菜品。请尝试上传更清晰的图片，或直接输入菜品名称。"

        print(f"✅ 识别结果: {dish_name}")

        # 2. 构建查询（将识别结果融入用户问题）
        if "这道菜" in question or "这个" in question:
            # 替换指代词
            query = question.replace("这道菜", dish_name).replace("这个", dish_name)
        else:
            # 默认查询
            query = f"{dish_name}怎么做"

        print(f"📝 生成查询: {query}")

        # 3. 走现有RAG流程
        return self.ask_question(query)
```

### Step 3: 新增配置项

文件：`config.py`

```python
@dataclass
class RAGConfig:
    # ... 现有配置 ...

    # 多模态配置（新增）
    vlm_model: str = "kimi-k2-0711-preview"  # 视觉模型，可与LLM共用
    enable_multimodal: bool = True  # 是否启用多模态
```

### Step 4: 新增交互入口

文件：`main.py` 的 `if __name__ == "__main__"` 部分

```python
def main():
    # ... 现有初始化 ...
    rag = RecipeRAGSystem(config)
    rag.setup_image_analyzer()  # 新增

    print("\n" + "="*50)
    print("🍳 智能食谱问答系统（支持图片识别）")
    print("="*50)
    print("\n输入方式：")
    print("  1. 直接输入问题（如：宫保鸡丁怎么做）")
    print("  2. 上传图片（输入: /image 图片路径）")
    print("  3. 退出（输入: quit）")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n❓ 请输入问题或图片路径: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            # 图片输入模式
            if user_input.startswith("/image "):
                image_path = user_input[7:].strip()
                if os.path.exists(image_path):
                    answer = rag.ask_with_image(image_path)
                    print(f"\n🤖 回答:\n{answer}")
                else:
                    print(f"❌ 图片不存在: {image_path}")
                continue

            # 文本输入模式
            answer = rag.ask_question(user_input)
            print(f"\n🤖 回答:\n{answer}")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
```

---

## 扩展方案（可选）

### 扩展1：图片URL直接输入
支持用户粘贴网络图片URL：

```python
def analyze_image_url(self, image_url: str) -> str:
    """分析网络图片"""
    message = HumanMessage(
        content=[
            {"type": "text", "text": self.DISH_RECOGNITION_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    response = self.vlm.invoke([message])
    return response.content.strip()
```

### 扩展2：识别置信度 + 多候选
当识别不确定时，返回多个候选菜名，让用户选择：

```python
DISH_RECOGNITION_PROMPT_V2 = """你是一位专业的菜品识别专家。
请观察图片，判断这是什么菜。

要求：
1. 返回最可能的1-3个菜品名称，按可能性排序
2. 格式：菜名1 | 菜名2 | 菜名3
3. 如果完全无法识别，返回"未知"

示例：
宫保鸡丁 | 辣子鸡丁 | 酱爆鸡丁
"""
```

### 扩展3：菜品相似度推荐
如果识别的菜品不在库中，推荐相似菜品：

```python
def ask_with_image_smart(self, image_path: str) -> str:
    dish_name = self.image_analyzer.analyze_image(image_path)

    # 先精确检索
    result = self.ask_question(f"{dish_name}怎么做")

    # 如果找不到，用向量相似度找最接近的菜品
    if "没有找到" in result:
        similar = self.retrieval_module.hybrid_search(dish_name, top_k=3)
        # ... 推荐相似菜品 ...

    return result
```

### 扩展4：CLIP图像向量检索（长期方向）
如果有菜品图片数据集，可以训练CLIP模型：

```
菜品图片 → CLIP编码 → 向量索引
用户图片 → CLIP编码 → 向量相似度检索 → 返回最相似的菜品
```

优点：无需识别菜名，直接图像→图像匹配
缺点：需要构建菜品图片库、训练成本

---

## 文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `rag_modules/image_analyzer.py` | 图像识别模块 |
| 修改 | `rag_modules/__init__.py` | 导出ImageAnalyzerModule |
| 修改 | `config.py` | 添加vlm_model、enable_multimodal配置 |
| 修改 | `main.py` | 添加ask_with_image方法、图片交互入口 |
| 新增 | `docs/multimodal-plan.md` | 本文档 |

---

## 成本估算

| 项目 | 说明 | 估算 |
|------|------|------|
| VLM API调用 | 每次图片识别1次调用 | ~0.01-0.05元/次 |
| 现有RAG流程 | 无变化 | 与原成本相同 |
| 开发成本 | 约半天工作量 | - |

---

## 验证方式

1. 准备5-10张常见菜品图片（宫保鸡丁、红烧肉、西红柿炒鸡蛋等）
2. 运行 `python main.py`
3. 输入 `/image path/to/宫保鸡丁.jpg`
4. 检查识别结果和返回的菜谱是否匹配

## 下一步

1. 确认VLM API提供商（Moonshot/GPT-4V/Qwen-VL）
2. 实现 `image_analyzer.py`
3. 修改 `main.py` 和 `config.py`
4. 测试验证
