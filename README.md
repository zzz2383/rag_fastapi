# RAG FastAPI - 基于检索增强生成的知识问答系统

基于 FastAPI + LangChain + DeepSeek 构建的 RAG（检索增强生成）知识问答系统，以学生手册为知识库，提供智能问答服务。

## 功能特性

- **混合检索**：结合 BM25 关键词检索（权重 0.4）与 FAISS 语义向量检索（权重 0.6），提升召回质量
- **查询扩展**：利用 LLM 将用户问题改写为 2 个同义表述，3 路并发检索后去重，提高覆盖率
- **语义分块**：使用 SemanticChunker 对 PDF 文档进行语义级切分，控制 Token 消耗
- **启动缓存**：文档分块和向量索引在首次构建后持久化至磁盘，加速后续重启
- **Web 界面**：响应式单页应用，支持深色模式，桌面端与移动端均可使用
- **评估工具**：附带自动化测试脚本，支持语义相似度与 ROUGE-L 指标评估

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| Web 框架 | FastAPI + Uvicorn |
| LLM | DeepSeek Chat (deepseek-chat) |
| 向量检索 | FAISS (CPU) |
| 关键词检索 | BM25 (LangChain) |
| 文本嵌入 | sentence-transformers/all-MiniLM-L6-v2 |
| 文档解析 | PyMuPDF |
| 文本分块 | LangChain SemanticChunker |
| 前端 | 原生 HTML + CSS + JavaScript |

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API 密钥

### 安装

```bash
# 克隆仓库
git clone https://github.com/zzz2383/rag_fastapi.git
cd rag_fastapi

# 安装依赖
pip install -r requirements.txt
```

### 配置环境变量

在项目根目录创建 `.env` 文件：

```ini
DEEPSEEK_API_KEY=your_deepseek_api_key_here
# 可选：HuggingFace 镜像地址（国内用户）
HF_ENDPOINT=https://hf-mirror.com
```

### 运行

```bash
python app.py
```

服务启动后，访问 `http://localhost:8000` 即可打开问答界面。

首次启动时会自动执行以下流程：

1. 加载 `knowledge_base/学生手册.pdf`
2. 对文档进行语义分块
3. 构建 FAISS 向量索引
4. 持久化缓存到本地磁盘

后续启动将直接从缓存加载，速度显著提升。

## API 文档

### 问答接口

```
POST /ask
```

请求体：

```json
{
  "question": "学生手册中关于奖学金的规定是什么？"
}
```

响应体：

```json
{
  "answer": "根据学生手册相关规定，奖学金...",
  "elapsed_time": 3.25
}
```

### 静态页面

```
GET /  -> 重定向至 /static/index.html
```

## 项目结构

```
rag_fastapi/
├── app.py                      # 主程序（FastAPI 应用 + RAG 逻辑）
├── requirements.txt            # Python 依赖
├── .env                        # 环境变量配置（需自行创建）
├── knowledge_base/
│   └── 学生手册.pdf            # 知识库文档
├── faiss_index_deepseek/       # FAISS 向量索引缓存
├── chunks_cache.pkl            # 文档分块缓存
├── static/
│   └── index.html              # Web 前端页面
└── test/
    ├── test_rag.py             # 自动化评估脚本
    ├── generate_testset.py     # 测试集生成工具
    ├── test_qa.json            # 问答测试数据集
    └── test_report.json        # 评估报告
```

## 测试与评估

### 生成测试集

从知识库中随机采样文档片段，利用 LLM 自动生成 Q&A 对：

```bash
python test/generate_testset.py
```

### 运行评估

确保服务已在 `localhost:8000` 运行，然后执行：

```bash
python test/test_rag.py
```

评估指标：

- **语义相似度**：基于 paraphrase-multilingual-MiniLM-L12-v2 计算余弦相似度
- **ROUGE-L F1**：评估生成答案与参考答案的文本重叠度

## 关键配置

以下参数可在 `app.py` 中调整：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| LLM 模型 | deepseek-chat | 使用的语言模型 |
| 温度 | 0.3 | 生成随机性控制 |
| 最大 Token | 500 | 回答长度上限 |
| 嵌入模型 | all-MiniLM-L6-v2 | 文本向量化模型 |
| BM25 检索数 | 35 | 关键词检索返回数量 |
| FAISS 检索数 | 35 | 向量检索返回数量 |
| 权重配比 | BM25 0.4 / FAISS 0.6 | 混合检索权重 |
| 查询扩展数 | 2 | 同义改写数量 |

## 开发规划

- [ ] 支持多知识库切换
- [ ] 添加流式输出（SSE）
- [ ] 支持文档上传与实时索引
- [ ] Docker 容器化部署
- [ ] 增加对话历史与上下文记忆
