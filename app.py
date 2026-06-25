import os
import time
import asyncio
import pickle
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# -------------------- 原 RAG 模块导入 --------------------
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 加载环境变量（含 HF_ENDPOINT 镜像地址）
load_dotenv()

# -------------------- 全局变量 --------------------
retriever = None
llm = None
rewrite_chain = None
document_chain = None


# -------------------- 初始化函数（仅执行一次） --------------------
def init_rag():
    global retriever, llm, rewrite_chain, document_chain

    print(" 正在初始化 RAG 系统...")
    start_init = time.time()

    # 1. 初始化 LLM（远程 API，无需缓存）
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3, max_tokens=500)

    # 2. 初始化嵌入模型（会从本地缓存加载，若不存在则从镜像下载）
    print(" 加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder="./model_cache"   # 指定缓存目录，确保与手动下载位置一致
    )

    # 3. 定义本地缓存路径
    index_path = "faiss_index_deepseek"
    chunks_cache_path = "chunks_cache.pkl"

    # 4. 尝试加载本地缓存（向量库 + 文档块）
    if (os.path.exists(index_path) and
        os.path.exists(os.path.join(index_path, "index.faiss")) and
        os.path.exists(chunks_cache_path)):
        print(" 发现本地向量库和文档块缓存，正在加载...")
        # 加载文档块（用于 BM25）
        with open(chunks_cache_path, "rb") as f:
            chunks = pickle.load(f)
        print(f"  加载了 {len(chunks)} 个文档块。")
        # 加载 FAISS 向量库
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("  FAISS 向量库加载成功。")
    else:
        print(" 未找到缓存，重新构建向量库...")
        # 加载 PDF
        loader = DirectoryLoader(
            'knowledge_base/',
            glob='**/*.pdf',
            loader_cls=PyMuPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        print(f"  成功加载 {len(documents)} 页。")

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  切分为 {len(chunks)} 个文本块。")

        # 构建 FAISS
        vectorstore = FAISS.from_documents(chunks, embeddings)
        # 保存向量库
        vectorstore.save_local(index_path)
        print("  向量库已保存。")
        # 保存文档块（供下次加载 BM25 使用）
        with open(chunks_cache_path, "wb") as f:
            pickle.dump(chunks, f)
        print("  文档块缓存已保存。")

    # 5. 创建混合检索器（无论加载还是新建，都使用 chunks 和 vectorstore）
    print(" 创建混合检索器...")
    bm25_retriever = BM25Retriever.from_documents(chunks, k=50)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    retriever = ensemble_retriever

    # 6. 查询改写链与问答链（依赖 llm，每次都需要重新创建）
    rewrite_prompt = PromptTemplate.from_template(
        """你是一个查询重写助手。请将以下用户问题改写为3个不同角度、更具体、更易于检索的版本。
只输出改写后的3个问题，每行一个。

原始问题: {question}

改写结果:"""
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    prompt_template = PromptTemplate.from_template(
        """基于以下上下文回答问题。如果无法回答，请说明。

上下文：
{context}

问题：{input}

回答："""
    )
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    elapsed = time.time() - start_init
    print(f" RAG 系统初始化完成，耗时 {elapsed:.2f} 秒。")


# -------------------- 初始化（在模块加载时执行，仅一次） --------------------
init_rag()


# -------------------- FastAPI 生命周期管理 --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时无需再初始化（已提前完成）
    yield
    # 关闭时执行清理（如果有需要）


app = FastAPI(
    title="RAG 问答 API",
    description="基于 DeepSeek + 混合检索的文档问答服务",
    version="1.0",
    lifespan=lifespan,
)

# 允许跨域（便于前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------- 请求与响应模型 --------------------
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    elapsed_time: float

# -------------------- API 端点 --------------------
@app.get("/")
async def root():
    return {"message": "RAG API 已启动，请访问 /docs 查看接口文档"}

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后重试。")

    question = request.question
    start = time.time()

    loop = asyncio.get_event_loop()
    try:
        docs = await loop.run_in_executor(None, retrieve_documents, question)
        answer = await loop.run_in_executor(None, generate_answer, question, docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

    elapsed = time.time() - start
    return QueryResponse(answer=answer, elapsed_time=elapsed)


# -------------------- 核心检索函数 --------------------
def retrieve_documents(question: str):
    rewritten = rewrite_chain.invoke({"question": question})
    queries = [q.strip() for q in rewritten.split('\n') if q.strip()]
    queries.append(question)

    all_docs = []
    seen_content = set()
    for q in queries:
        docs = retriever.invoke(q)
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                all_docs.append(doc)
    return all_docs

def generate_answer(question: str, docs):
    return document_chain.invoke({"input": question, "context": docs})


# -------------------- 启动脚本 --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)