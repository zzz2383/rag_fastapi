import json
import random
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 加载文档并切块
loader = DirectoryLoader('../knowledge_base/', glob='**/*.pdf', loader_cls=PyMuPDFLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 随机抽取 20 个块（可根据需求调整）
sample_chunks = random.sample(chunks, min(20, len(chunks)))

# 初始化 LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

prompt = PromptTemplate.from_template(
    """根据以下文本片段，生成一个中文问题及其对应的答案。要求问题清晰、答案来自文本。
输出格式为 JSON：{{"question": "...", "answer": "..."}}
文本：{text}
JSON："""
)

qa_pairs = []
for chunk in sample_chunks:
    response = llm.invoke(prompt.format(text=chunk.page_content))
    try:
        # 提取 JSON 子串（防止多余内容）
        raw = response.content.strip()
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end != -1:
            qa = json.loads(raw[start:end])
            qa_pairs.append(qa)
    except:
        continue

# 保存为测试集
with open('test_qa.json', 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(f"生成 {len(qa_pairs)} 个问答对，已保存至 test_qa.json")