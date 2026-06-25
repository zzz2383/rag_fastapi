import json
import time
import requests
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

# ---------- 配置 ----------
API_URL = "http://localhost:8000/ask"
TEST_JSON = "test_qa.json"  # 你的测试数据集
OUTPUT_REPORT = "test_report.json"

# 加载语义相似度模型（轻量级中文模型）
sim_model = SentenceTransformer('/models/paraphrase-multilingual-MiniLM-L12-v2')

# 初始化 ROUGE 评估器（可选）
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ---------- 加载测试集 ----------
with open(TEST_JSON, 'r', encoding='utf-8') as f:
    test_data = json.load(f)


# ---------- 核心评估函数 ----------
def evaluate_answers(candidates, references):
    """
    计算语义相似度（余弦）和 ROUGE-L
    candidates: 列表，RAG生成的答案
    references: 列表，标准答案
    """
    # 编码为向量
    cand_emb = sim_model.encode(candidates)
    ref_emb = sim_model.encode(references)
    # 计算余弦相似度
    sim = np.diag(np.dot(cand_emb, ref_emb.T))  # 对应位置的余弦相似度
    # 也可使用余弦相似度函数，但这里因为已经归一化（默认），直接点积即可
    # 实际上 SentenceTransformer 默认为归一化向量，点积即为余弦相似度
    # 但为了保险，使用 cosine_similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_scores = cosine_similarity(cand_emb, ref_emb).diagonal()

    # 计算 ROUGE-L（每个样本的 F1）
    rouge_l_scores = []
    for cand, ref in zip(candidates, references):
        scores = rouge_scorer.score(ref, cand)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    return sim_scores, rouge_l_scores


# ---------- 执行测试 ----------
results = []
total_elapsed = 0.0
failed = 0

print(f"开始测试，共 {len(test_data)} 个问题。")
for item in tqdm(test_data):
    question = item['question']
    ref_answer = item['answer']

    start = time.time()
    try:
        resp = requests.post(API_URL, json={"question": question}, timeout=30)
        elapsed = time.time() - start
        total_elapsed += elapsed
        if resp.status_code == 200:
            data = resp.json()
            gen_answer = data['answer']
            server_time = data.get('elapsed_time', 0)
        else:
            gen_answer = f"ERROR: {resp.status_code}"
            failed += 1
    except Exception as e:
        gen_answer = f"EXCEPTION: {str(e)}"
        elapsed = time.time() - start
        failed += 1

    results.append({
        "question": question,
        "reference": ref_answer,
        "generated": gen_answer,
        "elapsed": elapsed,
        "status": "success" if "ERROR" not in gen_answer and "EXCEPTION" not in gen_answer else "failed"
    })

# 提取生成成功的样本（用于指标计算）
valid_results = [r for r in results if r['status'] == 'success']
if valid_results:
    gen_answers = [r['generated'] for r in valid_results]
    ref_answers = [r['reference'] for r in valid_results]
    sim_scores, rouge_l_scores = evaluate_answers(gen_answers, ref_answers)
    # 将得分写回 results
    for i, r in enumerate(valid_results):
        r['semantic_similarity'] = float(sim_scores[i])
        r['rougeL_f1'] = float(rouge_l_scores[i])

# ---------- 汇总统计 ----------
num_total = len(results)
num_success = len(valid_results)
avg_time = total_elapsed / num_total if num_total else 0
avg_sim = np.mean([r['semantic_similarity'] for r in valid_results]) if valid_results else 0
avg_rouge = np.mean([r['rougeL_f1'] for r in valid_results]) if valid_results else 0

report = {
    "total_questions": num_total,
    "successful": num_success,
    "failed": num_total - num_success,
    "average_elapsed_seconds": round(avg_time, 3),
    "average_semantic_similarity": round(avg_sim, 4),
    "average_rougeL_f1": round(avg_rouge, 4),
    "details": results
}

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n测试完成。报告已保存至 {OUTPUT_REPORT}")
print(f"成功率: {num_success}/{num_total}")
print(f"平均耗时: {avg_time:.3f}s")
print(f"平均语义相似度: {avg_sim:.4f}")
print(f"平均 ROUGE-L F1: {avg_rouge:.4f}")