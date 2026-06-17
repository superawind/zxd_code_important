import numpy as np
from math import log

# 假设简单的语料库
corpus = [
    "大 模型 训练 基础设施",
    "算力 集群 网络 优化",
    "模型 推理 性能 测试"
]
tokenized_corpus = [doc.split() for doc in corpus]

# 计算全局参数
avgdl = sum(len(d) for d in tokenized_corpus) / len(tokenized_corpus)
# 简化版 IDF 计算
def get_idf(word, corpus):
    n_containing = sum(1 for d in corpus if word in d)
    return log((len(corpus) - n_containing + 0.5) / (n_containing + 0.5) + 1)

# 给定一段新文本
text = "模型 性能 优化"
tokens = text.split()
doc_len = len(tokens)

# 计算向量 (以原语料库词表为准)
vocab = list(set([word for d in tokenized_corpus for word in d]))
bm25_vector = []

k1, b = 1.5, 0.75

for word in vocab:
    if word in tokens:
        tf = tokens.count(word)
        idf = get_idf(word, tokenized_corpus)
        # BM25 公式计算该维度的权重
        score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avgdl))
        bm25_vector.append(score)
    else:
        bm25_vector.append(0.0)

print(f"BM25 稀疏向量: {bm25_vector}")