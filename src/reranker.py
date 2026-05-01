from math import sqrt

from .config import RERANK_CANDIDATE_CAP
from .llm import embed_query, embed_texts


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_rerank(question: str, hits: list[dict], top_n: int) -> list[dict]:
    """
    语义重排（轻量版）：
    - 用 embedding 计算 query 与候选文档的余弦相似度
    - 失败时自动回退原始顺序
    """
    if not hits:
        return []

    # 只对融合分最高的前若干条做向量重排，避免候选过多时 embedding 调用爆炸。
    pool = sorted(
        hits,
        key=lambda x: float(x.get("hybrid_score", 0.0)),
        reverse=True,
    )[: max(RERANK_CANDIDATE_CAP, top_n)]

    docs: list[str] = []
    for h in pool:
        ent = h.get("entity", {})
        docs.append(
            f"{ent.get('chapter_title', '')}\n"
            f"{ent.get('summary', '')}\n"
            f"{ent.get('content', '')[:1200]}"
        )

    try:
        q_vec = embed_query(question)
        d_vecs = embed_texts(docs)
        scored: list[tuple[float, dict]] = []
        for h, dv in zip(pool, d_vecs):
            sem_score = _cosine(q_vec, dv)
            base = float(h.get("hybrid_score", 0.0))
            # 混合分：兼顾召回融合分与语义重排分
            final = 0.35 * base + 0.65 * sem_score
            h["rerank_score"] = final
            scored.append((final, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:top_n]]
    except Exception:
        return hits[:top_n]
