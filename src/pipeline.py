from pathlib import Path

from .chunking import build_chunks
from .config import DEFAULT_SANGUO_PATH, KEYWORD_RRF_K, RECALL_TOP_K, VECTOR_RRF_K
from .llm import embed_query, embed_texts, generate_answer
from .retrieval import BM25Retriever
from .schemas import AskResponse, Citation
from .vector_store import SanguoVectorStore


def ingest_sanguo_file(file_path: Path) -> int:
    """
    把原始文本文件导入向量库。

    完整流程：
    1) 读取文本
    2) 切块（chapter + sliding window）
    3) 生成向量（教学版默认 mock embedding）
    4) 组织为 Milvus 行数据并写入

    返回值：本次成功写入的 chunk 条数
    """
    # 读取 UTF-8 文本（《三国演义》）
    text = file_path.read_text(encoding="utf-8")

    # 文本切块：把长文拆成便于检索的小片段
    chunks = build_chunks(text)
    if not chunks:
        return 0

    # 初始化向量库客户端，确保集合存在
    store = SanguoVectorStore()
    store.ensure_collection()

    # 对每个 chunk 做向量化，得到 list[list[float]]
    vectors = embed_texts([c.content for c in chunks])

    # 组装成 Milvus 插入数据结构（每个元素是一行）
    rows: list[dict] = []
    for c, v in zip(chunks, vectors):
        rows.append(
            {
                "id": c.chunk_id,
                "chapter_no": c.chapter_no,
                "chapter_title": c.chapter_title,
                "summary": c.summary,
                "content": c.content,
                "vector": v,
            }
        )

    # 批量写入 Milvus
    store.upsert(rows)
    return len(rows)


_BM25 = BM25Retriever()


def _hybrid_rerank(vector_hits: list[dict], bm25_hits: list[dict]) -> list[dict]:
    """
    混合重排（生产常见两段式中的第二段）：
    - 第一信号：向量召回 rank
    - 第二信号：BM25 召回 rank
    - 融合方式：RRF（Reciprocal Rank Fusion）
    """
    if not vector_hits and not bm25_hits:
        return []

    vector_rank_map: dict[str, int] = {}
    merged_by_id: dict[str, dict] = {}

    for idx, h in enumerate(vector_hits):
        ent = h.get("entity", {})
        cid = ent.get("id", f"row-{idx}")
        vector_rank_map[cid] = idx + 1
        merged_by_id[cid] = h

    keyword_rank_map: dict[str, int] = {}
    for idx, h in enumerate(bm25_hits):
        ent = h.get("entity", {})
        cid = ent.get("id", f"kw-{idx}")
        keyword_rank_map[cid] = idx + 1
        if cid not in merged_by_id:
            merged_by_id[cid] = h

    # RRF 融合并回写 score（给 citations 展示）
    fused: list[tuple[float, dict]] = []
    for cid, h in merged_by_id.items():
        ent = h.get("entity", {})
        if "id" not in ent:
            ent["id"] = cid
        vr = vector_rank_map.get(cid, len(merged_by_id))
        kr = keyword_rank_map.get(cid, len(merged_by_id))
        fused_score = 1.0 / (VECTOR_RRF_K + vr) + 1.0 / (KEYWORD_RRF_K + kr)
        h["hybrid_score"] = fused_score
        fused.append((fused_score, h))

    fused.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in fused]


def ask_question(question: str, top_k: int = 5, recall_top_k: int | None = None) -> AskResponse:
    """
    RAG 问答主流程。

    核心步骤：
    1) 将问题转为向量
    2) 在 Milvus 做相似度检索（top_k）
    3) 把检索结果组织成上下文与引用
    4) 调用 LLM（或教学回退逻辑）生成答案
    """
    store = SanguoVectorStore()

    # 问题向量化：优先真实 embedding，失败自动回退 mock
    q_vec = embed_query(question)

    # 第一段：大召回（Recall）
    # - 向量召回 + BM25 召回并行
    recall_k = max(top_k, recall_top_k or RECALL_TOP_K)
    vector_hits = store.search(q_vec, top_k=recall_k)
    bm25_docs = _BM25.search(store, query=question, top_k=recall_k)
    bm25_hits = [
        {
            "entity": {
                "id": d.chunk_id,
                "chapter_no": d.chapter_no,
                "chapter_title": d.chapter_title,
                "summary": d.summary,
                "content": d.content,
            }
        }
        for d in bm25_docs
    ]

    # 第二段：RRF 融合后截断为最终 top_k
    reranked_hits = _hybrid_rerank(vector_hits, bm25_hits)[:top_k]

    # contexts 用于喂给大模型生成答案
    # citations 用于前端/调用方展示“答案来源”
    contexts: list[str] = []
    citations: list[Citation] = []
    for h in reranked_hits:
        ent = h.get("entity", {})
        content = ent.get("content", "")
        contexts.append(
            f"[{ent.get('id')}] 第{ent.get('chapter_no')}回 {ent.get('chapter_title')}\n"
            f"摘要：{ent.get('summary')}\n"
            f"内容：{content[:500]}"
        )
        citations.append(
            Citation(
                chunk_id=ent.get("id", ""),
                chapter_no=int(ent.get("chapter_no", 0)),
                chapter_title=ent.get("chapter_title", ""),
                score=float(h.get("hybrid_score", h.get("distance", 0.0))),
                content=content[:240],
            )
        )

    # 生成最终回答：若未配置真实 LLM，会走教学模式回退文本
    answer = generate_answer(question, contexts)
    return AskResponse(answer=answer, citations=citations)


def bootstrap_data(auto_ingest: bool = True) -> tuple[bool, int, str]:
    """
    确保集合可用；如为空且 auto_ingest=True，则自动导入《三国演义》。
    返回: (是否导入, 当前行数, 提示信息)
    """
    store = SanguoVectorStore()

    # 1) 集合不存在就创建（幂等：已存在时不会重复创建）
    store.ensure_collection()

    # 2) 查询当前已有行数
    count = store.row_count()
    if count > 0:
        return False, count, f"集合已有数据: {count} 条"

    # 3) 集合为空且不允许自动导入，则直接返回提示
    if not auto_ingest:
        return False, 0, "集合为空，且未启用自动导入"

    # 4) 检查默认文本路径是否存在
    if not DEFAULT_SANGUO_PATH.exists():
        return False, 0, f"未找到文本: {DEFAULT_SANGUO_PATH}"

    # 5) 真正执行导入
    inserted = ingest_sanguo_file(DEFAULT_SANGUO_PATH)
    return True, inserted, f"已自动导入: {inserted} 条"
