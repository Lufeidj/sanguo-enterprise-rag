import hashlib
import random

from openai import OpenAI

from .config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)


def mock_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    生成教学用“伪向量”。

    特点：
    - 同一文本每次都会得到同一向量（可复现）
    - 便于课堂演示，不依赖外部 embedding 服务
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量向量化入口。

    策略：
    1) 配置了 EMBEDDING_BASE_URL/EMBEDDING_API_KEY/EMBEDDING_MODEL -> 调真实 embedding
    2) 未完整配置或调用失败 -> 自动回退 mock embedding
    """
    if not texts:
        return []

    if not EMBEDDING_BASE_URL or not EMBEDDING_API_KEY or not EMBEDDING_MODEL:
        return [mock_embedding(t) for t in texts]

    try:
        client = OpenAI(api_key=EMBEDDING_API_KEY, base_url=EMBEDDING_BASE_URL)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vectors = [item.embedding for item in resp.data]

        # 防御性校验：若返回维度与配置不一致，回退 mock，避免入库/检索异常。
        if any(len(v) != EMBEDDING_DIM for v in vectors):
            return [mock_embedding(t) for t in texts]
        return vectors
    except Exception:
        return [mock_embedding(t) for t in texts]


def embed_query(text: str) -> list[float]:
    """
    查询向量化入口，与 embed_texts 共享同一策略。
    """
    return embed_texts([text])[0]


def generate_answer(question: str, contexts: list[str]) -> str:
    """
    根据检索上下文生成答案。

    两种模式：
    1) 未配置真实 LLM：返回教学模式拼接答案
    2) 已配置真实 LLM：调用 chat.completions 生成自然语言回答
    """
    def _fallback_text() -> str:
        joined = "\n".join(contexts[:3])
        return (
            "【教学模式回答】\n"
            f"问题：{question}\n\n"
            "根据检索片段，建议答案如下（可配置真实 LLM 获得更自然生成）：\n"
            f"{joined[:500]}"
        )

    if not LLM_API_KEY or not LLM_BASE_URL or not LLM_MODEL:
        # 教学回退模式：不用外部模型也能跑通“检索+回答”链路
        return _fallback_text()

    # 真实模型模式
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    prompt = (
        "你是三国知识库问答助手。请严格基于给定上下文回答，并给出简洁结论。\n\n"
        f"问题：{question}\n\n"
        "上下文：\n"
        + "\n\n".join(contexts[:5])
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        # 生产兜底：外部 LLM 网络/配额问题时，服务仍返回可解释结果。
        return _fallback_text()
