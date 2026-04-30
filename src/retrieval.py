from dataclasses import dataclass
from math import log
import re

from .vector_store import SanguoVectorStore


@dataclass
class RetrievalDoc:
    chunk_id: str
    chapter_no: int
    chapter_title: str
    summary: str
    content: str


def _tokenize(text: str) -> list[str]:
    """
    轻量中文+英文 token 化：
    - 中文：按连续中文串生成 bi-gram + 原串
    - 英文数字：按单词切分
    """
    text = (text or "").lower()
    tokens: list[str] = []

    for m in re.finditer(r"[\u4e00-\u9fff]+", text):
        seg = m.group(0)
        if len(seg) >= 2:
            for i in range(len(seg) - 1):
                tokens.append(seg[i : i + 2])
        tokens.append(seg)

    for w in re.findall(r"[a-z0-9]+", text):
        tokens.append(w)
    return tokens


class BM25Retriever:
    """
    轻量 BM25 检索器（内存索引版）。
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._indexed_row_count = -1
        self._docs: list[RetrievalDoc] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._avg_dl = 0.0

    def _rebuild(self, store: SanguoVectorStore) -> None:
        rows = store.fetch_all_chunks()
        docs: list[RetrievalDoc] = []
        doc_tokens: list[list[str]] = []
        doc_tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        dl_sum = 0

        for r in rows:
            doc = RetrievalDoc(
                chunk_id=r.get("id", ""),
                chapter_no=int(r.get("chapter_no", 0)),
                chapter_title=r.get("chapter_title", ""),
                summary=r.get("summary", ""),
                content=r.get("content", ""),
            )
            docs.append(doc)
            text = f"{doc.chapter_title}\n{doc.summary}\n{doc.content}"
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            dl_sum += len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            doc_tf.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1

        n = len(docs)
        self._docs = docs
        self._doc_tokens = doc_tokens
        self._doc_tf = doc_tf
        self._df = df
        self._avg_dl = (dl_sum / n) if n else 0.0
        self._indexed_row_count = n

    def _ensure_index(self, store: SanguoVectorStore) -> None:
        current = store.row_count()
        if current != self._indexed_row_count:
            self._rebuild(store)

    def search(self, store: SanguoVectorStore, query: str, top_k: int) -> list[RetrievalDoc]:
        self._ensure_index(store)
        if not self._docs:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        n = len(self._docs)
        scores: list[tuple[float, int]] = []
        for i, tf in enumerate(self._doc_tf):
            dl = len(self._doc_tokens[i]) or 1
            score = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                df = self._df.get(t, 0)
                idf = log((n - df + 0.5) / (df + 0.5) + 1.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_dl or 1.0))
                score += idf * ((f * (self.k1 + 1)) / denom)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        chosen = [self._docs[i] for s, i in scores[:top_k] if s > 0]
        return chosen
