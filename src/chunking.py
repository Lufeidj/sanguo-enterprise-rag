import re
from dataclasses import dataclass


CHAPTER_PATTERN = re.compile(r"(第[一二三四五六七八九十百千0-9]+回[^\n]*)")


@dataclass
class Chunk:
    """
    单个文本切片的数据结构。

    字段说明：
    - chunk_id: 片段唯一 id（如 ch001_003）
    - chapter_no/chapter_title: 章节信息，便于溯源
    - content: 原始片段文本
    - summary: 片段摘要（用于展示/拼上下文）
    """

    chunk_id: str
    chapter_no: int
    chapter_title: str
    content: str
    summary: str


def split_by_chapter(text: str) -> list[tuple[int, str, str]]:
    """
    先按“第X回”切成大段（章节级）。

    返回结构：
    [(chapter_no, chapter_title, chapter_body), ...]
    """
    parts = CHAPTER_PATTERN.split(text)
    results: list[tuple[int, str, str]] = []
    current_chapter = 0

    # split 后数组形式通常是：前导文本, 标题1, 正文1, 标题2, 正文2, ...
    i = 1
    while i < len(parts):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        current_chapter += 1
        results.append((current_chapter, title, body))
        i += 2

    # 兜底：如果没有识别到章节标题，也至少保留一章，避免整本书丢失
    if not results and text.strip():
        results.append((1, "第1回 未识别标题", text.strip()))

    return results


def sliding_chunks(text: str, chunk_size: int = 800, overlap: int = 160) -> list[str]:
    """
    章节内再做滑动窗口切分。

    - chunk_size: 每段最大字符数
    - overlap: 相邻两段重叠字符数（用于减少上下文断裂）
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def make_summary(content: str, max_len: int = 80) -> str:
    # 教学版摘要策略：
    # - 优先取首句（到第一个句号）
    # - 超长就截断
    # 生产场景可替换为 LLM 摘要以提升语义质量
    first = content.split("。")[0].strip()
    if len(first) > max_len:
        return first[:max_len] + "..."
    return first or content[:max_len]


def build_chunks(text: str) -> list[Chunk]:
    """
    总切块入口：章节切分 + 滑窗细切 + 生成 chunk_id/summary。
    """
    all_chunks: list[Chunk] = []
    for chapter_no, title, body in split_by_chapter(text):
        smalls = sliding_chunks(body, chunk_size=800, overlap=160)
        for idx, c in enumerate(smalls, start=1):
            # 例：第 1 回第 3 个切片 -> ch001_003
            cid = f"ch{chapter_no:03d}_{idx:03d}"
            all_chunks.append(
                Chunk(
                    chunk_id=cid,
                    chapter_no=chapter_no,
                    chapter_title=title,
                    content=c,
                    summary=make_summary(c),
                )
            )
    return all_chunks
