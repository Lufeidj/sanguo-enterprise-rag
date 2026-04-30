from pathlib import Path
import json
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ask_question  # noqa: E402


DEFAULT_CASES = [
    {
        "question": "赤壁之战前孙刘联盟如何形成？",
        "must_keywords": ["孙", "刘", "联盟", "赤壁"],
    },
    {
        "question": "诸葛亮出山的关键背景是什么？",
        "must_keywords": ["诸葛亮", "刘备", "三顾", "出山"],
    },
    {
        "question": "桃园结义讲的是哪三个人？",
        "must_keywords": ["刘备", "关羽", "张飞"],
    },
    {
        "question": "官渡之战中曹操的对手是谁？",
        "must_keywords": ["曹操", "袁绍", "官渡"],
    },
    {
        "question": "空城计对应的主要人物是谁？",
        "must_keywords": ["诸葛亮", "司马懿", "空城计"],
    },
]


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k in text)
    return hit / len(keywords)


def run_eval(top_k: int = 5) -> dict:
    rows: list[dict] = []
    answer_scores: list[float] = []
    citation_scores: list[float] = []

    for case in DEFAULT_CASES:
        q = case["question"]
        kws = case["must_keywords"]
        resp = ask_question(q, top_k=top_k)

        answer_score = keyword_hit_rate(resp.answer, kws)
        citation_text = " ".join(c.content for c in resp.citations)
        citation_score = keyword_hit_rate(citation_text, kws)

        answer_scores.append(answer_score)
        citation_scores.append(citation_score)
        rows.append(
            {
                "question": q,
                "answer_keyword_hit_rate": round(answer_score, 3),
                "citation_keyword_hit_rate": round(citation_score, 3),
                "citations_count": len(resp.citations),
            }
        )

    return {
        "top_k": top_k,
        "case_count": len(DEFAULT_CASES),
        "avg_answer_keyword_hit_rate": round(statistics.mean(answer_scores), 3),
        "avg_citation_keyword_hit_rate": round(statistics.mean(citation_scores), 3),
        "details": rows,
    }


def main() -> None:
    top_k = 5
    result = run_eval(top_k=top_k)
    out_file = PROJECT_ROOT / "data" / "eval_basic_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[EVAL] 评测完成")
    print(f"[EVAL] top_k={result['top_k']}, case_count={result['case_count']}")
    print(f"[EVAL] 平均答案关键词命中率: {result['avg_answer_keyword_hit_rate']}")
    print(f"[EVAL] 平均引用关键词命中率: {result['avg_citation_keyword_hit_rate']}")
    print(f"[EVAL] 报告文件: {out_file}")


if __name__ == "__main__":
    main()
