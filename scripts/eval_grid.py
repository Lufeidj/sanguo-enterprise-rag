from pathlib import Path
import argparse
import itertools
import json
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ask_question  # noqa: E402
import src.llm as llm_mod  # noqa: E402


DEFAULT_CASES = [
    {"question": "赤壁之战前孙刘联盟如何形成？", "must_keywords": ["孙", "刘", "联盟", "赤壁"]},
    {"question": "诸葛亮出山的关键背景是什么？", "must_keywords": ["诸葛亮", "刘备", "三顾", "出山"]},
    {"question": "桃园结义讲的是哪三个人？", "must_keywords": ["刘备", "关羽", "张飞"]},
    {"question": "官渡之战中曹操的对手是谁？", "must_keywords": ["曹操", "袁绍", "官渡"]},
    {"question": "空城计对应的主要人物是谁？", "must_keywords": ["诸葛亮", "司马懿", "空城计"]},
]


def keyword_hit_rate(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    hit = sum(1 for k in keywords if k in (text or ""))
    return hit / len(keywords)


def run_one_combo(
    top_k: int, recall_top_k: int, reranker_on: bool, cases: list[dict] | None = None
) -> dict:
    case_list = cases if cases is not None else DEFAULT_CASES
    answer_scores: list[float] = []
    citation_scores: list[float] = []
    latencies: list[float] = []
    details: list[dict] = []

    for case in case_list:
        q = case["question"]
        kws = case["must_keywords"]
        t0 = time.perf_counter()
        resp = ask_question(
            q,
            top_k=top_k,
            recall_top_k=recall_top_k,
            enable_reranker=reranker_on,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer_score = keyword_hit_rate(resp.answer, kws)
        citation_text = " ".join(c.content for c in resp.citations)
        citation_score = keyword_hit_rate(citation_text, kws)

        answer_scores.append(answer_score)
        citation_scores.append(citation_score)
        latencies.append(latency_ms)
        details.append(
            {
                "question": q,
                "answer_keyword_hit_rate": round(answer_score, 3),
                "citation_keyword_hit_rate": round(citation_score, 3),
                "latency_ms": round(latency_ms, 1),
                "citations_count": len(resp.citations),
            }
        )

    avg_ans = statistics.mean(answer_scores)
    avg_cit = statistics.mean(citation_scores)
    avg_lat = statistics.mean(latencies)

    # 综合分（可按项目需要调整权重）
    # 目标：优先回答质量，其次引用质量，再兼顾延迟
    latency_penalty = min(avg_lat / 3000.0, 1.0)  # >= 3s 记满惩罚
    composite = 0.6 * avg_ans + 0.3 * avg_cit + 0.1 * (1.0 - latency_penalty)

    return {
        "top_k": top_k,
        "recall_top_k": recall_top_k,
        "reranker_on": reranker_on,
        "case_count": len(case_list),
        "avg_answer_keyword_hit_rate": round(avg_ans, 3),
        "avg_citation_keyword_hit_rate": round(avg_cit, 3),
        "avg_latency_ms": round(avg_lat, 1),
        "composite_score": round(composite, 4),
        "details": details,
    }


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Grid eval over top_k / recall_top_k / reranker.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid (2x2x2) for smoke test; faster finish.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        metavar="N",
        help="Use only first N eval cases (default: all).",
    )
    args = parser.parse_args()

    # 评测阶段强制走本地回退答案，避免外部 LLM 网络波动影响评测稳定性与速度。
    llm_mod.LLM_API_KEY = ""
    llm_mod.LLM_BASE_URL = ""
    llm_mod.LLM_MODEL = ""

    if args.quick:
        top_k_grid = [3, 5]
        recall_top_k_grid = [20, 40]
        reranker_grid = [False, True]
    else:
        top_k_grid = [3, 5, 8]
        recall_top_k_grid = [20, 30, 50]
        reranker_grid = [False, True]

    cases = DEFAULT_CASES[: args.cases] if args.cases is not None else DEFAULT_CASES
    if not cases:
        print("[EVAL_GRID] No cases selected; exiting.", file=sys.stderr)
        sys.exit(1)

    total_combos = len(top_k_grid) * len(recall_top_k_grid) * len(reranker_grid)
    print(
        f"[EVAL_GRID] start: combos={total_combos}, cases={len(cases)}, quick={args.quick}",
        flush=True,
    )

    runs: list[dict] = []
    idx = 0
    for top_k, recall_top_k, reranker_on in itertools.product(
        top_k_grid, recall_top_k_grid, reranker_grid
    ):
        idx += 1
        print(
            f"[EVAL_GRID] ({idx}/{total_combos}) top_k={top_k} recall_top_k={recall_top_k} "
            f"reranker={reranker_on}",
            flush=True,
        )
        runs.append(run_one_combo(top_k, recall_top_k, reranker_on, cases))

    runs.sort(key=lambda x: x["composite_score"], reverse=True)
    best = runs[0] if runs else None
    report = {
        "options": {"quick": args.quick, "case_count": len(cases)},
        "grid": {
            "top_k": top_k_grid,
            "recall_top_k": recall_top_k_grid,
            "reranker_on": reranker_grid,
        },
        "best": best,
        "runs": runs,
    }

    out_file = PROJECT_ROOT / "data" / "eval_grid_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[EVAL_GRID] 评测完成")
    print(f"[EVAL_GRID] 组合数: {len(runs)}")
    if best:
        print(
            f"[EVAL_GRID] 最优组合: top_k={best['top_k']}, "
            f"recall_top_k={best['recall_top_k']}, reranker_on={best['reranker_on']}"
        )
        print(
            f"[EVAL_GRID] 指标: ans={best['avg_answer_keyword_hit_rate']}, "
            f"cit={best['avg_citation_keyword_hit_rate']}, "
            f"lat={best['avg_latency_ms']}ms, score={best['composite_score']}"
        )
    print(f"[EVAL_GRID] 报告文件: {out_file}")


if __name__ == "__main__":
    main()
