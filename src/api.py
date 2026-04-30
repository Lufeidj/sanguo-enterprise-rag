from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import ask_question
from .schemas import AskRequest, AskResponse


# FastAPI 应用本体。
# title/version 会直接显示在 /docs 页面，方便演示和调试。
app = FastAPI(title="Sanguo Enterprise RAG", version="0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """
    前端入口页。

    打开根路径时，直接返回一个最小可用的问答页面，便于课堂演示。
    """
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """
    健康检查接口。

    用途：
    - 快速确认服务进程是否存活
    - 给运维/监控系统做心跳探测
    """
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """
    问答接口（RAG 主入口）。

    输入：用户问题 + top_k（检索条数）
    输出：答案文本 + 引用片段列表（citations）
    """
    # API 层只负责“接收请求并转发”。
    # 真正的检索与生成逻辑在 pipeline.ask_question() 中。
    return ask_question(req.question, top_k=req.top_k, recall_top_k=req.recall_top_k)
