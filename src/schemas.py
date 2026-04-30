from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="最终返回给 LLM 和用户的条数")
    recall_top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="召回阶段候选条数（为空时走系统默认）",
    )


class Citation(BaseModel):
    chunk_id: str
    chapter_no: int
    chapter_title: str
    score: float
    content: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
