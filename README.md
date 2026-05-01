# Sanguo Enterprise RAG

企业级教学版《三国演义》RAG 项目（Milvus + FastAPI）。

**小白向全流程说明**（环境 → 入库 → 检索 → 生成）：见 [课上/理解/端到端流程详解.md](课上/理解/端到端流程详解.md)。

## 功能
- 章节感知切块（优先按“第X回”切分）
- 入库到 Milvus（含章节、摘要、来源元数据）
- 混合召回问答（向量 + BM25 融合 + 语义重排，返回引用片段）

## 快速开始
1. 安装依赖
   - `pip install -r requirements.txt`
2. 配置环境变量
   - 复制 `.env.example` 为 `.env`
   - 如需启用真实 embedding，配置：
     - `EMBEDDING_BASE_URL`
     - `EMBEDDING_API_KEY`
     - `EMBEDDING_MODEL`
   - 推荐将 `EMBEDDING_DIM` 设为与模型返回一致（例如百炼 `text-embedding-v4` 默认 1024）
3. 初始化向量库（推荐先执行）
   - 只建表：`python scripts/vdb_init.py`
   - 建表并在集合为空时自动导入：`python scripts/vdb_init.py --ingest`
4. 启动 API 服务
   - `python main.py`
5. 打开前端演示页
   - `http://127.0.0.1:8008/`

可选：手动导入
- `python scripts/ingest_sanguo.py`
- `python scripts/eval_basic.py`（生成基础评测报告）
- `python scripts/eval_grid.py`（网格评测：top_k / recall_top_k / reranker）

## 接口
- `GET /health`
- `POST /ask`
  - body: `{"question":"赤壁之战前联盟如何形成？","top_k":5,"recall_top_k":30}`
  - 说明：
    - `recall_top_k`：召回候选规模（用于重排前）
    - `top_k`：重排后最终用于回答和展示的条数

## 说明
- 默认读取 Day08 项目中的《三国演义》文本。
- 你可以在 Attu 的 `default` 数据库里看到 `sanguo_chunks` 集合与实体数增长。
- 若从 mock embedding 切换到真实 embedding，建议清空并重建集合后重新导入，以保证向量空间一致。
