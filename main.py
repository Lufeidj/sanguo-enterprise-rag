import uvicorn

from src.config import API_HOST, API_PORT


if __name__ == "__main__":
    # main.py 现在只负责“启动 API 服务”。
    # 向量库建表/导入等初始化动作，请改用 scripts/vdb_init.py。
    print("[API] API 启动中...")
    print(f"[API] 访问地址: http://{API_HOST}:{API_PORT}")
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=False)
