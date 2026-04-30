from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUTO_INGEST_ON_STARTUP, DEFAULT_SANGUO_PATH  # noqa: E402
from src.pipeline import ingest_sanguo_file  # noqa: E402
from src.vector_store import SanguoVectorStore  # noqa: E402


def ensure_collection() -> tuple[int, str]:
    """
    只做向量库初始化（幂等）：
    - 若集合不存在则创建
    - 返回当前集合行数
    """
    store = SanguoVectorStore()
    store.ensure_collection()
    count = store.row_count()
    return count, f"集合已就绪，当前行数: {count}"


def bootstrap_data(auto_ingest: bool) -> tuple[bool, int, str]:
    """
    启动前数据准备：
    - 集合存在 -> 返回当前行数
    - 集合为空且允许自动导入 -> 执行导入
    """
    store = SanguoVectorStore()
    store.ensure_collection()
    count = store.row_count()
    if count > 0:
        return False, count, f"集合已有数据: {count} 条"

    if not auto_ingest:
        return False, 0, "集合为空，且未启用自动导入"

    if not DEFAULT_SANGUO_PATH.exists():
        return False, 0, f"未找到文本: {DEFAULT_SANGUO_PATH}"

    inserted = ingest_sanguo_file(DEFAULT_SANGUO_PATH)
    return True, inserted, f"已自动导入: {inserted} 条"


def main() -> None:
    parser = argparse.ArgumentParser(description="初始化 Milvus 集合与数据")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="若集合为空，自动导入默认《三国演义》文本",
    )
    parser.add_argument(
        "--force-ingest",
        action="store_true",
        help="忽略集合是否有数据，强制再次导入（会追加数据）",
    )
    args = parser.parse_args()

    prefix = "[VDB_INIT]"
    if args.force_ingest:
        if not DEFAULT_SANGUO_PATH.exists():
            raise FileNotFoundError(f"未找到文本: {DEFAULT_SANGUO_PATH}")
        inserted = ingest_sanguo_file(DEFAULT_SANGUO_PATH)
        print(f"{prefix} 强制导入完成，写入 chunk 数量: {inserted}")
        return

    if args.ingest:
        imported, count, msg = bootstrap_data(auto_ingest=True)
        print(f"{prefix} {msg}")
        if imported:
            print(f"{prefix} 当前写入行数: {count}")
        return

    # 默认只初始化集合，不导入
    count, msg = ensure_collection()
    print(f"{prefix} {msg}")
    print(
        f"{prefix} 如需导入数据，请执行: "
        f"python scripts/vdb_init.py --ingest "
        f"(AUTO_INGEST_ON_STARTUP={AUTO_INGEST_ON_STARTUP})"
    )


if __name__ == "__main__":
    main()
