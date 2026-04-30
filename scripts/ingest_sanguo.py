from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SANGUO_PATH  # noqa: E402
from src.pipeline import ingest_sanguo_file  # noqa: E402


def main() -> None:
    file_path = DEFAULT_SANGUO_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"未找到《三国演义》文本：{file_path}")

    total = ingest_sanguo_file(file_path)
    print(f"[OK] 入库完成，写入 chunk 数量：{total}")


if __name__ == "__main__":
    main()
