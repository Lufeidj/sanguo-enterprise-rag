from pymilvus import DataType, MilvusClient

from .config import EMBEDDING_DIM, MILVUS_COLLECTION, MILVUS_DB_NAME, MILVUS_URI


class SanguoVectorStore:
    """
    Milvus 访问层（轻量封装）。

    目的：
    - 把“建集合、写数据、查数据”的数据库细节集中管理
    - 让 pipeline 层只关注业务流程，不关心底层 API 细节
    """

    def __init__(self) -> None:
        # 注意：这里连接的是“已有数据库名”（默认 default），
        # 不是自动新建数据库。
        self.client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB_NAME, timeout=10)
        self.collection = MILVUS_COLLECTION

    def ensure_collection(self) -> None:
        """
        确保目标集合存在，不存在则按约定 schema 创建。
        """
        if self.client.has_collection(self.collection):
            return

        # auto_id=False 表示主键 id 由我们自己提供（例如 ch001_001）
        # enable_dynamic_field=True 会开启 $meta 动态字段
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("chapter_no", DataType.INT64)
        schema.add_field("chapter_title", DataType.VARCHAR, max_length=512)
        schema.add_field("summary", DataType.VARCHAR, max_length=4096)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

        # 先建集合，再建索引（向量检索性能关键）
        self.client.create_collection(collection_name=self.collection, schema=schema)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_idx",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_index(self.collection, index_params=index_params)

    def upsert(self, rows: list[dict]) -> None:
        """
        写入向量与文本数据。

        教学项目里用 insert 即可，若主键重复时行为由 Milvus 配置决定。
        """
        self.client.insert(collection_name=self.collection, data=rows)
        # 显式 flush，确保统计行数与后续检索能及时看到最新写入数据。
        self.client.flush(self.collection)

    def row_count(self) -> int:
        """
        获取集合当前实体行数。
        """
        if not self.client.has_collection(self.collection):
            return 0
        stats = self.client.get_collection_stats(self.collection)
        return int(stats.get("row_count", 0))

    def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
        """
        向量检索入口。

        参数：
        - vector: 查询向量（通常来自用户问题）
        - top_k: 返回最相似的前 k 条结果
        """
        # 检索前加载集合到内存，避免首次查询失败
        self.client.load_collection(self.collection)
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            anns_field="vector",
            output_fields=["id", "chapter_no", "chapter_title", "summary", "content"],
            limit=top_k,
            search_params={"params": {"ef": 64}},
        )
        return results[0]

    def fetch_all_chunks(self) -> list[dict]:
        """
        拉取集合中的全部文本块（用于关键词/BM25 召回）。
        """
        total = self.row_count()
        if total <= 0:
            return []
        return self.client.query(
            collection_name=self.collection,
            filter='id > ""',
            output_fields=["id", "chapter_no", "chapter_title", "summary", "content"],
            limit=total,
        )
