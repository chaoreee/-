"""混合检索:dense + sparse prefetch → RRF 融合 + payload 过滤。

dense 抓语义、sparse 抓术语精准,RRF 融合二者的排名。
"""
from __future__ import annotations

from typing import Optional

from .config import COLLECTION, DENSE_LIMIT, DENSE_VECTOR, QDRANT_PATH, RRF_LIMIT, SPARSE_VECTOR
from .embed import Embedder


_client = None


def get_client():
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def _build_filter(filters: Optional[dict]):
    """{book:"本草纲目", yao:"甘草"} → Filter(must=[FieldCondition...])"""
    from qdrant_client import models
    if not filters:
        return None
    conditions = []
    for k, v in filters.items():
        if v is None:
            continue
        conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    return models.Filter(must=conditions) if conditions else None


def search(
    query: str,
    top_k: int = RRF_LIMIT,
    filters: Optional[dict] = None,
    dense_limit: int = DENSE_LIMIT,
) -> list[dict]:
    """混合检索。

    Args:
        query: 查询文本
        top_k: 最终返回数量
        filters: payload 过滤,如 {"book":"伤寒论辑义","fang":"桂枝汤方"}
        dense_limit: dense/sparse 各自预取的候选数(默认 50)
    Returns:
        [{"text","metadata","score"}, ...] 按 RRF 分降序
    """
    from qdrant_client import models

    embedder = Embedder.get()
    dense, indices, values = embedder.encode_one(query)

    client = get_client()
    filt = _build_filter(filters)
    result = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using=DENSE_VECTOR, limit=dense_limit, filter=filt),
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR, limit=dense_limit, filter=filt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=filt,
        limit=top_k,
        with_payload=True,
    )
    points = result.points
    return [
        {
            "text": p.payload.get("text", "") if p.payload else "",
            "metadata": {k: v for k, v in (p.payload or {}).items() if k != "text"} ,
            "score": float(p.score) if p.score is not None else 0.0,
        }
        for p in points
    ]


def search_dense_only(query: str, top_k: int = RRF_LIMIT, filters: Optional[dict] = None) -> list[dict]:
    """仅 dense 检索(用于对比验证 hybrid 效果)。"""
    from qdrant_client import models
    embedder = Embedder.get()
    dense, _, _ = embedder.encode_one(query)
    client = get_client()
    result = client.query_points(
        collection_name=COLLECTION,
        query=dense,
        using=DENSE_VECTOR,
        query_filter=_build_filter(filters),
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": p.payload.get("text", "") if p.payload else "",
            "metadata": {k: v for k, v in (p.payload or {}).items() if k != "text"},
            "score": float(p.score) if p.score is not None else 0.0,
        }
        for p in result.points
    ]


if __name__ == "__main__":
    import sys, json
    q = sys.argv[1] if len(sys.argv) > 1 else "桂枝汤主治什么"
    print(json.dumps(search(q), ensure_ascii=False, indent=2))
