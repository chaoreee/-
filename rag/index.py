"""索引脚本:切分 4 书译文 → BGE-M3 编码 → 批量入 Qdrant。

用法(从仓库根目录):
    python -m rag.index
"""
from __future__ import annotations

import json

from .config import (
    CHUNKS_CACHE, COLLECTION, DENSE_DIM, DENSE_VECTOR, QDRANT_PATH,
    SPARSE_VECTOR, ENCODE_BATCH,
)
from .chunking import chunk_all
from .embed import Embedder


def get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(path=QDRANT_PATH)


def build_index(rebuild: bool = True) -> int:
    from qdrant_client import models

    client = get_client()

    # 1. 建集合
    if rebuild and client.collection_exists(COLLECTION):
        print(f"[索引] 删除旧集合 {COLLECTION}")
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=DENSE_DIM,
                                                  distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(),
            },
        )
        print(f"[索引] 创建集合 {COLLECTION}(dense={DENSE_DIM}d + sparse)")

    # 2. 切分
    print("[索引] 切分译文...")
    chunks = chunk_all()
    print(f"[索引] 共 {len(chunks)} 块")
    if not chunks:
        raise RuntimeError("切分为空,中止")

    # 3. 编码
    embedder = Embedder.get()
    texts = [c.text for c in chunks]
    print(f"[索引] 编码 {len(texts)} 段(batch_size={ENCODE_BATCH})...")
    dense, indices, values = embedder.encode(texts)
    print("[索引] 编码完成")

    # 4. 构造点
    points = []
    for i, c in enumerate(chunks):
        points.append(models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR: dense[i],
                SPARSE_VECTOR: models.SparseVector(indices=indices[i], values=values[i]),
            },
            payload=c.payload(),
        ))

    # 5. 批量 upsert
    BATCH = 256
    for i in range(0, len(points), BATCH):
        client.upsert(COLLECTION, points=points[i:i + BATCH])
        done = min(i + BATCH, len(points))
        if done % 1024 == 0 or done == len(points):
            print(f"  upsert {done}/{len(points)}")

    # 6. 缓存切分结果
    CHUNKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.payload(), ensure_ascii=False) + "\n")

    print(f"[索引] 完成:{len(points)} 点入 {COLLECTION},缓存 → {CHUNKS_CACHE}")
    return len(points)


if __name__ == "__main__":
    build_index()
