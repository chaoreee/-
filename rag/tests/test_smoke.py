"""冒烟测试:切分、索引、检索。

需先建索引(python -m rag.index)。test_chunking 不依赖索引。
从仓库根运行:pytest rag/tests/test_smoke.py
"""
import os
import sys

# 让 tests 能 import rag 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_chunking_nonempty():
    from rag.chunking import chunk_all
    chunks = chunk_all()
    assert len(chunks) > 1000, f"块数过少: {len(chunks)}"
    books = {c.book for c in chunks}
    assert books == {"伤寒论辑义", "千金方", "本草纲目", "温病条辨"}, f"书目不符: {books}"


def test_chunk_payload_has_text():
    from rag.chunking import chunk_all
    chunks = chunk_all()
    for c in chunks[:50]:
        p = c.payload()
        assert "text" in p and p["text"].strip()
        assert p["book"] in {"伤寒论辑义", "千金方", "本草纲目", "温病条辨"}
        assert p["is_translation"] is True


def test_index_exists():
    from qdrant_client import QdrantClient
    from rag.config import COLLECTION, QDRANT_PATH
    client = QdrantClient(path=QDRANT_PATH)
    assert client.collection_exists(COLLECTION), f"集合 {COLLECTION} 不存在,请先 python -m rag.index"
    info = client.get_collection(COLLECTION)
    assert info.points_count > 1000, f"点数过少: {info.points_count}"


def test_search_returns_results():
    from rag.retrieve import search
    results = search("桂枝汤", top_k=3)
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]


def test_search_filter():
    from rag.retrieve import search
    results = search("甘草", top_k=5, filters={"book": "本草纲目"})
    assert len(results) > 0
    for r in results:
        assert r["metadata"].get("book") == "本草纲目"
