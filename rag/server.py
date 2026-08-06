"""FastAPI 服务:/search、/health、/schema。

启动(从仓库根目录):
    python -m rag.server
    # 或: uvicorn rag.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from . import config, retrieve
from .config import API_HOST, API_PORT, COLLECTION


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: Optional[dict] = None


app = FastAPI(title="中医古籍 RAG", description="BGE-M3 + Qdrant 混合检索")


@app.on_event("startup")
def _preload():
    """启动时预加载 BGE-M3,避免首次请求慢。"""
    from .embed import Embedder
    Embedder.get()


@app.get("/health")
def health():
    try:
        client = retrieve.get_client()
        info = client.get_collection(COLLECTION)
        return {
            "status": "ok",
            "collection": COLLECTION,
            "points_count": info.points_count,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/search")
def search(req: SearchRequest):
    results = retrieve.search(req.query, top_k=req.top_k, filters=req.filters)
    return {"query": req.query, "top_k": req.top_k, "count": len(results), "results": results}


@app.get("/schema")
def schema():
    """返回可用过滤字段与书目,便于 Agent 构造 filters。"""
    return {
        "filter_fields": [
            "book", "is_translation", "juan", "pian", "bu",
            "yao", "fang", "binglei", "tiaowen_no",
            "fangfang_no", "fangfang_name", "xijie",
        ],
        "books": list(config.TRANSLATION_FILES.keys()),
        "example": {"query": "桂枝汤", "filters": {"book": "伤寒论辑义"}},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag.server:app", host=API_HOST, port=API_PORT)
