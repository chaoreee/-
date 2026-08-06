"""BGE-M3 编码:同时生成 dense(1024维) + sparse(token_id→weight) 向量。

FlagEmbedding 的 BGEM3FlagModel.encode(return_dense=True, return_sparse=True)
返回 dense_vecs(np.ndarray) 与 lexical_weights(list[dict {token_id: weight}]),
后者直接映射到 Qdrant SparseVector(indices, values)。
"""
from __future__ import annotations

from typing import Tuple

from .config import EMBED_MODEL, USE_FP16, MAX_LENGTH, ENCODE_BATCH


def _resolve_model_path(model: str) -> str:
    """解析模型路径:本地路径优先;否则用 modelscope 缓存或下载。

    HuggingFace 在国内不稳定(huggingface_hub SSL/连接问题),
    改用 modelscope 镜像下载 BAAI/bge-m3。
    """
    from pathlib import Path
    if Path(model).exists():
        return model
    # modelscope 缓存:~/.cache/modelscope/models/{org--name}/snapshots/{rev}/
    cache = Path.home() / ".cache" / "modelscope" / "models" / model.replace("/", "--")
    snaps = cache / "snapshots"
    if snaps.exists():
        for d in sorted(snaps.iterdir(), reverse=True):
            if (d / "config.json").exists():
                return str(d)
    # 本地无缓存,用 modelscope 下载
    print(f"[Embedder] 本地无缓存,从 modelscope 下载 {model}...")
    from modelscope import snapshot_download
    return str(snapshot_download(model))


class Embedder:
    """BGE-M3 编码器,单例(模型加载一次,常驻 GPU)。"""

    _instance: "Embedder | None" = None

    def __init__(self, model: str = EMBED_MODEL, fp16: bool = USE_FP16):
        from FlagEmbedding import BGEM3FlagModel
        path = _resolve_model_path(model)
        print(f"[Embedder] 加载 {path}(fp16={fp16})...")
        self.model = BGEM3FlagModel(path, use_fp16=fp16)
        print("[Embedder] 加载完成")

    @classmethod
    def get(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(
        self, texts: list[str], batch_size: int = ENCODE_BATCH
    ) -> Tuple[list[list[float]], list[list[int]], list[list[float]]]:
        """批量编码。

        返回 (dense_vecs, sparse_indices, sparse_values):
          - dense_vecs: 每条 1024 维 float 列表
          - sparse_indices: 每条 token_id 列表(int)
          - sparse_values: 每条权重列表(float,与 indices 对应)
        """
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_arr = out["dense_vecs"]
        # np.ndarray → list[list[float]]
        dense = dense_arr.tolist() if hasattr(dense_arr, "tolist") else [
            d.tolist() if hasattr(d, "tolist") else list(d) for d in dense_arr
        ]
        sparse = out["lexical_weights"]  # list[dict]
        indices: list[list[int]] = []
        values: list[list[float]] = []
        for d in sparse:
            idx = [int(k) for k in d.keys()]
            val = [float(v) for v in d.values()]
            # BGE-M3 理论上任何文本都有 token;空时给占位避免 Qdrant 报错
            if not idx:
                idx, val = [0], [0.0]
            indices.append(idx)
            values.append(val)
        return dense, indices, values

    def encode_one(self, text: str) -> Tuple[list[float], list[int], list[float]]:
        dense, indices, values = self.encode([text])
        return dense[0], indices[0], values[0]
