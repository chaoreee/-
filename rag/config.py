"""全局配置:路径、模型、向量库、切分参数。"""
from pathlib import Path

# === 路径 ===
RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parent
DATA_DIR = RAG_DIR / "data"
QDRANT_PATH = str(DATA_DIR / "qdrant")          # Qdrant 本地持久化目录
CHUNKS_CACHE = DATA_DIR / "chunks.jsonl"         # 切分结果缓存

# === 源数据(仅译文)===
TRANSLATION_FILES: dict[str, Path] = {
    "伤寒论辑义": REPO_ROOT / "伤寒论辑义_今译.md",
    "千金方":     REPO_ROOT / "千金方_今译.md",
    "本草纲目":   REPO_ROOT / "本草纲目_今译.md",
    "温病条辨":   REPO_ROOT / "温病条辨_今译.md",
}

# === Embedding (BGE-M3) ===
EMBED_MODEL = "BAAI/bge-m3"
DENSE_DIM = 1024                # BGE-M3 dense 维度
USE_FP16 = True                 # GPU 推理用 fp16 提速
MAX_LENGTH = 8192               # BGE-M3 最大 token
ENCODE_BATCH = 16               # 编码批大小(按显存调整,4060 8GB 建议 12-16)

# === Qdrant ===
COLLECTION = "tcm_modern"
DENSE_VECTOR = "dense"          # 命名稠密向量字段
SPARSE_VECTOR = "sparse"        # 命名稀疏向量字段
DENSE_LIMIT = 50                # 检索时 dense/sparse 各取的候选数
RRF_LIMIT = 8                   # 默认返回 top_k

# === 切分 ===
MAX_CHUNK_CHARS = 800           # 单块超过此长度按句号二次切分

# === FastAPI ===
API_HOST = "0.0.0.0"
API_PORT = 8000
