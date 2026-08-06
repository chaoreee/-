# 中医古籍 RAG 检索后端

把 4 部中医古籍的现代文译文灌入 Qdrant 向量库,用 BGE-M3 的 **dense + sparse 混合检索 + RRF 融合** 召回相关段落,通过 FastAPI 对外提供检索服务,供中医 Agent(商业 LLM)调用。

## 技术栈

- **Embedding**:BGE-M3(FlagEmbedding,本地 GPU,fp16)——同时输出 dense(1024维)与 sparse(词项权重),一套模型满足混合检索
- **向量库**:Qdrant 本地模式(`QdrantClient(path=...)`,无需 Docker,持久化到磁盘)
- **检索**:dense(语义)+ sparse(术语精准)prefetch → RRF 融合 + payload 过滤
- **接口**:FastAPI HTTP 服务

## 安装

需 Python 3.10+(用 Anaconda 3.11)、NVIDIA GPU、CUDA。

```bash
# 1. 创建环境
conda create -n tcm-rag python=3.11 -y
conda activate tcm-rag

# 2. 装 torch(按 CUDA 版本,例如 CUDA 12.1)
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
#   或 pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 装其余依赖
uv pip install -e rag
#   或 pip install -e rag

# 4. 首次运行会自动从 modelscope 下载 BAAI/bge-m3(~2GB,国内镜像)
#    HuggingFace 在国内不稳定,embed.py 已改用 modelscope 镜像下载
```

## 索引

```bash
# 从仓库根目录运行
python -m rag.index
```
- 切分 4 书译文 → BGE-M3 编码 → 入 Qdrant
- 预计 2-3 万块,首次编码几分钟(看 GPU)
- 切分结果缓存到 `rag/data/chunks.jsonl`

## 启动服务

```bash
python -m rag.server
# 或: uvicorn rag.server:app --host 0.0.0.0 --port 8000
```

## 调用

```bash
# 检索
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "桂枝汤主治什么", "top_k": 5}'

# 带过滤(按书/药/方)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "甘草 口舌生疮", "top_k": 5, "filters": {"book": "本草纲目"}}'

# 健康检查
curl http://localhost:8000/health

# 可用过滤字段
curl http://localhost:8000/schema
```

## 命令行检索

```bash
python -m rag.cli "少阴病脉微细但欲寐"
```

## 切分体例(按书)

| 书 | 切分单元 | 主要 metadata |
|---|---|---|
| 伤寒论辑义 | 条文(太阳病等起首)+ 方剂 | book, juan, pian, fang |
| 千金方 | 方剂(## 方名) | book, bu, pian, fang |
| 本草纲目 | 编号附方 + 药条头部 | book, bu, yao, xijie, fangfang_no, fangfang_name |
| 温病条辨 | 条文(汉字编号)+ 方剂 | book, juan, pian, binglei, tiaowen_no, fang |

## 后续扩展

- 加原文索引(`is_translation=False`)
- 加 BGE-Reranker 重排
- 迁 Qdrant server(docker)以支持集群
- 扩更多古籍(在 `chunking.py` 加切分函数)
