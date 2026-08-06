"""命令行检索(本地测试)。

用法(从仓库根目录):
    python -m rag.cli "少阴病脉微细但欲寐"
    python -m rag.cli "甘草 口舌生疮" 5
"""
import sys

from . import retrieve


def main():
    if len(sys.argv) < 2:
        print("用法: python -m rag.cli <query> [top_k]")
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    results = retrieve.search(query, top_k=top_k)
    print(f"查询: {query}(返回 {len(results)} 条)\n")
    for i, r in enumerate(results):
        print(f"[{i + 1}] score={r['score']:.4f}")
        text = r["text"].replace("\n", " ")
        print(f"    {text[:150]}...")
        meta = r["metadata"]
        meta_short = {k: v for k, v in meta.items() if k in ("book", "fang", "yao", "pian", "tiaowen_no", "fangfang_no")}
        print(f"    meta: {meta_short}")
        print()


if __name__ == "__main__":
    main()
