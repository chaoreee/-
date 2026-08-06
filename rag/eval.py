"""检索质量评估:5 个中医 query,对比 hybrid(dense+sparse+RRF) vs 纯 dense。

用法(需先建索引):
    python -m rag.eval
"""
from .retrieve import search, search_dense_only

QUERIES = [
    "桂枝汤主治什么",          # 术语精准,验 sparse 起效
    "温病上焦风温怎么治",      # 语义 + 术语
    "甘草治哪些口舌生疮",      # 药 + 证
    "妇人妊娠恶阻方",          # 千金方过滤候选
    "少阴病脉微细但欲寐",      # 伤寒术语
]


def _print(results, n=3):
    for i, r in enumerate(results[:n]):
        meta = r["metadata"]
        loc = meta.get("book", "")
        if meta.get("fang"):
            loc += f" · {meta['fang']}"
        elif meta.get("yao"):
            loc += f" · {meta['yao']}"
        elif meta.get("tiaowen_no"):
            loc += f" · 条{meta['tiaowen_no']}"
        print(f"  [{i + 1}] {r['score']:.4f} | {loc} | {r['text'][:55].replace(chr(10), ' ')}...")


def main():
    for q in QUERIES:
        print(f"\n{'=' * 70}\n查询: {q}\n{'=' * 70}")
        print("\n[hybrid: dense + sparse + RRF]")
        _print(search(q, top_k=5))
        print("\n[dense only]")
        _print(search_dense_only(q, top_k=5))


if __name__ == "__main__":
    main()
