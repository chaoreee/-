"""4 书分别切分 + metadata 提取 + ## 噪声清洗。

每书体例不同,各自实现切分函数,统一输出 Chunk 列表。
- 千金方:方剂列表体例(## 方名)
- 本草纲目:药物条目体例(编号附方)
- 温病条辨:条辨体例(汉字编号条文)
- 伤寒论辑义:散文+注文体例(条文 + ## 方名)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .config import MAX_CHUNK_CHARS, TRANSLATION_FILES


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Chunk:
    text: str
    book: str
    is_translation: bool = True
    # 通用层级
    juan: Optional[str] = None
    pian: Optional[str] = None
    bu: Optional[str] = None
    yao: Optional[str] = None
    fang: Optional[str] = None
    # 温病
    binglei: Optional[str] = None
    tiaowen_no: Optional[str] = None
    # 本草
    fangfang_no: Optional[int] = None
    fangfang_name: Optional[str] = None
    xijie: Optional[str] = None

    def payload(self) -> dict:
        """Qdrant payload:text + book + is_translation + 非空层级字段。"""
        d: dict = {"text": self.text, "book": self.book, "is_translation": self.is_translation}
        for k in ("juan", "pian", "bu", "yao", "fang",
                  "binglei", "tiaowen_no", "fangfang_no", "fangfang_name", "xijie"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


# ============================================================
# 通用工具
# ============================================================

def _split_long(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """超长文本按句末标点切分,每块不超过 max_chars;单句超长则硬切。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[。！？!?])", text)
    parts = [p for p in parts if p]
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur:
                chunks.append(cur)
            if len(p) > max_chars:  # 单句超长,硬切
                for i in range(0, len(p), max_chars):
                    chunks.append(p[i:i + max_chars])
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _emit(chunks: list[Chunk], text: str, book: str, **meta) -> None:
    """切长后追加 Chunk。"""
    body = text.strip()
    if not body:
        return
    for t in _split_long(body):
        chunks.append(Chunk(text=t, book=book, **meta))


def _clean_title(title: str) -> str:
    """去行尾 ## 残留、首尾空白。"""
    return title.strip().rstrip("#").strip()


# ============================================================
# 千金方 — 方剂列表体例
# ============================================================

def chunk_qianjin(text: str) -> list[Chunk]:
    book = "千金方"
    lines = text.split("\n")
    chunks: list[Chunk] = []
    cur_bu = cur_pian = cur_fang = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            _emit(chunks, "\n".join(buf), book, bu=cur_bu, pian=cur_pian, fang=cur_fang)
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^##\s+(.+)$", s)
        if m:
            title = _clean_title(m.group(1))
            if not title:
                continue
            # 书名
            if "千金方" in title:
                flush()
                cur_bu = cur_pian = cur_fang = None
                continue
            # 部类·篇名(含 ·)
            if "·" in title:
                flush()
                parts = title.split("·", 1)
                cur_bu = parts[0].strip()
                cur_pian = parts[1].strip()
                cur_fang = None
                continue
            # 子标题(含"N首"等,非方名)
            if re.search(r"\d+首|[方论灸]\d+|首[）)]", title):
                flush()
                cur_fang = None  # 归当前 pian 下,不设 fang
                continue
            # 方名
            flush()
            cur_fang = title
            continue
        buf.append(line)
    flush()
    return chunks


# ============================================================
# 本草纲目 — 药物条目体例(编号附方)
# ============================================================

_BENCAO_SECTIONS = {"主治", "释名", "气味", "附方", "发明"}


def _extract_fangfang_name(body: str) -> Optional[str]:
    for pat in (r"此方名[“\"']([^”\"']+)[”\"']",
                r"名[叫为][“\"']([^”\"']+)[”\"']",
                r"叫[“\"']([^”\"']+)[”\"']"):
        m = re.search(pat, body)
        if m:
            return m.group(1).strip()
    return None


def chunk_bencao(text: str) -> list[Chunk]:
    book = "本草纲目"
    lines = text.split("\n")
    chunks: list[Chunk] = []
    cur_bu = cur_yao = None
    cur_ff_no: Optional[int] = None
    ff_buf: list[str] = []
    head_buf: list[str] = []
    in_head = False  # 药条头部(未遇编号附方)

    def flush_ff():
        nonlocal ff_buf, cur_ff_no
        if ff_buf:
            body = "\n".join(ff_buf)
            fname = _extract_fangfang_name(body)
            _emit(chunks, body, book, bu=cur_bu, yao=cur_yao,
                  xijie="附方", fangfang_no=cur_ff_no, fangfang_name=fname)
            ff_buf = []

    def flush_head():
        nonlocal head_buf
        if head_buf:
            _emit(chunks, "\n".join(head_buf), book, bu=cur_bu, yao=cur_yao, xijie="概览")
            head_buf = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^##\s+(.+)$", s)
        if m:
            title = _clean_title(m.group(1))
            if not title:
                continue
            # 部类(草部/木部...)
            if re.match(r"^.{0,4}部$", title):
                flush_ff(); flush_head()
                cur_bu = title
                cur_yao = None
                cur_ff_no = None
                in_head = False
                continue
            # 小节标题(## 主治/释名/气味 等)——归当前累积,不切分
            if title in _BENCAO_SECTIONS:
                (ff_buf if cur_ff_no is not None else head_buf).append(line)
                continue
            # 误转 ##(含标点或过长,如 "## 部""## 患处。""## 三、四天后…")
            if re.search(r"[。！？；，、：]", title) or len(title) > 12:
                (ff_buf if cur_ff_no is not None else head_buf).append(line)
                continue
            # 药名
            flush_ff(); flush_head()
            cur_yao = title
            cur_ff_no = None
            in_head = True
            head_buf = []
            continue
        # 药名前的内容跳过
        if cur_yao is None:
            continue
        # 编号附方
        mf = re.match(r"^(\d+)、", s)
        if mf:
            flush_ff()
            if in_head:
                flush_head()
                in_head = False
            cur_ff_no = int(mf.group(1))
            ff_buf = [line]
            continue
        # 普通行
        if cur_ff_no is not None:
            ff_buf.append(line)
        else:
            head_buf.append(line)
    flush_ff(); flush_head()
    return chunks


# ============================================================
# 温病条辨 — 条辨体例(汉字编号条文)
# ============================================================

_WB_TIAOWEN = re.compile(r"^([一二三四五六七八九十]+)、")
_WB_FANG_TAIL = re.compile(r"(汤|散|丸|丹|饮|膏|方)$")


def chunk_wenbing(text: str) -> list[Chunk]:
    book = "温病条辨"
    lines = text.split("\n")
    chunks: list[Chunk] = []
    cur_juan = cur_pian = cur_binglei = cur_tiaowen = cur_fang = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            _emit(chunks, "\n".join(buf), book, juan=cur_juan, pian=cur_pian,
                  binglei=cur_binglei, tiaowen_no=cur_tiaowen, fang=cur_fang)
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^#+\s+(.+)$", s)  # 统一 # / ##
        if m:
            title = _clean_title(m.group(1))
            if not title:
                continue
            # 卷·篇·病类
            mb = re.match(r"^(卷.)·(.+篇)·(.+)$", title)
            if mb:
                flush()
                cur_juan, cur_pian, cur_binglei = mb.group(1), mb.group(2), mb.group(3)
                cur_tiaowen = cur_fang = None
                continue
            # 卷·篇(无病类)
            mb2 = re.match(r"^(卷.)·(.+)$", title)
            if mb2:
                flush()
                cur_juan, cur_pian = mb2.group(1), mb2.group(2)
                cur_binglei = cur_tiaowen = cur_fang = None
                continue
            # 书名
            if "温病条辨" in title:
                flush()
                cur_juan = cur_pian = cur_binglei = cur_tiaowen = cur_fang = None
                continue
            # 条文编号被提成 ## (如 "## 二、凡病温者…")
            mt = _WB_TIAOWEN.match(title)
            if mt:
                flush()
                cur_tiaowen = mt.group(1)
                cur_fang = None
                buf = [title]  # 标题即条文首句,作为正文
                continue
            # 方名(以 汤/散/丸/丹/饮/膏/方 结尾)
            if _WB_FANG_TAIL.search(title):
                flush()
                cur_fang = title
                cur_tiaowen = None
                continue
            # 其他标题归当前
            buf.append(line)
            continue
        # 纯文本条文编号
        mt = _WB_TIAOWEN.match(s)
        if mt:
            flush()
            cur_tiaowen = mt.group(1)
            cur_fang = None
            buf = [line]
            continue
        buf.append(line)
    flush()
    return chunks


# ============================================================
# 伤寒论辑义 — 散文+注文体例
# ============================================================

_SH_JUAN = re.compile(r"^卷[一二三四五六七]$")
_SH_PIAN = re.compile(r"^辨.+脉证并治")
_SH_TIAOWEN = re.compile(r"^(太阳病|少阳病|阳明病|少阴病|厥阴病|太阴病|霍乱|阴阳易)")
_SH_FANG_TAIL = re.compile(r"(汤方|散方|丸方|汤|散|丸)$")


def chunk_shanghan(text: str) -> list[Chunk]:
    book = "伤寒论辑义"
    lines = text.split("\n")
    chunks: list[Chunk] = []
    cur_juan = cur_pian = cur_fang = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            _emit(chunks, "\n".join(buf), book, juan=cur_juan, pian=cur_pian, fang=cur_fang)
            buf = []

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^##\s+(.+)$", s)
        if m:
            title = _clean_title(m.group(1))
            if not title:
                continue
            # 误转 ##(案汪/不/含标点/过长)
            if title in ("案汪", "不") or re.search(r"[。！？；]", title) or len(title) > 15:
                buf.append(line)
                continue
            # 书名
            if "伤寒论辑义" in title:
                flush()
                cur_juan = cur_pian = cur_fang = None
                continue
            # 卷 / 篇
            if _SH_JUAN.match(title):
                flush()
                cur_juan = title
                cur_pian = cur_fang = None
                continue
            if _SH_PIAN.match(title):
                flush()
                cur_pian = title
                cur_fang = None
                continue
            # 方名
            if _SH_FANG_TAIL.search(title):
                flush()
                cur_fang = title
                continue
            # 其他(附录文献名等)作为方名层级
            flush()
            cur_fang = title
            continue
        # 纯文本卷 / 篇
        if _SH_JUAN.match(s):
            flush()
            cur_juan = s
            cur_pian = cur_fang = None
            continue
        if _SH_PIAN.match(s):
            flush()
            cur_pian = s
            cur_fang = None
            continue
        # 条文起首 → 新块(注文随条文)
        if _SH_TIAOWEN.match(s) and buf:
            flush()
            cur_fang = None
        buf.append(line)
    flush()
    return chunks


# ============================================================
# 汇总
# ============================================================

_CHUNKERS = {
    "千金方": chunk_qianjin,
    "本草纲目": chunk_bencao,
    "温病条辨": chunk_wenbing,
    "伤寒论辑义": chunk_shanghan,
}


def chunk_all() -> list[Chunk]:
    """切分全部 4 书译文,返回 Chunk 列表。"""
    all_chunks: list[Chunk] = []
    for book, path in TRANSLATION_FILES.items():
        if not path.exists():
            print(f"[警告] 缺失文件: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        chunker = _CHUNKERS[book]
        book_chunks = chunker(text)
        print(f"  {book}: {len(book_chunks)} 块")
        all_chunks.extend(book_chunks)
    return all_chunks


# ============================================================
# 命令行入口(验证用)
# ============================================================

if __name__ == "__main__":
    from collections import Counter
    print("切分 4 书译文...")
    chunks = chunk_all()
    print(f"\n总块数: {len(chunks)}")
    by_book = Counter(c.book for c in chunks)
    for book, n in by_book.items():
        print(f"  {book}: {n}")
    # 抽样每书 3 块
    for book in TRANSLATION_FILES:
        book_chunks = [c for c in chunks if c.book == book]
        if not book_chunks:
            continue
        print(f"\n--- {book} 抽样(前3块) ---")
        for c in book_chunks[:3]:
            print(f"  text: {c.text[:50].replace(chr(10),' ')}...")
            meta = {k: v for k, v in c.payload().items() if k not in ("text", "book", "is_translation")}
            print(f"  meta: {meta}")
