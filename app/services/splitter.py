"""文本清洗 + 切分，使用 LangChain RecursiveCharacterTextSplitter。

切分后的 chunk 会带上 metadata：
- doc_id
- filename
- page
- chunk_id (顺序编号)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.services.document_loader import PageContent
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int
    metadata: dict


def clean_text(text: str) -> str:
    """基础文本清洗。"""
    if not text:
        return ""
    # 统一空白
    text = text.replace("\u3000", " ")
    # 去掉连续空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉行尾空白
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _make_chunk_id(doc_id: str, page: int, idx: int) -> str:
    raw = f"{doc_id}:{page}:{idx}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def split_document(
    pages: List[PageContent],
    doc_id: str,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Chunk]:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )

    chunks: List[Chunk] = []
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        texts = splitter.split_text(cleaned)
        for idx, t in enumerate(texts):
            cid = _make_chunk_id(doc_id, page.page, idx)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=t,
                    page=page.page,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page.page,
                        "chunk_id": cid,
                    },
                )
            )
    logger.info("文档 %s 切分完成，共 %d 个 chunk", filename, len(chunks))
    return chunks
