"""文档解析：支持 PDF / TXT / Markdown，PDF 保留页码信息。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    page: int
    text: str


def load_document(file_path: Path) -> List[PageContent]:
    """返回按页切分的文档内容列表；TXT/MD 视为单页。"""
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _load_pdf(file_path)
        if suffix in {".txt", ".md", ".markdown"}:
            return _load_text(file_path)
        raise ValueError(f"暂不支持的文件类型: {suffix}")
    except Exception as e:  # noqa: BLE001
        logger.error("解析文档失败 %s: %s", file_path, e)
        raise


def _load_pdf(file_path: Path) -> List[PageContent]:
    import pypdf

    pages: List[PageContent] = []
    reader = pypdf.PdfReader(str(file_path))
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PageContent(page=idx, text=text))
    return pages


def _load_text(file_path: Path) -> List[PageContent]:
    # markdown 作为纯文本处理，保留原内容
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return [PageContent(page=1, text=text)]
