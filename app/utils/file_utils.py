"""文件工具：doc_id 生成、文件保存、扩展名校验。"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def generate_doc_id(filename: str) -> str:
    """根据文件名 + 时间戳 hash 生成稳定 doc_id。"""
    import time

    raw = f"{filename}:{time.time()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def save_upload_file(upload_file, save_dir: Path, doc_id: str) -> Path:
    """把 UploadFile / 文件流保存到 save_dir。返回保存后的绝对路径。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    # 文件名加 doc_id 前缀避免重名覆盖
    safe_name = f"{doc_id}_{Path(upload_file.filename).name}"
    target = save_dir / safe_name
    with target.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return target


def find_file_by_doc_id(raw_dir: Path, doc_id: str) -> Optional[Path]:
    """根据 doc_id 前缀在 raw 目录中查找文件。"""
    for p in raw_dir.glob(f"{doc_id}_*"):
        return p
    return None


def extract_doc_id_from_filename(filename: str) -> str:
    """从 'docid_realname.pdf' 中提取 doc_id 前缀。"""
    return filename.split("_", 1)[0]
