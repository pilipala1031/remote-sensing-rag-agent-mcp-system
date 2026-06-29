"""文档相关 FastAPI 路由：上传、入库、列表、删除。"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agents.tools import clear_agent_search_cache
from app.agents.response_cache import (
    clear_agent_response_cache,
    invalidate_corpus_version,
)
from app.config import get_settings
from app.schemas import (
    ChunkInfo,
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestRequest,
    IngestResponse,
    UploadResponse,
)
from app.services.document_loader import load_document
from app.services.splitter import split_document
from app.services.vector_store import VectorStore
from app.utils.file_utils import (
    extract_doc_id_from_filename,
    find_file_by_doc_id,
    generate_doc_id,
    is_supported,
    save_upload_file,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    settings = get_settings()
    if not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型，仅支持 pdf/txt/md: {file.filename}",
        )
    doc_id = generate_doc_id(file.filename)
    saved = save_upload_file(file, settings.raw_data_path, doc_id)
    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        saved_path=str(saved),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(req: IngestRequest) -> IngestResponse:
    settings = get_settings()
    raw_dir = settings.raw_data_path

    if req.doc_id:
        targets = [find_file_by_doc_id(raw_dir, req.doc_id)]
        targets = [t for t in targets if t]
        if not targets:
            raise HTTPException(status_code=404, detail=f"未找到 doc_id={req.doc_id} 的文件")
    else:
        targets = [
            p
            for p in raw_dir.glob("*")
            if p.is_file() and is_supported(p.name)
        ]
        if not targets:
            raise HTTPException(status_code=404, detail="raw 目录下没有可入库文件")

    store = VectorStore()
    # 聚合结果（单文件为主）
    last_resp: IngestResponse | None = None
    total_chunks = 0

    for path in targets:
        filename = path.name
        doc_id = extract_doc_id_from_filename(filename)
        pages = load_document(path)
        chunks = split_document(pages, doc_id=doc_id, filename=filename)
        added = store.add_chunks(chunks)
        total_chunks += added
        last_resp = IngestResponse(
            doc_id=doc_id,
            filename=filename,
            chunk_count=added,
            chunks=[
                ChunkInfo(chunk_id=c.chunk_id, page=c.page, content_preview=c.text[:120])
                for c in chunks[:10]
            ],
            message=f"入库成功，本次共切分 {added} 个 chunk",
        )

    if last_resp is None:
        raise HTTPException(status_code=500, detail="入库失败")
    last_resp.message = f"共入库 {total_chunks} 个 chunk"

    # 知识库已更新，清空 Agent 检索缓存 + 响应缓存
    clear_agent_search_cache()
    invalidate_corpus_version()
    clear_agent_response_cache()

    return last_resp


@router.get("", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    store = VectorStore()
    docs = store.list_documents()
    items = [DocumentInfo(**d) for d in docs]
    return DocumentListResponse(total=len(items), documents=items)


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str) -> DeleteResponse:
    store = VectorStore()
    deleted = store.delete_by_doc_id(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"未找到 doc_id={doc_id} 的数据")

    # 知识库已更新，清空 Agent 检索缓存 + 响应缓存
    clear_agent_search_cache()
    invalidate_corpus_version()
    clear_agent_response_cache()

    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)
