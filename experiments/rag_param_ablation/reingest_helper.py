"""临时向量库构建器：为消融实验创建独立的 Chroma 实例。

核心职责：
1. 在指定 persist_dir 创建独立 Chroma（不触碰正式 data/chroma）；
2. 从 examples/sample_docs/ 加载原始文档；
3. 用指定 chunk_size / chunk_overlap 切分（通过 split_document 显式传参）；
4. 写入临时向量库并返回可检索的 AblationVectorStore。

注意：
- Chroma 返回的是 cosine distance，VectorStore.search 内部已转换为
  similarity = 1.0 - distance，调用方无需再次转换。
- SiliconFlowEmbeddingClient 需要有效的 SILICONFLOW_API_KEY。
- 不修改正式 .env、不触碰正式 data/chroma、不删除已有文档。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.embeddings import SiliconFlowEmbeddingClient
from app.services.document_loader import load_document, PageContent
from app.services.splitter import split_document, Chunk
from app.services.vector_store import VectorStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 示例文档目录（原始干净文件名，无 doc_id 前缀）
# 项目根目录：reingest_helper.py → rag_param_ablation → experiments → project_root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCS_DIR = _PROJECT_ROOT / "examples" / "sample_docs"


class AblationVectorStore(VectorStore):
    """消融实验专用向量库，覆写 __init__ 使用独立 persist_dir 和 collection_name。

    继承 VectorStore 的 add_chunks / search / count 等方法，
    只是构造时不读取 get_settings() 中的 chroma_persist_dir。
    """

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        # 不调用 super().__init__()，完全自定义 Chroma 连接
        self.persist_dir = persist_dir
        self.embedder = SiliconFlowEmbeddingClient()
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )


def _make_deterministic_doc_id(filename: str, chunk_size: int, chunk_overlap: int) -> str:
    """根据文件名 + chunk 参数生成确定性 doc_id（非时间戳）。

    保证同一文件在不同 chunk 参数下 doc_id 不同，
    同一文件在同一 chunk 参数下 doc_id 相同（可复现）。
    """
    raw = f"ablation:{filename}:{chunk_size}:{chunk_overlap}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def build_temp_store(
    chunk_size: int,
    chunk_overlap: int,
    source_dir: Path | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> AblationVectorStore:
    """加载文档，按指定 chunk 参数切分并写入临时向量库。

    Args:
        chunk_size: 切分块大小。
        chunk_overlap: 切分重叠大小。
        source_dir: 文档源目录，默认 examples/sample_docs/。
        persist_dir: Chroma 持久化目录，必须与正式 data/chroma 不同。
        collection_name: Chroma collection 名称。

    Returns:
        可直接调用 .search() 的 AblationVectorStore 实例。

    Raises:
        FileNotFoundError: 源目录无文档。
        RuntimeError: embedding 或写入失败。
    """
    src = source_dir or SAMPLE_DOCS_DIR
    if persist_dir is None:
        raise ValueError("persist_dir 不能为空，必须指定独立临时目录")
    col_name = collection_name or f"ablation_c{chunk_size}_o{chunk_overlap}"

    persist_dir.mkdir(parents=True, exist_ok=True)
    store = AblationVectorStore(str(persist_dir), col_name)

    # 加载所有支持的文档
    supported = {".pdf", ".txt", ".md", ".markdown"}
    doc_files = sorted(
        f for f in src.iterdir() if f.suffix.lower() in supported
    )
    if not doc_files:
        raise FileNotFoundError(f"源目录 {src} 中没有可用的文档")

    total_chunks = 0
    for doc_path in doc_files:
        filename = doc_path.name
        doc_id = _make_deterministic_doc_id(filename, chunk_size, chunk_overlap)

        pages: List[PageContent] = load_document(doc_path)

        # split_document 接受显式 chunk_size / chunk_overlap，覆盖 get_settings() 默认值
        chunks: List[Chunk] = split_document(
            pages=pages,
            doc_id=doc_id,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        count = store.add_chunks(chunks)
        total_chunks += count
        logger.info(
            "切分 %s → %d chunks (chunk_size=%d, overlap=%d)",
            filename, count, chunk_size, chunk_overlap,
        )

    logger.info(
        "临时向量库构建完成: persist_dir=%s, collection=%s, 总 chunks=%d",
        persist_dir, col_name, total_chunks,
    )
    return store


def cleanup_temp_store(persist_dir: Path) -> None:
    """清理临时 Chroma 目录。

    在 tempfile.TemporaryDirectory 上下文管理器之外手动清理时使用。
    """
    import shutil
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
        logger.info("已清理临时向量库: %s", persist_dir)
