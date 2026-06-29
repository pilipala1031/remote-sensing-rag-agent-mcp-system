"""Chroma 向量数据库封装：add / search / delete / list。"""
from __future__ import annotations

from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.embeddings import SiliconFlowEmbeddingClient
from app.services.splitter import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "remote_sensing_rag"


class VectorStore:
    """Chroma 持久化向量库操作封装。

    
通过 SiliconFlowEmbeddingClient 生成文本向量，并将向量、文本和元数据写入 Chroma。
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.persist_dir = str(settings.chroma_path)
        self.embedder = SiliconFlowEmbeddingClient()
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 写入 ----------
    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("写入 %d 个 chunk 到 Chroma", len(chunks))
        return len(chunks)

    # ---------- 检索 ----------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """返回按相关度降序的 chunk 列表，元素含 score(相似度=1-distance)。"""
        settings = get_settings()
        top_k = top_k or settings.top_k
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

        query_emb = self.embedder.embed_query(query)
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where,
        )

        results: List[dict] = []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        for _id, dist, doc, meta in zip(ids, dists, docs, metas):
            score = 1.0 - float(dist)  # cosine distance -> similarity
            if score < threshold:
                continue
            item: dict[str, Any] = {
                "chunk_id": _id,
                "score": score,
                "content": doc,
                **(meta or {}),
            }
            results.append(item)
        return results

    # ---------- 删除 ----------
    def delete_by_doc_id(self, doc_id: str) -> int:
        before = self.collection.count()
        self.collection.delete(where={"doc_id": doc_id})
        after = self.collection.count()
        deleted = before - after
        logger.info("删除 doc_id=%s, 共 %d 条", doc_id, deleted)
        return max(deleted, 0)

    # ---------- 列表 ----------
    def list_documents(self) -> List[dict]:
        """按 doc_id 聚合返回每个文档的 filename 与 chunk 数量。"""
        res = self.collection.get()
        metas = res.get("metadatas") or []
        agg: dict[str, dict] = {}
        for m in metas:
            if not m:
                continue
            did = m.get("doc_id")
            fname = m.get("filename", "unknown")
            if did not in agg:
                agg[did] = {"doc_id": did, "filename": fname, "chunk_count": 0}
            agg[did]["chunk_count"] += 1
        return list(agg.values())

    def count(self) -> int:
        return self.collection.count()
