"""RAG 参数消融实验子包。

只关注 chunk_size、chunk_overlap、similarity_threshold 三个参数，
通过三阶段实验（Retrieval / Refusal / Answer）确定合理取值。
"""
