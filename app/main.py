"""FastAPI 应用入口。"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import agent as agent_api
from app.api import chat as chat_api
from app.api import documents as documents_api
from app.api import work_units as work_units_api
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Remote Sensing RAG",
        description="基于 LangChain + Chroma + SiliconFlow bge-m3 的遥感知识库问答系统",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_api.router)
    app.include_router(chat_api.router)
    app.include_router(agent_api.router)
    app.include_router(work_units_api.router)

    @app.get("/")
    def root() -> dict:
        return {"status": "ok", "service": "remote-sensing-rag"}

    @app.get("/health")
    def health() -> dict:
        return {"status": "healthy"}

    logger.info("FastAPI 应用已创建")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
