# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Remote Sensing RAG is a production-grade RAG + Multi-Tool Agent knowledge-base QA system for the remote-sensing semantic-segmentation domain. It ingests PDF/TXT/MD documents, splits them (CN-aware), embeds them with SiliconFlow `BAAI/bge-m3` (1024-dim), persists them in Chroma (cosine HNSW), and answers questions via two parallel paths:

1. **Plain RAG** (`/api/chat/query`) — retrieve (optional rerank) → refuse-if-empty → LLM.
2. **Multi-Tool Agent** (`/api/agent/query`) — LangChain 1.0 `create_agent` with 7 tools (knowledge-base search, plan-and-search decomposition, dataset overview, and 4 structured domain-data tools), plus optional post-hoc Evidence Verification.

A Streamlit frontend (`frontend/streamlit_app.py`) wraps both modes behind a UI toggle.

A standalone **MCP server** (`mcp_server/server.py`, registered in `.mcp.json` as `remote-sensing-kb`) exposes the same domain knowledge as two MCP tools (`search_remote_sensing_kb`, `calculate_remote_sensing_metric`) so external MCP clients (e.g. Claude Code/Desktop) can query the KB directly. It shares the IoU/Precision/Recall/F1 kernel with the Agent path via `core/metrics.py`.

All code comments, prompts, log messages, tool outputs, and API error strings are in **Chinese**; preserve this convention when editing. The only intentional English surfaces are `@tool` `description` fields (so the LLM can reliably choose tools).

## Commands

```bash
# Install deps (Python 3.10+)
pip install -r requirements.txt

# Run backend (FastAPI on :8000, Swagger at /docs)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend (Streamlit on :8501, expects RAG_API_BASE or default http://127.0.0.1:8000)
streamlit run frontend/streamlit_app.py

# Run all tests
pytest -v

# Run a single test file / test
pytest tests/test_rag.py -v
pytest tests/test_rag.py::test_rag_refuse_when_empty -v

# Eval harness (NOT pytest; requires backend running on :8000)
python eval/run_rag_eval.py
python eval/run_agent_eval.py
# Results written to eval/results/*.json

# Ablation experiments (NOT pytest; standalone scripts, see experiments/ section below)
python -m experiments.rag_param_ablation.run_ablation
python -m experiments.rag_rerank_ablation.run_rerank_ablation

# Docker (backend only)
docker build -t remote-sensing-rag .
docker run -d --name rs-rag -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data remote-sensing-rag
# On Windows replace $(pwd) with %cd% or the absolute path.

# Docker Compose (full backend + frontend stack; see docker-compose.yml)
docker compose up -d --build

# MCP server (standalone; launched by MCP clients via .mcp.json)
python -m mcp_server.server
```

Required env (see `.env.example`): `SILICONFLOW_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `RAG_API_BASE` (frontend → backend URL), `DEMO_PASSWORD` (Streamlit password gate for public demos). Tunable params: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD` (default 0.3), `USE_RERANK` (default false), `RERANK_CANDIDATE_K` (default 10), `ENABLE_AGENT_VERIFICATION` (default true), `AGENT_VERIFICATION_MODE` (default `deferred`), `AGENT_VERIFICATION_LEVEL` (default `lightweight`), `AGENT_MAX_TOKENS` (default 1000), `ENABLE_AGENT_CACHE` (default false), `ENABLE_AGENT_RESPONSE_CACHE` (default true), `AGENT_RESPONSE_CACHE_TTL_SECONDS` (default 600), `AGENT_RESPONSE_CACHE_MAX_SIZE` (default 100). Rerank reuses `SILICONFLOW_API_KEY`/`SILICONFLOW_BASE_URL`; `RERANK_MODEL` defaults to `BAAI/bge-reranker-v2-m3`.

> ⚠️ The defaults above are the **code defaults** in `app/config.py`. Note that the shipped `.env.example` deliberately *flips* two of them to demo-friendlier values: it sets `USE_RERANK=true` and `ENABLE_AGENT_RESPONSE_CACHE=false`. Copying `.env.example` to `.env` therefore turns rerank ON and the L1 response cache OFF — the opposite of the code defaults. Keep this in mind when reasoning about observed behavior on a fresh checkout.

Tests live in `tests/` (14 `test_*.py` files + shared `conftest.py`). `test_embeddings.py` hits the real SiliconFlow API and is auto-skipped without a key; everything else mocks `Retriever`/`LLM`/`build_chat_model` and runs offline.

## Architecture

Layered backend under `app/` — request flow is `api → services → core`, with `agents/` as a peer orchestration layer for the Agent path:

```
api/        FastAPI APIRouter + Pydantic validation
  ├─ documents.py   /api/documents: upload / ingest / list / delete
  │                 (clears agent search cache + response cache on ingest & delete)
  ├─ chat.py        /api/chat/query (module-level RAGService singleton)
  └─ agent.py       /api/agent/query (module-level AgentService singleton)
                   /api/agent/verify (standalone Evidence Verification)
services/   RAG business orchestration
  ├─ document_loader.py   load_document() -> List[PageContent] (PDF keeps page no.)
  ├─ splitter.py          clean_text + RecursiveCharacterTextSplitter (CN-aware),
  │                       deterministic chunk_id = md5(doc_id:page:idx)[:12]
  ├─ vector_store.py      Chroma PersistentClient, cosine HNSW, search() = 1-distance
  ├─ retriever.py         cosine recall → optional rerank → top-K. use_rerank param
  │                       overrides settings per-request (None = use .env config)
  ├─ reranker.py          SiliconFlow bge-reranker-v2-m3 cross-encoder client;
  │                       graceful degradation (fallback to vector order on API failure)
  └─ rag_service.py       retrieve → refuse-if-empty → LLM.chat()
agents/     Multi-Tool Agent layer (LangChain 1.0)
  ├─ agent_service.py     RemoteSensingAgentService (thin orchestration + error backstop
  │                       + L1 response cache + L2 LLM cache toggle)
  ├─ langchain_agent.py   build_chat_model() (ChatOpenAI), create_agent, run_langchain_agent,
  │                       result parsing, verification toggle; get_remote_sensing_agent()
  │                       is @lru_cache(maxsize=1); _TrackingInMemoryCache for L2 LLM cache
  ├─ response_cache.py    AgentResponseCache (TTL + max_size OrderedDict); caches full
  │                       Agent response keyed on question + config + corpus_version
  ├─ tools.py             knowledge_base_search @tool (RAG-as-Tool, no LLM), @lru_cache(128);
  │                       set_rerank_override() module-level flag for per-request rerank control
  ├─ planning_tools.py    plan_and_search @tool (LLM-based query decomposition + merge);
  │                       uses get_agent_llm() to share cached ChatOpenAI instance
  ├─ domain_tools.py      5 @tools over static JSON (dataset overview/spec, models, metrics, calculator)
  ├─ domain_data_loader.py  load_json_data; get_datasets/models/metrics_data @lru_cache(1)
  ├─ verification.py      verify_answer(): refusal/no-evidence short-circuits + LLM check
  ├─ prompts.py           AGENT_SYSTEM_PROMPT + Evidence Verification template
  └─ types.py             AgentRunResult / AgentSource / AgentToolCall Pydantic models
core/       External model clients + shared pure-logic kernels (LangChain-compatible)
  ├─ llm.py            OpenAICompatibleLLMClient (BaseChatModel) + direct .chat() helper
  ├─ embeddings.py     SiliconFlowEmbeddingClient (Embeddings), batches 32, lazy dim probe
  ├─ prompts.py        RAG_SYSTEM_PROMPT + REFUSAL_ANSWER
  └─ metrics.py        calculate_metric() / SUPPORTED_METRICS / METRIC_ALIASES — shared IoU /
                       Precision / Recall / F1 kernel called by BOTH the Agent metrics_calculator
                       @tool and the MCP @tool (extracted to avoid duplicating the formula)
domain_data/  Static JSON knowledge: datasets.json (5), models.json (6), metrics.json (8)
utils/        file_utils (doc_id, save, lookup, SUPPORTED_EXTENSIONS), logger (stdout, INFO)
config.py     Settings (pydantic-settings) via @lru_cache get_settings() singleton
schemas.py    All Pydantic request/response models (RAG + Agent); both ChatQueryRequest
              and AgentQueryRequest have use_rerank (Optional[bool], default None) and
              AgentQueryRequest has include_trace flag (default true)
main.py       create_app() factory + CORS + 3 routers (documents, chat, agent)
frontend/     streamlit_app.py — RAG/Agent toggle, rerank checkbox, cache toggle,
              tool-calls/trace/verification/timing panels
eval/         Standalone eval scripts + shared metrics.py + eval_questions.json
              (21 questions) + eval_questions_with_labels.json (labeled variant) +
              generate_eval_labels.py (label generator) + results/
experiments/  Ablation experiments (see below)
tests/        pytest unit tests (14 test files + conftest.py); mocks Retriever/LLM/build_chat_model
docs/         Design & ops docs: agent_trace_iteration.md, DEPLOY.md (deployment
              incl. Cloudflare Tunnel), project_deep_dive.md
examples/     sample_docs/ (10 .md sample docs 01_..10_) + sample_questions.json
mcp_server/   Standalone MCP server (server.py, FastMCP "remote-sensing-kb") — two
              MCP @tools over the same domain knowledge as the Agent; shares core/metrics.py
.mcp.json     Registers the MCP server: `python -m mcp_server.server`
data/         Runtime data: chroma/ (persistent vector store), raw/ (uploaded raw files)
```

**Where new code goes**: new capability/business logic → `services/` (RAG) or `agents/` (Agent); new model provider → `core/`; new HTTP route → `api/`.

### experiments/ — Ablation & Analysis Scripts

Standalone scripts for parameter tuning and rerank evaluation. Each sub-project has its own README with results and conclusions.

```
experiments/
  ├─ eval_with_labels.py           Label adapter: flattens eval_questions_with_labels.json
  │                                (nested eval_labels.*) into the flat structure metrics.py expects
  ├─ agent_trace_analyzer.py       Parses agent trace JSON into per-step timing breakdowns
  ├─ rag_param_ablation/           Block 2: chunk_size/threshold/top_k ablation
  │   ├─ reingest_helper.py        AblationVectorStore subclass + build_temp_store() for isolated
  │   │                            Chroma collections (avoids polluting production DB)
  │   ├─ run_ablation.py           3-stage ablation (retrieval → refusal-safety → answer-level),
  │   │                            Windows-safe temp dir cleanup (gc.collect + shutil.rmtree)
  │   └─ results/*.json            Parameter sensitivity + recommended params
  └─ rag_rerank_ablation/          Block 4: SiliconFlow bge-reranker-v2-m3 evaluation
      ├─ reranker.py               Experiment-internal rerank client (separate from production
      │                            app/services/reranker.py); reads RERANK_* env then falls
      │                            back to SILICONFLOW_* credentials via Settings
      ├─ run_rerank_ablation.py    3-stage rerank ablation (retrieval → out-of-scope → answer)
      └─ results/*.json            Per-stage results + final recommendation (rerank_k10 best)
```

Key patterns shared by both ablation experiments:
- **AblationVectorStore**: subclasses `VectorStore`, overrides `__init__` for independent `persist_dir`/`collection_name` — tests run against isolated temp Chroma DBs, not production data.
- **ablation_temp_dir**: context manager with `gc.collect()` + `shutil.rmtree(ignore_errors=True)` for Windows Chroma file-locking compatibility.
- **Shared scoring formulas**: `retrieval_score`, `refusal_score`, `answer_score` — weighted combinations of source_hit_rate, recall@k, MRR, keyword_coverage, etc.

## Key Design Decisions

- **Anti-hallucination, 3 layers for RAG / 4 for Agent** — preserve all layers when modifying query logic:
  1. `VectorStore.search` filters out chunks with `score < SIMILARITY_THRESHOLD` *before* results leave the store (`vector_store.py`).
  2. `RAGService.answer` short-circuits to `REFUSAL_ANSWER` *without calling the LLM* when the filtered retrieval is empty (`rag_service.py`).
  3. `RAG_SYSTEM_PROMPT` instructs the LLM to refuse when context is insufficient (`core/prompts.py`).
  4. (Agent only) `verify_answer()` runs a post-hoc LLM check of the answer against `sources` + `tool_calls` outputs; refusal answers short-circuit to `verified=True`, no-evidence answers short-circuit to `verified=False` (`agents/verification.py`, gated by `ENABLE_AGENT_VERIFICATION`).

- **Agent Verification modes** (`AGENT_VERIFICATION_MODE`): `off` = skip, `sync` = run inline in `/api/agent/query`, `deferred` (default) = `/api/agent/query` returns immediately, frontend calls `/api/agent/verify` separately. Level (`AGENT_VERIFICATION_LEVEL`) controls payload truncation: `lightweight` (default) trims answer/sources/tool_calls, `full` is relatively complete but still capped.

- **Agent Trace**: `AgentQueryRequest.include_trace` (default `true`) controls whether `agent_trace`, `trace_events`, and detailed `tool_calls` are returned. Set to `false` in production to reduce response size.

- **Agent caching, two layers** — do not collapse or remove either layer:
  - **L1 Response Cache** (`response_cache.py`, `ENABLE_AGENT_RESPONSE_CACHE`, default true): caches the complete Agent response dict in `AgentResponseCache` (TTL + max_size OrderedDict). Key = sha256 of normalized_question + use_rerank + top_k + similarity_threshold + rerank_candidate_k + llm_model + agent_max_tokens + verification_mode + verification_level + corpus_version + domain_data_hash + include_trace. On hit: zero LLM/tool calls, sub-ms response. Error results (containing "异常" in errors) are not cached.
  - **L2 LLM Cache** (`_TrackingInMemoryCache` in `langchain_agent.py`, `ENABLE_AGENT_CACHE`, default false): LangChain `InMemoryCache` subclass with hit/miss counting and tool_call_id normalization in cache keys. Only effective for Agent Round 1 (LLM deciding tools); Round 2+ misses due to non-deterministic tool execution order in LangGraph. Controlled per-request via `AgentQueryRequest.enable_cache`.
  - **Cache invalidation**: `documents.py` calls `clear_agent_search_cache()` + `clear_agent_response_cache()` + `invalidate_corpus_version()` after ingest *and* after delete.

- **Cosine score convention**: `VectorStore.search` computes `score = 1.0 - distance` (Chroma returns cosine *distance*, not similarity). Threshold applies to this similarity score.

- **Rerank (production)**: `Retriever.retrieve()` is the single injection point for both RAG and Agent paths. When `use_rerank=True` (from request body or `.env`), it retrieves `RERANK_CANDIDATE_K` (default 10) candidates, calls SiliconFlow `bge-reranker-v2-m3` cross-encoder via `reranker.py`, and returns the top-K reranked results. Graceful degradation: API failure falls back to original vector order (`used_fallback=True`). Both `ChatQueryRequest` and `AgentQueryRequest` accept an optional `use_rerank` field (default `None` = use `.env` config). For the Agent path, `AgentService.query()` sets a module-level `_rerank_override` flag in `tools.py` before invoking the agent and resets it after.

- **Two LLM clients, intentional split**: the RAG path uses `OpenAICompatibleLLMClient` via its direct `.chat()` helper (not `_generate`). The Agent path uses `langchain_openai.ChatOpenAI` (built in `agents/langchain_agent.py::build_chat_model`) because it needs `bind_tools` for tool-calling. `planning_tools.py` uses `get_agent_llm()` to reuse the same cached ChatOpenAI instance — do not create new ChatOpenAI instances. Do not collapse these into one.

- **ID conventions**: `doc_id` = `md5(filename:time.time())[:16]` (time-based, unique per upload); prefixed onto the saved raw file as `{doc_id}_{filename}`. `chunk_id` = `md5(doc_id:page:idx)[:12]` (deterministic given same doc_id/page/order). `extract_doc_id_from_filename` relies on the `{doc_id}_` prefix — **do not rename raw files**.

- **Singletons vs. fresh instances**:
  - Cached: `get_settings()` (`@lru_cache`), module-level `_rag_service` in `chat.py`, module-level `_agent_service` in `agent.py`, `get_remote_sensing_agent()` (`@lru_cache(1)`), `get_agent_llm()` (returns shared `_agent_llm`), `_agent_response_cache` singleton in `response_cache.py`, `get_datasets/models/metrics_data` (`@lru_cache(1)` each), `_cached_search` in `agents/tools.py` (`@lru_cache(128)` keyed on `normalize_query`).
  - **Not cached**: `VectorStore` — `documents.py` creates a fresh instance per request because each instantiation opens a new Chroma client connection.

- **RAG-as-Tool compression**: `knowledge_base_search` caps `contexts.content ≤ 500 chars` and `sources.content_preview ≤ 150 chars` (with `"..."`) to limit LLM input tokens. `normalize_query` (strip + lowercase + whitespace-collapse) maximizes LRU cache hits.

- **Plan-and-search merge** (`planning_tools.py`): dedups sub-query results by `chunk_id` (keeping highest score), re-sorts by score desc, renumbers `source_id` from `source_1`, aggregates `search_elapsed`. `_parse_decomposition` handles clean/markdown-fenced/embedded JSON and falls back to `[original_query]`.

- **Domain tools explicitly avoid fabricating metrics**: `model_comparison_table` returns only the curated metadata (no mIoU/params/FLOPs numbers). `metrics_calculator` computes IoU/Precision/Recall/F1 from TP/FP/FN with zero-denominator guards via the shared `core/metrics.py::calculate_metric()` kernel. That same kernel is reused by the MCP server's `calculate_remote_sensing_metric` tool — keep the two callers in sync by editing `core/metrics.py`, not by duplicating the formula.

- **LangChain integration**: `OpenAICompatibleLLMClient` and `SiliconFlowEmbeddingClient` implement `BaseChatModel`/`Embeddings` so they can drop into LangChain chains; all 7 Agent tools use `@tool`. The Agent is built via LangChain 1.0 `create_agent` — LangGraph `StateGraph` is *not* used directly.

## Conventions

- **Chinese** comments/logs/user-facing strings/tool outputs throughout. English only in `@tool` `description` fields.
- `from __future__ import annotations` is used in every module — keep type hints as strings.
- Logger is obtained per-module via `get_logger(__name__)`; no `print`.
- Supported upload extensions: `.pdf`, `.txt`, `.md`, `.markdown` (see `utils/file_utils.SUPPORTED_EXTENSIONS`).
- Errors are broadly caught (`except Exception`) and re-raised as `RuntimeError` with Chinese context in `core/` clients; the HTTP layer converts to `HTTPException`.
- Agent timing: `AgentRunResult.timing` carries `total_elapsed`, `agent_invoke_elapsed`, per-tool `elapsed`, and (when enabled) `verification_elapsed`. `agent_trace` follows `agent_started → tool_called → tool_result_parsed → agent_finished`.
- All config defaults live in `app/config.py` `Settings` class via `Field(default=..., alias=...)`. Variables not in `.env` silently use these defaults — no runtime error.
