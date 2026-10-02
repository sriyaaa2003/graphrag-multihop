"""HTTP API. Handlers validate, authenticate and delegate."""

import logging
import secrets
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .. import graphstore
from ..config import Settings, get_settings
from ..db import Database
from ..embeddings import Embedder, FastEmbedEmbedder
from ..extract import Extractor, LLMExtractor, ProperNounExtractor
from ..llm import build_provider
from ..service import GraphRagService, ServiceError

log = logging.getLogger("graphhop")


class QueryBody(BaseModel):
    collection: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default="hybrid", pattern="^(vector|graph|hybrid)$")
    k: int | None = Field(default=None, ge=1)
    generate: bool = False
    w_graph: float | None = Field(default=None, ge=0)
    w_dense: float | None = Field(default=None, ge=0)
    max_hops: int | None = Field(default=None, ge=1, le=6)
    graph_strategy: str | None = Field(default=None, pattern="^(expand|ppr)$")


class DocumentBody(BaseModel):
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=500)
    text: str = Field(min_length=1, max_length=2_000_000)


def build_service(settings: Settings, embedder: Embedder | None = None) -> GraphRagService:
    db = Database(settings.db_path)
    llm = None
    if settings.llm_provider and settings.llm_model_id:
        llm = build_provider(
            settings.llm_provider,
            settings.llm_model_id,
            base_url=settings.llm_base_url,
            api_key_env=settings.llm_api_key_env,
            timeout=settings.llm_timeout_seconds,
        )
    extractor: Extractor
    if settings.extractor == "llm":
        if llm is None:
            raise RuntimeError("GH_EXTRACTOR=llm requires an LLM to be configured")
        extractor = LLMExtractor(llm)
    else:
        extractor = ProperNounExtractor()
    return GraphRagService(
        db, embedder or FastEmbedEmbedder(settings.embedding_model), extractor, settings, llm
    )


def create_app(
    settings: Settings | None = None, *, service: GraphRagService | None = None
) -> FastAPI:
    settings = settings or get_settings()
    svc = service or build_service(settings)
    app = FastAPI(title="graphhop", version="0.1.0")

    def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if not settings.admin_api_key:
            raise HTTPException(503, "admin API key is not configured on the server")
        if not x_api_key or not secrets.compare_digest(x_api_key, settings.admin_api_key):
            raise HTTPException(401, "invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/query")
    def query(body: QueryBody) -> dict[str, Any]:
        try:
            r = svc.query(
                body.collection,
                body.query,
                mode=body.mode,
                generate=body.generate,
                k=body.k,
                w_graph=body.w_graph,
                w_dense=body.w_dense,
                max_hops=body.max_hops,
                graph_strategy=body.graph_strategy,
            )
        except ServiceError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return {
            "results": [
                {
                    "document": c.external_id,
                    "title": c.title,
                    "text": c.text,
                    "score": c.score,
                    "dense_rank": c.dense_rank,
                    "graph_rank": c.graph_rank,
                }
                for c in r.chunks
            ],
            "trace": {
                "mode": r.trace.mode,
                "query_entities": r.trace.query_seeds,
                "dense_seed_entities": r.trace.dense_seed_entities,
                "top_entities": r.trace.top_entities,
            },
            "answer": r.answer,
            "abstained": r.abstained,
            "citations": r.citations,
        }

    @app.put("/v1/collections/{name}/documents", dependencies=[Depends(require_admin)])
    def put_document(name: str, body: DocumentBody) -> dict[str, Any]:
        try:
            changed = graphstore.ingest_document(
                svc.db.conn,
                svc.embedder,
                svc.extractor,
                collection=name,
                external_id=body.external_id,
                title=body.title,
                text=body.text,
                params=graphstore.IngestParams(
                    settings.chunk_chars, settings.chunk_overlap_chars, settings.title_as_entity
                ),
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"external_id": body.external_id, "changed": changed}

    @app.delete(
        "/v1/collections/{name}/documents/{external_id:path}", dependencies=[Depends(require_admin)]
    )
    def delete_document(name: str, external_id: str) -> dict[str, str]:
        if not graphstore.delete_document(svc.db.conn, name, external_id):
            raise HTTPException(404, "document not found")
        return {"deleted": external_id}

    @app.get("/v1/collections/{name}/stats")
    def stats(name: str) -> dict[str, int]:
        c = svc.db.conn
        row = c.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown collection")
        cid = row["id"]

        def q(sql: str) -> int:
            return int(c.execute(sql, (cid,)).fetchone()[0])

        return {
            "documents": q("SELECT COUNT(*) FROM documents WHERE collection_id=?"),
            "chunks": q("SELECT COUNT(*) FROM chunks WHERE collection_id=?"),
            "entities": q("SELECT COUNT(*) FROM entities WHERE collection_id=?"),
            "edges": q(
                "SELECT COUNT(*) FROM (SELECT 1 FROM edge_contrib e JOIN entities x ON x.id=e.a "
                "WHERE x.collection_id=? GROUP BY e.a, e.b)"
            ),
        }

    return app
