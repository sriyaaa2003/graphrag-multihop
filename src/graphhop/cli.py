"""Command-line interface: graphhop <command>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import graphstore
from .config import get_settings
from .db import Database
from .embeddings import FastEmbedEmbedder
from .extract import ProperNounExtractor


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("graphhop.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    s = get_settings()
    db = Database(s.db_path)
    embedder = FastEmbedEmbedder(s.embedding_model)
    extractor = ProperNounExtractor()
    params = graphstore.IngestParams(s.chunk_chars, s.chunk_overlap_chars, s.title_as_entity)
    changed = skipped = 0
    for p in args.paths:
        path = Path(p)
        files = (
            [f for f in sorted(path.rglob("*")) if f.suffix in {".txt", ".md"}]
            if path.is_dir()
            else [path]
        )
        for f in files:
            did = graphstore.ingest_document(
                db.conn,
                embedder,
                extractor,
                collection=args.collection,
                external_id=str(f),
                title=f.stem.replace("_", " ").replace("-", " "),
                text=f.read_text(encoding="utf-8"),
                params=params,
            )
            changed += did
            skipped += not did
    print(f"ingested {changed} document(s), {skipped} unchanged")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from .service import GraphRagService

    s = get_settings()
    db = Database(s.db_path)
    svc = GraphRagService(db, FastEmbedEmbedder(s.embedding_model), ProperNounExtractor(), s)
    r = svc.query(args.collection, args.query, mode=args.mode, k=args.k)
    print(
        json.dumps(
            {
                "trace": {
                    "query_entities": r.trace.query_seeds,
                    "top_entities": r.trace.top_entities,
                },
                "results": [
                    {
                        "document": c.external_id,
                        "score": round(c.score, 5),
                        "dense_rank": c.dense_rank,
                        "graph_rank": c.graph_rank,
                    }
                    for c in r.chunks
                ],
            },
            indent=2,
        )
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval_hotpot import format_table, run

    s = get_settings()
    llm_extractor = None
    if args.llm_extractor:
        from .extract import LLMExtractor
        from .llm import build_provider

        llm_extractor = LLMExtractor(
            build_provider(
                s.llm_provider or "openai_compat",
                s.llm_model_id or "extractor",
                base_url=s.llm_base_url,
                api_key_env=s.llm_api_key_env,
                timeout=s.llm_timeout_seconds,
            )
        )
    res = run(
        args.parquet,
        args.n,
        args.db,
        FastEmbedEmbedder(s.embedding_model),
        args.out,
        sweep=not args.no_sweep,
        offset=args.offset,
        reuse_db=args.reuse_db,
        llm_extractor=llm_extractor,
    )
    print(format_table(res))
    if "chosen_params" in res:
        print(f"\nchosen on dev half: {res['chosen_params']}")
    for metric in ("pair@2", "pair@5", "pair@10"):
        print()
        print(f"{metric} difference vs vector, with 95% bootstrap CI:")
        for name, v in res[f"{metric}_minus_vector_ci95"].items():
            print(f"  {name}: {v['diff']:+.3f}  CI {v['ci95']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=get_settings().log_level)
    p = argparse.ArgumentParser(prog="graphhop")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("ingest", help="ingest .txt/.md files (file name = document title)")
    i.add_argument("collection")
    i.add_argument("paths", nargs="+")
    i.set_defaults(func=_cmd_ingest)

    q = sub.add_parser("query", help="retrieve from a collection")
    q.add_argument("collection")
    q.add_argument("query")
    q.add_argument("--mode", default="hybrid", choices=["vector", "graph", "hybrid"])
    q.add_argument("-k", type=int, default=5)
    q.set_defaults(func=_cmd_query)

    e = sub.add_parser("eval-hotpot", help="retrieval benchmark on HotpotQA (see scripts/)")
    e.add_argument("parquet", help="HotpotQA distractor validation parquet")
    e.add_argument("-n", type=int, default=500, help="number of questions to pool")
    e.add_argument("--db", default="eval_hotpot.db")
    e.add_argument("--out", default="results/hotpot_results.json")
    e.add_argument("--offset", type=int, default=0, help="first question of the slice")
    e.add_argument("--reuse-db", action="store_true", help="reuse an existing index at --db")
    e.add_argument("--no-sweep", action="store_true")
    e.add_argument(
        "--llm-extractor",
        action="store_true",
        help="add LLM triples at ingestion (needs GH_LLM_PROVIDER / GH_LLM_BASE_URL / GH_LLM_MODEL_ID)",
    )
    e.set_defaults(func=_cmd_eval)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
