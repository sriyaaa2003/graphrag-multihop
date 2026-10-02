"""Ingestion: chunk -> embed -> extract entities -> store chunks, mentions and edge contributions.

All writes for one document happen in one transaction. Edge weights are derived from per-document
contributions, so re-ingesting or deleting a document exactly undoes its effect on the graph.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from .chunking import chunk_text
from .db import transaction
from .embeddings import Embedder, to_blob
from .extract import Extraction, Extractor, normalise_key


@dataclass(frozen=True)
class IngestParams:
    chunk_chars: int = 800
    overlap: int = 100
    title_as_entity: bool = True


def _collection(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO collections(name, graph_version, created_at) VALUES (?,1,?)",
        (name, time.time()),
    )
    return int(cur.lastrowid or 0)


def _entity(
    conn: sqlite3.Connection, collection_id: int, name: str, cache: dict[str, int]
) -> int | None:
    key = normalise_key(name)
    if not key:
        return None
    if key in cache:
        return cache[key]
    row = conn.execute(
        "SELECT id FROM entities WHERE collection_id=? AND key=?", (collection_id, key)
    ).fetchone()
    if row:
        eid = int(row["id"])
    else:
        eid = int(
            conn.execute(
                "INSERT INTO entities(collection_id, key, name) VALUES (?,?,?)",
                (collection_id, key, name),
            ).lastrowid
            or 0
        )
    cache[key] = eid
    return eid


def ingest_document(
    conn: sqlite3.Connection,
    embedder: Embedder,
    extractor: Extractor,
    *,
    collection: str,
    external_id: str,
    title: str,
    text: str,
    params: IngestParams | None = None,
) -> bool:
    """Idempotent upsert. Returns False when the content is unchanged."""
    params = params or IngestParams()
    pieces = chunk_text(text, params.chunk_chars, params.overlap)
    if not pieces:
        raise ValueError("document has no content")
    content_hash = hashlib.sha256((title + "\n" + text).encode()).hexdigest()
    vectors = embedder.embed(pieces)
    extractions: list[Extraction] = [extractor.extract(p) for p in pieces]  # may call an LLM

    with transaction(conn):
        cid = _collection(conn, collection)
        existing = conn.execute(
            "SELECT id, content_hash FROM documents WHERE collection_id=? AND external_id=?",
            (cid, external_id),
        ).fetchone()
        if existing and existing["content_hash"] == content_hash:
            return False
        if existing:
            conn.execute("DELETE FROM documents WHERE id=?", (existing["id"],))

        cache: dict[str, int] = {}
        title_eid = _entity(conn, cid, title, cache) if (params.title_as_entity and title) else None
        doc_id = int(
            conn.execute(
                "INSERT INTO documents(collection_id, external_id, title, content_hash, "
                "title_entity, created_at) VALUES (?,?,?,?,?,?)",
                (cid, external_id, title, content_hash, title_eid, time.time()),
            ).lastrowid
            or 0
        )
        contrib: dict[tuple[int, int, str], float] = defaultdict(float)

        def add_edge(x: int, y: int, relation: str, w: float) -> None:
            if x != y:
                a, b = (x, y) if x < y else (y, x)
                contrib[(a, b, relation)] += w

        for ordinal, (piece, vec, ex) in enumerate(zip(pieces, vectors, extractions, strict=True)):
            chunk_id = int(
                conn.execute(
                    "INSERT INTO chunks(document_id, collection_id, ordinal, text, embedding) "
                    "VALUES (?,?,?,?,?)",
                    (doc_id, cid, ordinal, piece, to_blob(vec)),
                ).lastrowid
                or 0
            )
            chunk_entities: set[int] = set()
            for sent in ex.sentences:
                ids = [e for e in (_entity(conn, cid, n, cache) for n in sent.entities) if e]
                chunk_entities.update(ids)
                for x, y in combinations(sorted(set(ids)), 2):
                    add_edge(x, y, "cooccurs", 1.0)
            for subj, rel, obj in ex.triples:
                s, o = _entity(conn, cid, subj, cache), _entity(conn, cid, obj, cache)
                if s and o:
                    chunk_entities.update((s, o))
                    add_edge(s, o, rel[:100], 1.0)
            if title_eid:
                chunk_entities.add(title_eid)
                for e in chunk_entities - {title_eid}:
                    add_edge(title_eid, e, "described_in", 1.0)
            conn.executemany(
                "INSERT OR IGNORE INTO mentions(entity_id, chunk_id) VALUES (?,?)",
                [(e, chunk_id) for e in chunk_entities],
            )
        conn.executemany(
            "INSERT INTO edge_contrib(document_id, a, b, relation, weight) VALUES (?,?,?,?,?)",
            [(doc_id, a, b, rel, w) for (a, b, rel), w in contrib.items()],
        )
        conn.execute(
            "DELETE FROM entities WHERE collection_id=? AND id NOT IN (SELECT entity_id FROM mentions)",
            (cid,),
        )
        conn.execute("UPDATE collections SET graph_version = graph_version + 1 WHERE id=?", (cid,))
    return True


def delete_document(conn: sqlite3.Connection, collection: str, external_id: str) -> bool:
    with transaction(conn):
        row = conn.execute(
            "SELECT d.id, d.collection_id FROM documents d JOIN collections c ON c.id=d.collection_id "
            "WHERE c.name=? AND d.external_id=?",
            (collection, external_id),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM documents WHERE id=?", (row["id"],))
        conn.execute(
            "DELETE FROM entities WHERE collection_id=? AND id NOT IN (SELECT entity_id FROM mentions)",
            (row["collection_id"],),
        )
        conn.execute(
            "UPDATE collections SET graph_version = graph_version + 1 WHERE id=?",
            (row["collection_id"],),
        )
    return True
