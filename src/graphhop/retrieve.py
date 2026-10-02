"""Hybrid retrieval: dense vector search fused with knowledge-graph traversal.

Channels
  dense : cosine similarity between the query and chunk embeddings.
  graph : seed entities come from (a) entities linked in the query and (b, hybrid only) entities in
          the top dense chunks (pseudo-relevance feedback). Personalised PageRank spreads mass
          from the seeds over the entity graph; a chunk's graph score is the PageRank mass of the
          entities it mentions. This is what lets a chunk that shares no words with the question
          (the second hop) rank highly because it is linked to the first-hop chunk.
Fusion: weighted reciprocal-rank fusion, so the two score scales never have to be compared.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from .db import Database
from .embeddings import Embedder, from_blob
from .extract import Extractor, normalise_key


@dataclass(frozen=True)
class RetrievalParams:
    k: int = 5
    candidates_per_channel: int = 50
    max_hops: int = 3
    ppr_alpha: float = 0.85
    max_subgraph_nodes: int = 3000
    graph_strategy: str = "expand"  # "expand" (graph proposes, dense ranks) | "ppr"
    expand_hops: int = 1
    title_only: bool = True  # expand to articles ABOUT nearby entities (needs title_as_entity)
    seed_from_dense_top_n: int = 1
    dense_seed_weight: float = 0.5
    max_entity_df_fraction: float = 0.2  # entities in more than this share of documents are hubs
    hub_min_documents: int = 50  # hub pruning only applies to collections at least this large
    w_dense: float = 1.0
    w_graph: float = 1.0
    rrf_k: int = 60


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    external_id: str
    title: str
    text: str
    score: float
    dense_rank: int | None = None
    graph_rank: int | None = None


@dataclass
class Trace:
    mode: str
    query_seeds: list[str] = field(default_factory=list)
    dense_seed_entities: list[str] = field(default_factory=list)
    top_entities: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class _Graph:
    version: int
    g: nx.Graph
    names: dict[int, str]
    key_to_id: dict[str, int]
    suffix_to_ids: dict[str, list[int]]
    chunk_entities: dict[int, list[int]]
    entity_chunks: dict[int, list[int]]
    hubs: set[int]
    chunk_title_entity: dict[int, int | None]
    chunk_ids: list[int]
    row_of: dict[int, int]
    mat: NDArray[np.float32]
    chunk_meta: dict[int, tuple[int, str, str, str]]  # chunk -> (doc id, external id, title, text)


class Retriever:
    def __init__(self, db: Database, embedder: Embedder, extractor: Extractor) -> None:
        self._db = db
        self._embedder = embedder
        self._extractor = extractor
        self._lock = threading.Lock()
        self._cache: dict[int, _Graph] = {}

    # ----- loading --------------------------------------------------------------------------
    def _load(self, collection_id: int, version: int, params: RetrievalParams) -> _Graph:
        with self._lock:
            c = self._cache.get(collection_id)
            if c and c.version == version:
                return c
        conn: sqlite3.Connection = self._db.conn
        names = {
            r["id"]: r["name"]
            for r in conn.execute(
                "SELECT id, name FROM entities WHERE collection_id=?", (collection_id,)
            )
        }
        key_to_id = {
            r["key"]: r["id"]
            for r in conn.execute(
                "SELECT id, key FROM entities WHERE collection_id=?", (collection_id,)
            )
        }
        suffix: dict[str, list[int]] = defaultdict(list)
        for key, eid in key_to_id.items():
            parts = key.split()
            if len(parts) > 1:
                suffix[parts[-1]].append(eid)
        chunk_entities: dict[int, list[int]] = defaultdict(list)
        entity_chunks: dict[int, list[int]] = defaultdict(list)
        docs_of: dict[int, set[int]] = defaultdict(set)
        for r in conn.execute(
            "SELECT m.entity_id, m.chunk_id, c.document_id FROM mentions m "
            "JOIN chunks c ON c.id = m.chunk_id WHERE c.collection_id=?",
            (collection_id,),
        ):
            chunk_entities[r["chunk_id"]].append(r["entity_id"])
            entity_chunks[r["entity_id"]].append(r["chunk_id"])
            docs_of[r["entity_id"]].add(r["document_id"])
        n_docs = conn.execute(
            "SELECT COUNT(*) n FROM documents WHERE collection_id=?", (collection_id,)
        ).fetchone()["n"]
        hubs: set[int] = set()
        if n_docs >= params.hub_min_documents:
            limit = params.max_entity_df_fraction * n_docs
            hubs = {e for e, d in docs_of.items() if len(d) > limit}
        g: nx.Graph = nx.Graph()
        for r in conn.execute(
            "SELECT e.a, e.b, SUM(e.weight) w FROM edge_contrib e JOIN entities x ON x.id = e.a "
            "WHERE x.collection_id=? GROUP BY e.a, e.b",
            (collection_id,),
        ):
            if r["a"] in hubs or r["b"] in hubs:
                continue
            g.add_edge(r["a"], r["b"], weight=1.0 + math.log(r["w"]))
        rows = conn.execute(
            "SELECT c.id, c.document_id, c.text, c.embedding, d.external_id, d.title, d.title_entity FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE c.collection_id=? ORDER BY c.id",
            (collection_id,),
        ).fetchall()
        mat = (
            np.stack([from_blob(r["embedding"]) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        built = _Graph(
            version,
            g,
            names,
            key_to_id,
            dict(suffix),
            dict(chunk_entities),
            dict(entity_chunks),
            hubs,
            {r["id"]: r["title_entity"] for r in rows},
            [r["id"] for r in rows],
            {r["id"]: i for i, r in enumerate(rows)},
            mat,
            {r["id"]: (r["document_id"], r["external_id"], r["title"], r["text"]) for r in rows},
        )
        with self._lock:
            self._cache[collection_id] = built
        return built

    # ----- entity linking -----------------------------------------------------------------------
    def link_query(self, query: str, G: _Graph) -> list[int]:
        """Link entity mentions in the query to graph entities (exact key, then a unique
        last-name style match, then shorter sub-spans)."""
        found: list[int] = []

        def add(eid: int | None) -> None:
            if eid is not None and eid not in found and eid not in G.hubs:
                found.append(eid)

        for sent in self._extractor.extract(query).sentences:
            for name in sent.entities:
                key = normalise_key(name)
                if key in G.key_to_id:
                    add(G.key_to_id[key])
                    continue
                words = key.split()
                if len(words) == 1 and len(G.suffix_to_ids.get(key, [])) == 1:
                    add(G.suffix_to_ids[key][0])
                    continue
                for size in range(len(words) - 1, 0, -1):  # longest sub-span that is an entity
                    hit = next(
                        (
                            G.key_to_id[" ".join(words[i : i + size])]
                            for i in range(len(words) - size + 1)
                            if " ".join(words[i : i + size]) in G.key_to_id
                        ),
                        None,
                    )
                    if hit is not None:
                        add(hit)
                        break
        return found

    # ----- channels ----------------------------------------------------------------------------
    def _dense(self, G: _Graph, qvec: NDArray[np.float32], n: int) -> list[tuple[int, float]]:
        if len(G.chunk_ids) == 0:
            return []
        scores = G.mat @ qvec
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return [(G.chunk_ids[i], float(scores[i])) for i in top]

    def _bfs_nodes(self, g: nx.Graph, seeds: list[int], hops: int, cap: int) -> set[int]:
        seen = set(s for s in seeds if s in g)
        frontier = deque((s, 0) for s in seen)
        while frontier and len(seen) < cap:
            node, d = frontier.popleft()
            if d >= hops:
                continue
            for nb in g.neighbors(node):
                if nb not in seen:
                    seen.add(nb)
                    frontier.append((nb, d + 1))
                    if len(seen) >= cap:
                        break
        return seen

    def _graph_channel(
        self, G: _Graph, personalization: dict[int, float], p: RetrievalParams, n: int
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        seeds = [e for e in personalization if e in G.g]
        if not seeds:
            return [], []
        nodes = self._bfs_nodes(G.g, seeds, p.max_hops, p.max_subgraph_nodes)
        sub = G.g.subgraph(nodes)
        pers = {e: personalization[e] for e in seeds}
        total = sum(pers.values())
        pers = {e: w / total for e, w in pers.items()}
        ppr = nx.pagerank(sub, alpha=p.ppr_alpha, personalization=pers, weight="weight")
        scores: dict[int, float] = defaultdict(float)
        for e, mass in ppr.items():
            for ch in G.entity_chunks.get(e, ()):
                scores[ch] += mass
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
        top_entities = sorted(ppr.items(), key=lambda kv: -kv[1])[:15]
        return ranked, top_entities

    def _expand_channel(
        self,
        G: _Graph,
        seeds: list[int],
        qvec: NDArray[np.float32],
        p: RetrievalParams,
        n: int,
    ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
        """Graph proposes, dense ranks.

        Candidates are the chunks of articles *about* entities within ``expand_hops`` of the seed
        entities (or, with ``title_only=False`` or in collections without title entities, every chunk
        that mentions such an entity). Candidates are then ordered by their similarity to the query.
        This keeps the graph's reachability (the second-hop paragraph shares no words with the
        question but is one link away) while avoiding the weak discrimination of spreading-activation
        scores over hundreds of neighbours.
        """
        near: dict[int, float] = {}
        for s in seeds:
            if s in G.g:
                near[s] = 1.0
        frontier = list(near)
        for hop in range(p.expand_hops):
            nxt: list[int] = []
            for node in frontier:
                for nb in G.g.neighbors(node):
                    if nb not in near and nb not in G.hubs:
                        near[nb] = 1.0 / (hop + 2)
                        nxt.append(nb)
            frontier = nxt
        if not near:
            return [], []
        has_titles = any(t is not None for t in G.chunk_title_entity.values())
        title_only = p.title_only and has_titles
        cands: set[int] = set()
        for e in near:
            for ch in G.entity_chunks.get(e, ()):
                if not title_only or G.chunk_title_entity.get(ch) == e:
                    cands.add(ch)
        if not cands:
            return [], []
        row_of = G.row_of
        sims = G.mat @ qvec
        ranked = sorted(((c, float(sims[row_of[c]])) for c in cands), key=lambda kv: -kv[1])[:n]
        top_entities = sorted(near.items(), key=lambda kv: -kv[1])[:15]
        return ranked, top_entities

    # ----- public API -------------------------------------------------------------------------------
    def retrieve(
        self, collection_id: int, version: int, query: str, mode: str, p: RetrievalParams
    ) -> tuple[list[RetrievedChunk], Trace]:
        if mode not in ("vector", "graph", "hybrid"):
            raise ValueError("mode must be 'vector', 'graph' or 'hybrid'")
        G = self._load(collection_id, version, p)
        qvec = self._embedder.embed([query])[0]
        trace = Trace(mode)
        n = p.candidates_per_channel
        dense = self._dense(G, qvec, n) if mode in ("vector", "hybrid") else []

        graph_ranked: list[tuple[int, float]] = []
        if mode in ("graph", "hybrid"):
            seeds = self.link_query(query, G)
            trace.query_seeds = [G.names[e] for e in seeds]
            pers: dict[int, float] = {e: 1.0 for e in seeds}
            if mode == "hybrid":
                for rank, (chunk_id, _) in enumerate(dense[: p.seed_from_dense_top_n]):
                    ents = [e for e in G.chunk_entities.get(chunk_id, ()) if e not in G.hubs]
                    for e in ents:
                        pers[e] = pers.get(e, 0.0) + p.dense_seed_weight / (rank + 1) / len(ents)
                trace.dense_seed_entities = [
                    G.names[e] for e in pers if G.names[e] not in trace.query_seeds
                ][:15]
            if p.graph_strategy == "ppr":
                graph_ranked, top = self._graph_channel(G, pers, p, n)
            elif p.graph_strategy == "expand":
                graph_ranked, top = self._expand_channel(G, list(pers), qvec, p, n)
            else:
                raise ValueError("graph_strategy must be 'expand' or 'ppr'")
            trace.top_entities = [(G.names[e], round(s, 5)) for e, s in top]

        fused: dict[int, float] = defaultdict(float)
        dense_rank = {cid: r for r, (cid, _) in enumerate(dense)}
        graph_rank = {cid: r for r, (cid, _) in enumerate(graph_ranked)}
        if mode == "vector":
            order = [(cid, s) for cid, s in dense]
        elif mode == "graph":
            order = list(graph_ranked)
        else:
            for cid, r in dense_rank.items():
                fused[cid] += p.w_dense / (p.rrf_k + r + 1)
            for cid, r in graph_rank.items():
                fused[cid] += p.w_graph / (p.rrf_k + r + 1)
            order = sorted(fused.items(), key=lambda kv: -kv[1])
        out: list[RetrievedChunk] = []
        for cid, score in order[: p.k]:
            doc_id, ext, title, text = G.chunk_meta[cid]
            out.append(
                RetrievedChunk(
                    cid, doc_id, ext, title, text, score, dense_rank.get(cid), graph_rank.get(cid)
                )
            )
        return out, trace
