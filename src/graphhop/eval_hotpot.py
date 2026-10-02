"""Retrieval evaluation on HotpotQA (distractor setting), comparing vector, graph and hybrid.

Corpus : every paragraph from the first N questions, pooled and de-duplicated by title (so the
         retriever must find the right paragraphs among thousands, not among 10 candidates).
Gold   : the titles of each question's supporting paragraphs (usually 2).
Metrics: SP-Recall@k  = mean fraction of gold paragraphs found in the top k.
         Pair@k       = share of questions whose gold paragraphs are *all* in the top k (the
                        quantity that multi-hop answering actually needs).
No LLM is involved, so every number is measured, not estimated. Hyper-parameters are chosen on the
dev half of the questions and reported on the held-out test half; defaults are reported as well.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import graphstore
from .db import Database
from .embeddings import Embedder
from .extract import ProperNounExtractor
from .retrieve import RetrievalParams, Retriever

KS = (2, 5, 10)


class CachedEmbedder:
    """Embeds unique texts once, in large batches, then serves them from memory."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.cache: dict[str, Any] = {}

    def prime(self, texts: list[str], batch: int = 128) -> None:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for i in range(0, len(todo), batch):
            part = todo[i : i + batch]
            for t, v in zip(part, self.inner.embed(part), strict=True):
                self.cache[t] = v

    def embed(self, texts: list[str]) -> NDArray[Any]:
        missing = [t for t in texts if t not in self.cache]
        if missing:
            self.prime(missing)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 0), np.float32)


def load_hotpot(
    parquet_path: str, n: int, offset: int = 0
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).slice(offset, n).to_pylist()
    paragraphs: dict[str, str] = {}
    questions = []
    for r in rows:
        ctx = r["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"], strict=True):
            paragraphs.setdefault(title, " ".join(s.strip() for s in sents))
        questions.append(
            {
                "id": r["id"],
                "question": r["question"],
                "type": r["type"],
                "gold": sorted(set(r["supporting_facts"]["title"])),
            }
        )
    return questions, paragraphs


def build_index(db: Database, embedder: Embedder, paragraphs: dict[str, str]) -> float:
    extractor = ProperNounExtractor()
    params = graphstore.IngestParams(chunk_chars=100_000, overlap=0, title_as_entity=True)
    t0 = time.perf_counter()
    for title, text in paragraphs.items():
        if text.strip():
            graphstore.ingest_document(
                db.conn,
                embedder,
                extractor,
                collection="hotpot",
                external_id=title,
                title=title,
                text=text,
                params=params,
            )
    return time.perf_counter() - t0


def score(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    cid: int,
    version: int,
    mode: str,
    p: RetrievalParams,
) -> dict[str, Any]:
    kmax = max(KS)
    sp = {k: np.zeros(len(questions)) for k in KS}
    pair = {k: np.zeros(len(questions)) for k in KS}
    for i, q in enumerate(questions):
        chunks, _ = retriever.retrieve(cid, version, q["question"], mode, replace(p, k=kmax))
        ranked = [c.external_id for c in chunks]
        gold = set(q["gold"])
        for k in KS:
            hit = len(gold & set(ranked[:k]))
            sp[k][i] = hit / len(gold)
            pair[k][i] = float(hit == len(gold))
    out = {f"sp@{k}": sp[k] for k in KS}
    out.update({f"pair@{k}": pair[k] for k in KS})
    return out


def _summary(scores: dict[str, Any], mask: NDArray[Any] | None = None) -> dict[str, float]:
    return {m: float(v[mask].mean() if mask is not None else v.mean()) for m, v in scores.items()}


def bootstrap_diff(
    a: NDArray[Any], b: NDArray[Any], n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap CI of mean(a - b) over paired questions."""
    rng = np.random.default_rng(seed)
    d = a - b
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _candidates(default: RetrievalParams) -> list[RetrievalParams]:
    """The search space for dev tuning. Both graph strategies are searched, so a strategy is only
    'chosen' if it wins on the dev half."""
    grid: list[RetrievalParams] = []
    for w_graph, seeds in product((0.5, 1.0, 2.0), (1, 3)):
        base = replace(default, w_graph=w_graph, seed_from_dense_top_n=seeds)
        for hops in (2, 3):
            grid.append(replace(base, graph_strategy="ppr", max_hops=hops))
        for hops in (1, 2):
            grid.append(replace(base, graph_strategy="expand", expand_hops=hops))
    return grid


def _describe(p: RetrievalParams) -> dict[str, Any]:
    d: dict[str, Any] = {
        "graph_strategy": p.graph_strategy,
        "w_graph": p.w_graph,
        "seed_from_dense_top_n": p.seed_from_dense_top_n,
    }
    d["max_hops" if p.graph_strategy == "ppr" else "expand_hops"] = (
        p.max_hops if p.graph_strategy == "ppr" else p.expand_hops
    )
    return d


def run(
    parquet_path: str,
    n: int,
    db_path: str,
    embedder: Embedder,
    out_path: str | None = None,
    sweep: bool = True,
    offset: int = 0,
    reuse_db: bool = False,
) -> dict[str, Any]:
    questions, paragraphs = load_hotpot(parquet_path, n, offset)
    cached = CachedEmbedder(embedder)
    reuse = reuse_db and Path(db_path).exists()
    t0 = time.perf_counter()
    to_embed = [q["question"] for q in questions]
    if not reuse:
        to_embed += list(paragraphs.values())
    cached.prime(to_embed)
    embed_s = time.perf_counter() - t0

    if reuse:
        db = Database(db_path)
        ingest_s = 0.0
    else:
        if Path(db_path).exists():
            Path(db_path).unlink()
        db = Database(db_path)
        ingest_s = build_index(db, cached, paragraphs)
    row = db.conn.execute("SELECT id, graph_version FROM collections").fetchone()
    cid, version = int(row["id"]), int(row["graph_version"])
    retriever = Retriever(db, cached, ProperNounExtractor())

    idx = np.arange(len(questions))
    dev, test = idx % 2 == 0, idx % 2 == 1
    default = RetrievalParams()
    legacy_ppr = replace(
        default, graph_strategy="ppr", seed_from_dense_top_n=3, max_hops=3, w_graph=1.0
    )
    results: dict[str, Any] = {
        "slice": {"offset": offset, "n": len(questions)},
        "n_questions": len(questions),
        "n_paragraphs": len(paragraphs),
        "embed_seconds": round(embed_s, 1),
        "ingest_seconds": round(ingest_s, 1),
        "reused_index": reuse,
        "defaults": _describe(default),
    }

    chosen = default
    if sweep:
        dev_questions = [q for q, m in zip(questions, dev, strict=True) if m]
        grid = []
        for cand in _candidates(default):
            s = score(retriever, dev_questions, cid, version, "hybrid", cand)
            grid.append((float(s["pair@5"].mean()), cand))
        # ties are broken in favour of the earlier (simpler) candidate; max() keeps the first maximum
        chosen = max(grid, key=lambda t: t[0])[1]
        results["sweep_dev_pair@5"] = [{**_describe(c), "pair@5_dev": round(v, 4)} for v, c in grid]
        results["chosen_params"] = _describe(chosen)

    test_questions = [q for q, m in zip(questions, test, strict=True) if m]
    types = np.array([q["type"] for q in test_questions])
    modes: dict[str, tuple[str, RetrievalParams]] = {
        "vector": ("vector", default),
        "graph only (expand)": ("graph", default),
        "hybrid, PageRank (first version)": ("hybrid", legacy_ppr),
        "hybrid, expand (defaults)": ("hybrid", default),
    }
    if sweep:
        modes["hybrid (tuned on dev)"] = ("hybrid", chosen)
    per_mode: dict[str, dict[str, Any]] = {}
    for name, (mode, p) in modes.items():
        per_mode[name] = score(retriever, test_questions, cid, version, mode, p)
    results["test"] = {
        name: {
            "all": _summary(s),
            "bridge": _summary(s, types == "bridge"),
            "comparison": _summary(s, types == "comparison"),
        }
        for name, s in per_mode.items()
    }
    results["n_test"] = len(test_questions)
    results["test_counts_by_type"] = {t: int((types == t).sum()) for t in ("bridge", "comparison")}
    for metric in ("pair@2", "pair@5", "pair@10"):
        cis = results.setdefault(f"{metric}_minus_vector_ci95", {})
        for name, s in per_mode.items():
            if name.startswith("hybrid"):
                lo, hi = bootstrap_diff(s[metric], per_mode["vector"][metric])
                diff = float(s[metric].mean() - per_mode["vector"][metric].mean())
                cis[name] = {"diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def format_table(results: dict[str, Any]) -> str:
    lines = [
        f"HotpotQA distractor, questions {results['slice']['offset']}-"
        f"{results['slice']['offset'] + results['n_questions']} pooled into "
        f"{results['n_paragraphs']} paragraphs; held-out test half = {results['n_test']} questions",
        "",
        "| Method | SP-Recall@2 | SP-Recall@5 | SP-Recall@10 | Pair@2 | Pair@5 | Pair@10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, groups in results["test"].items():
        m = groups["all"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{m[f'sp@{k}']:.3f}" for k in KS)
            + " | "
            + " | ".join(f"{m[f'pair@{k}']:.3f}" for k in KS)
            + " |"
        )
    return "\n".join(lines)
