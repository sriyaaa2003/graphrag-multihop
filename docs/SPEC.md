# graphhop: technical specification

Items marked **REQUIRES USER DECISION** need an input that cannot be derived from the requirements.

## 1. Purpose and boundaries

**Does:** retrieve evidence for questions whose answer is spread across several documents ("multi-hop"),
by combining dense vector retrieval with traversal of an entity graph built from the corpus; optionally
generate a cited answer; explain retrieval through a trace.

**Why:** plain RAG ranks passages by similarity to the question. When the answer needs a second passage that
shares no words with the question (the passage about the *bridge* entity), similarity cannot find it. A graph
that links passages through the entities they mention can.

**Does not:** train models; resolve coreference; type entities or label relations without an LLM; provide
permissions or multi-tenancy (see the sibling `permission-aware-rag`); guarantee that the graph helps on
every corpus (see Evaluation: it is measured, not assumed).

## 2. Requirements

| # | Requirement | Priority | Verified by |
|---|---|---|---|
| F1 | Ingest documents idempotently, building chunks, entities, mentions and edges in one transaction | must | `test_graph_retrieval` |
| F2 | Replacing or deleting a document removes exactly its effect on the graph | must | `test_graph_retrieval` |
| F3 | Three retrieval modes: `vector`, `graph`, `hybrid`, with a retrieval trace | must | `test_graph_retrieval` |
| F4 | Hybrid retrieval finds a second-hop passage that vector search misses | must | `test_graph_channel_finds_...` |
| F5 | Entity linking of the query (exact, unique last-name style, sub-span) | must | `test_unique_last_name_...` |
| F6 | Hub entities pruned in large collections | should | `test_hub_entities_...` |
| F7 | Pluggable extractor: deterministic proper-noun extractor and a validated LLM triple extractor | should | `test_extract` |
| F8 | Optional generation with citation validation and abstention | should | `test_graph_retrieval` |
| F9 | Reproducible benchmark against a vector baseline on a public multi-hop dataset | must | `graphhop eval-hotpot` |
| N1 | No fabricated results; every reported number comes from a run recorded in `results/` | must | process |
| N2 | Typed, linted, tested | must | CI |

## 3. Architecture

Modular monolith: `extract.py` → `graphstore.py` (ingest) → SQLite; `retrieve.py` (load graph, link, rank, fuse)
→ `service.py` (modes, generation) → `api/app.py`. Evaluation lives in `eval_hotpot.py`.

### Ingestion

chunk (sentence-aware) → embed → extract per chunk → in **one transaction**: upsert document, insert chunks,
upsert entities (unique per collection by normalised key), insert `mentions`, accumulate edge weights per
document into `edge_contrib`, delete orphaned entities, bump `graph_version`.

* **Entity identity** = normalised surface form (lower-cased, article and possessive stripped, punctuation
  removed). Two surface forms that normalise equal are one entity. Different entities that share a name merge:
  a known limitation of string-based resolution (ADR-3).
* **Title as entity.** The document title becomes an entity attached to every chunk of the document and linked to
  every entity the document mentions. This models encyclopedic structure (an article *is about* its title) and is
  switchable (`GH_TITLE_AS_ENTITY`).
* **Edges** are co-occurrence within a sentence (weight = count) plus title-to-mention links. The LLM extractor
  additionally supplies typed relations stored in `edge_contrib.relation`.
* Edge weight in the graph = `1 + ln(Σ weight over documents)`, derived at load time from `edge_contrib` (so there is
  no second copy of the truth to keep consistent).

### Retrieval

1. **Dense channel**: cosine over chunk embeddings → ranked list (top `candidates_per_channel`).
2. **Graph channel**, two strategies behind `graph_strategy`:
   * **`expand` (default): the graph proposes, dense ranks.** Seeds are (a) entities linked in the query and
     (b, hybrid only) entities in the top-`seed_from_dense_top_n` dense chunks (pseudo-relevance feedback).
     Expand `expand_hops` steps through the entity graph (hubs excluded). Candidates are the chunks of
     articles *about* the entities reached (the chunk's document title entity is in the reached set); in
     collections without title entities, every chunk mentioning a reached entity. Candidates are ordered by
     similarity to the query. This keeps the graph's reachability (the second-hop passage shares no words with
     the question but is one link away) without relying on spreading-activation scores to pick the right
     neighbour out of hundreds.
   * **`ppr`**: personalised PageRank from the same seeds over a BFS subgraph (`max_hops`, `ppr_alpha`); a
     chunk's score is the PageRank mass of the entities it mentions. This was the first version; it is kept
     because it is a reasonable choice for corpora without article structure, and because the benchmark
     compares the two.
3. **Fusion**: weighted reciprocal-rank fusion `Σ w_c / (rrf_k + rank_c)`. RRF avoids comparing cosine scores with
   PageRank mass.
4. **Hub pruning**: entities mentioned in more than `max_entity_df_fraction` of documents are removed from the graph
   in collections of at least `hub_min_documents` (they connect everything to everything).
5. **Trace**: query entities, dense-seed entities and top PageRank entities are returned for explainability.

## 4. Data model (migrations in `src/graphhop/migrations`)

`collections(graph_version)` · `documents(content_hash, title_entity)` · `chunks(text, embedding)` ·
`entities(collection_id, key UNIQUE, name)` · `mentions(entity_id, chunk_id)` ·
`edge_contrib(document_id, a, b, relation, weight)` with `a < b`. Deleting a document cascades through chunks,
mentions and its edge contributions.

## 5. Configuration

Environment variables, prefix `GH_` (see `.env.example`); retrieval parameters can additionally be overridden
per request (`k`, `w_graph`, `w_dense`, `max_hops`). Changing environment values needs a restart. Provider keys are
referenced by variable name, never stored.

## 6. Failure matrix

| Component | Failure | Behaviour |
|---|---|---|
| Extractor (LLM) | provider error or malformed/invalid JSON | `ExtractionError`; the document is **not** ingested (transaction not started) |
| Extractor (proper-noun) | no entities in text | document still indexed for dense retrieval; graph channel simply has nothing to add |
| Query | no entity found in graph | hybrid falls back to dense seeds; graph-only returns nothing |
| Graph | stale after ingest | cache keyed on `graph_version`; rebuilt on next query |
| Large hub entity | floods PageRank | pruned (large collections) |
| Database | locked | WAL + busy timeout; writes are single transactions |
| LLM generation | error | 502; invalid citations → 502; abstention returned as such |
| Restart | cache lost | rebuilt lazily from SQLite |
| Concurrent ingest and query | query sees old or new `graph_version`, never a mix | per-thread connections, versioned cache |

## 7. Security

Admin key (`X-API-Key`, constant-time compare) protects ingestion/deletion; query is open unless fronted by a
gateway. Passage text is HTML-escaped in prompts and labelled untrusted; the model has no tools. Input limits on
request sizes. No per-user permissions: do not index content that some callers must not see.

## 8. Evaluation methodology (F9)

* **Data:** HotpotQA, distractor setting, validation split. The first *N* questions' paragraphs are pooled and
  de-duplicated by title, so the retriever must find the supporting paragraphs among thousands rather than ten.
* **Gold:** supporting-fact titles (normally two paragraphs).
* **Metrics:** SP-Recall@k (fraction of gold paragraphs in the top k) and Pair@k (all gold paragraphs in the top k),
  k ∈ {2, 5, 10}; broken down by question type (bridge, comparison); paired bootstrap 95% CI for hybrid − vector on Pair@5.
* **Protocol:** questions split into dev (even index) and test (odd index). A grid over both graph strategies
  (`graph_strategy`, `w_graph`, `seed_from_dense_top_n`, hops) is searched **on dev only**; the chosen setting and the
  untuned defaults are both reported on **test**, next to the first (PageRank) version.
* **Two slices.** Questions 0 to 500 were used to find the weakness of the first version and to design `expand`
  (diagnostics looked at that slice), so they are *not* a clean test of `expand`. Questions 500 to 1000 were not looked
  at before the design was fixed and are the independent confirmation. Both are reported. Vector retrieval uses the same embeddings and chunks, so the comparison
  isolates the graph channel.
* **Not measured:** end-to-end answer accuracy (EM/F1) needs an LLM and was not run; retrieval metrics are an upper
  bound on what an answerer can use.
* **Threats to validity:** pooled-paragraph corpus is easier than open-domain retrieval; one dataset; one
  embedding model; Wikipedia-style text suits the proper-noun extractor and title-as-entity assumption.

## 9. Architecture decision records

**ADR-1 SQLite + networkx.** Alternatives: Neo4j, a vector DB with graph features. Chosen for zero infrastructure;
the graph is loaded in memory per collection version. Consequence: bounded by RAM; the loader is the seam.

**ADR-2 Rank fusion over score fusion.** Dense cosine and PageRank mass are incomparable; RRF needs only ranks.

**ADR-3 Deterministic string-based extraction and resolution by default.** Alternatives: spaCy NER (a compiled
dependency that could not be loaded in the development environment because of an application-control policy), LLM
extraction (cost, nondeterminism). Chosen: a dependency-free proper-noun extractor; LLM extraction is available behind
the same interface with schema validation. Consequence: untyped entities, co-occurrence relations, name collisions
merge.

**ADR-4 Edge weights derived from per-document contributions.** Makes replace/delete exact without recomputing the
whole graph.

**ADR-5 "Graph proposes, dense ranks" as the default graph channel.** The first version used personalised PageRank
over the entity graph. Diagnostics on the benchmark showed the gold paragraphs were reachable (196 of 198 bridge
questions in the test half) yet ranked poorly, because PageRank mass cannot tell which of hundreds of neighbours
matters. Using the graph only for candidate generation and the dense score for ranking fixed most of that. PageRank
remains available as `graph_strategy=ppr`.

**ADR-6 Parameters tuned on a dev split only.** Prevents reporting a result that was tuned on the data it is
reported on.

## 10. Definition of done and status

Implemented and tested: F1–F9, N1–N2. **Not implemented/verified:** end-to-end answer accuracy, live LLM extraction
or generation, typed relations without an LLM, coreference, large-scale (millions of chunks) behaviour,
multi-language extraction.
