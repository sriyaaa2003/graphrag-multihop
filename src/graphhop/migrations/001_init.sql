CREATE TABLE collections (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    graph_version  INTEGER NOT NULL DEFAULT 1,
    created_at     REAL NOT NULL
);

CREATE TABLE documents (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    external_id   TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL,
    title_entity  INTEGER,
    created_at    REAL NOT NULL,
    UNIQUE (collection_id, external_id)
);

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX idx_chunks_collection ON chunks(collection_id);

-- One entity per normalised surface form within a collection.
CREATE TABLE entities (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    key           TEXT NOT NULL,
    name          TEXT NOT NULL,
    UNIQUE (collection_id, key)
);

CREATE TABLE mentions (
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    chunk_id  INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, chunk_id)
);
CREATE INDEX idx_mentions_chunk ON mentions(chunk_id);

-- Per-document edge contributions. The graph's edge weights are SUM(weight) over documents, so
-- replacing or deleting one document removes exactly its contribution.
CREATE TABLE edge_contrib (
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    a           INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    b           INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL DEFAULT 'cooccurs',
    weight      REAL NOT NULL,
    CHECK (a < b),
    PRIMARY KEY (document_id, a, b, relation)
);
CREATE INDEX idx_edge_contrib_ab ON edge_contrib(a, b);
