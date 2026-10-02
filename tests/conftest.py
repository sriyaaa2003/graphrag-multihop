"""Test doubles live here only. Production code never imports from tests."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from graphhop.config import Settings
from graphhop.db import Database
from graphhop.embeddings import normalise
from graphhop.extract import ProperNounExtractor
from graphhop.llm import LLMResult
from graphhop.service import GraphRagService


class HashEmbedder:
    """Deterministic bag-of-words embedder. Test-only."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1.0  # noqa: S324
        return normalise(out)


class FakeLLM:
    def __init__(self, reply: str = "It is in the passages [1] [2].") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> LLMResult:
        self.prompts.append(user)
        return LLMResult(self.reply, 10, 5, 1.0)


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "t.db"), admin_api_key="admin-key")


@pytest.fixture
def db(settings):
    return Database(settings.db_path)


@pytest.fixture
def svc(db, settings):
    return GraphRagService(db, HashEmbedder(), ProperNounExtractor(), settings, FakeLLM())
