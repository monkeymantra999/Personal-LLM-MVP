# engine.py
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import tiktoken
from openai import OpenAI

# Models (override in Streamlit sidebar / env)
ENC_MODEL = "text-embedding-3-large"
LLM_MODEL = os.getenv("REASONING_MODEL", "gpt-4.5-mini")


@dataclass
class Snippet:
    id: str
    pack: str
    weight: float
    text: str
    meta: Dict[str, Any]


class Retriever:
    def __init__(self, canon_path: str):
        # OPENAI_API_KEY must be set in env (Streamlit sidebar sets it)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.snippets: List[Snippet] = self._load_cards(canon_path)
        self.embeddings: Optional[np.ndarray] = None  # lazy init

    def _load_cards(self, path: str) -> List[Snippet]:
        out: List[Snippet] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                card = json.loads(line)

                body: List[str] = []

                def add(label: str, items: Any, field: Optional[str] = None) -> None:
                    if not items:
                        return
                    if isinstance(items, list):
                        if items and isinstance(items[0], dict) and field is not None:
                            vals = [str(i.get(field, "")) for i in items if isinstance(i, dict)]
                            vals = [v for v in vals if v]
                            if vals:
                                body.append(f"{label}: " + " | ".join(vals))
                        else:
                            vals = [str(v) for v in items if v]
                            if vals:
                                body.append(f"{label}: " + " | ".join(vals))

                add("THESES", card.get("theses"), "text")
                add("QUOTES", card.get("quotes"), "text")
                add("COUNTERS", card.get("counters"), "text")
                add("IMPLICATIONS", card.get("implications"))
                add("FALSIFIERS", card.get("falsifiers"))

                # Safe, multi-line f-string (prevents unterminated literal issues)
                header = (
                    f"{card.get('title', '')} — {card.get('author', '')} | "
                    f"{card.get('pack', '')} | {card.get('subtopic', '')}\n"
                )
                text = header + "\n".join([b for b in body if b])

                out.append(
                    Snippet(
                        id=str(card.get("id", "")),
                        pack=str(card.get("pack", "")),
                        weight=float(card.get("weight", 1.2)),
                        text=text,
                        meta={
                            "title": card.get("title", ""),
                            "parent": card.get("parent", ""),
                            "subtopic": card.get("subtopic", ""),
                            "tags": card.get("tags", []),
                        },
                    )
                )
        return out

    def _embed(self, texts: List[str]) -> np.ndarray:
        # OpenAI v1.x Embeddings API
        res = self.client.embeddings.create(model=ENC_MODEL, input=texts)
        arr = np.array([d.embedding for d in res.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norms

    def ensure_index(self) -> None:
        if self.embeddings is None:
            self.embeddings = self._embed([s.text for s in self.snippets])

    def retrieve(self, query: str, top_k: int = 12) -> List[Tuple[Snippet, float]]:
        self.ensure_index()
        q = self._embed([query])[0]  # (dim,)
        sims = (self.embeddings @ q).flatten()  # cosine similarities (because normalized)
        weights = np.array([s.weight for s in self.snippets], dtype=np.float32)
        scores = sims * weights
        idx = np.argsort(-scores)[:top_k]
        return [(self.snippets[i], float(scores[i])) for i in idx]


def build_context(
    snippets: List[Tuple[Snippet, float]],
    extra_docs: Optional[List[Dict[str, str]]] = None,
) -> str:
    blocks: List[str] = []
    for s, _score in snippets:
        blocks.append(
            f"[source_id:{s.id}] PACK:{s.pack} TITLE:{s.meta.get('title','')} "
            f"SUB:{s.meta.get('subtopic','')} WEIGHT:{s.weight:.2f}\n{s.text}"
        )
    if extra_docs:
        for d in extra_docs:
            sid = d.get("id", f"extra:{abs(hash(d.get('text','')))}")
            blocks.append(
                f"[source_id:{sid}] PACK:external TITLE:{d.get('_]()
