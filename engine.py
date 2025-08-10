import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import tiktoken
from openai import OpenAI

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
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.snippets: List[Snippet] = self._load_cards(canon_path)
        self.embeddings = None  # lazy

    def _load_cards(self, path: str) -> List[Snippet]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                card = json.loads(line)
                body = []
                def add(label, items, field=None):
                    if items:
                        if isinstance(items, list) and items and isinstance(items[0], dict):
                            vals = [i.get(field, "") for i in items if field is not None and i.get(field)]
                            body.append(f"{label}: " + " | ".join(vals))
                        elif isinstance(items, list):
                            body.append(f"{label}: " + " | ".join(items))
                add("THESES", card.get("theses"), "text")
                add("QUOTES", card.get("quotes"), "text")
                add("COUNTERS", card.get("counters"), "text")
                add("IMPLICATIONS", card.get("implications"))
                add("FALSIFIERS", card.get("falsifiers"))
                text = f"{card.get('title')} — {card.get('author','')} | {card.get('pack')} | {card.get('subtopic')}
" + "\n".join([b for b in body if b])
                out.append(Snippet(
                    id=card["id"],
                    pack=card.get("pack",""),
                    weight=float(card.get("weight",1.2)),
                    text=text,
                    meta={"title": card.get("title"), "parent": card.get("parent"), "subtopic": card.get("subtopic"), "tags": card.get("tags",[])}
                ))
        return out

    def _embed(self, texts: List[str]) -> np.ndarray:
        res = self.client.embeddings.create(model=ENC_MODEL, input=texts)
        arr = np.array([d.embedding for d in res.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-10
        return arr / norms

    def ensure_index(self):
        if self.embeddings is None:
            self.embeddings = self._embed([s.text for s in self.snippets])

    def retrieve(self, query: str, top_k: int = 12) -> List[Tuple[Snippet, float]]:
        self.ensure_index()
        q = self._embed([query])[0]
        sims = (self.embeddings @ q).flatten()
        scores = sims * np.array([s.weight for s in self.snippets], dtype=np.float32)
        idx = np.argsort(-scores)[:top_k]
        return [(self.snippets[i], float(scores[i])) for i in idx]

def build_context(snippets: List[Tuple[Snippet,float]], extra_docs: List[Dict[str,str]]|None=None) -> str:
    blocks = []
    for s,score in snippets:
        blocks.append(f"[source_id:{s.id}] PACK:{s.pack} TITLE:{s.meta.get('title')} SUB:{s.meta.get('subtopic')} WEIGHT:{s.weight:.2f}\n{s.text}")
    if extra_docs:
        for d in extra_docs:
            sid = d.get("id","extra:"+str(abs(hash(d.get('text','')))))
            blocks.append(f"[source_id:{sid}] PACK:external TITLE:{d.get('title','External')} SUB:None WEIGHT:1.00\n{d.get('text','')}")
    return "\n\n----\n\n".join(blocks)

def call_llm(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.3,
    )
    return resp.choices[0].message.content

def analyze(query: str, retriever: Retriever, system_prompt: str, opinion_prompt: str, critique_prompt: str, pasted_text: str|None=None):
    client = retriever.client
    hits = retriever.retrieve(query, top_k=12)
    extra = [{"id":"user:pasted","title":"User Pasted","text":pasted_text}] if pasted_text else None
    context = build_context(hits, extra_docs=extra)
    opinion = call_llm(client, system_prompt, f"{opinion_prompt}\n\nQUERY:\n{query}\n\nCONTEXT:\n{context}")
    critique = call_llm(client, system_prompt, f"{critique_prompt}\n\nQUERY:\n{query}\n\nCONTEXT:\n{context}\n\nPRIOR_OPINION:\n{opinion}")
    return opinion, critique, context
