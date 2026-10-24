"""Box 5 - Evidence gathering.
  * Retrieval:        "have we seen this before, and what was it?"  -> similar train_1 cases + their diagnoses
  * Knowledge graph:  "known links between the findings and possible diagnoses" -> triplets from the KG
Both return plain dicts so the synthesizer / verifier can cite them.
"""
import json, os, re
from collections import defaultdict
import numpy as np
from . import config as C
from .data import load_case_records, clean

STOP = set("""a an the and or of in on at to for with without from by is was were are be been has have had this that
these those patient patients year years old yo y o male female man woman history presents presented presenting
his her he she who which no not status post s p pt w""".split())


def short_dx(t):
    """'Renal Cell Carcinoma.' -> 'Renal Cell Carcinoma'; drops parenthetical remarks."""
    t = re.sub(r"\s*\(.*?\)", "", re.sub(r"\s+", " ", clean(t)))
    t = re.split(r"\.\s", t)[0]          # keep the first sentence only
    return t.strip(" .;,")


def tokens(t):
    return [w for w in re.findall(r"[a-z0-9]+", str(t).lower()) if len(w) > 1 and w not in STOP]


# --------------------------------------------------------------------------- retrieval
class Retriever:
    """BM25 over the same train_1 corpus used by RAG_costruction_history.py (history text + demographics).
    The diagnosis is kept as metadata only (not indexed), since the query never contains it."""

    def __init__(self, split=C.RETRIEVAL_CORPUS_SPLIT, backend=C.RETRIEVAL_BACKEND):
        cases, demo = load_case_records(split)
        self.docs = []
        for c in cases:
            age, sex, loc, mod = demo.get(c["U_id"], ("", "", "", ""))
            self.docs.append({
                "uid": c["U_id"], "age": age, "sex": sex, "location": loc, "modality": mod,
                "history": clean(c["Case"].get("History", "")),
                "diagnosis": short_dx(c["Case"].get("Title", "")),
                "diagnosis_full": re.sub(r"\s+", " ", clean(c["Case"].get("Title", ""))),
                "case_diagnosis": clean(c["Case"].get("Case Diagnosis", "")),
                "category": clean(c.get("Topic", {}).get("Category", "")),
            })
        self.backend = backend
        if backend == "faiss":
            self._init_faiss()
        else:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokens(f"{d['age']} {d['sex']} {d['history']}") or ["empty"] for d in self.docs])
        self.by_uid = {d["uid"]: d for d in self.docs}

    def _init_faiss(self):
        # Week-3 index built by code-DRMinerva/RAG_costruction_history.py (needs a GPU for Linq-Embed-Mistral)
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        emb = HuggingFaceEmbeddings(model_name="Linq-AI-Research/Linq-Embed-Mistral",
                                    model_kwargs={"device": "cuda", "trust_remote_code": True},
                                    encode_kwargs={"normalize_embeddings": True})
        self.faiss = FAISS.load_local(C.FAISS_DIR, emb, allow_dangerous_deserialization=True)

    def search(self, case, k=4, prefer_location=None):
        if self.backend == "faiss":
            hits = self.faiss.similarity_search_with_score(case.query_text, k=k * 3)
            ranked = [(self.by_uid[h.metadata["U_id"]], float(-s)) for h, s in hits if h.metadata["U_id"] in self.by_uid]
        else:
            s = self.bm25.get_scores(tokens(case.query_text) or ["empty"])
            order = np.argsort(-s)[: k * 3]
            ranked = [(self.docs[i], float(s[i])) for i in order]
        if prefer_location:  # long path: move neighbours from the predicted body region to the front
            ranked.sort(key=lambda t: (t[0]["location"] != prefer_location, -t[1]))
        out = []
        for d, sc in ranked[:k]:
            out.append({"uid": d["uid"], "score": round(sc, 3), "diagnosis": d["diagnosis"],
                        "category": d["category"], "location": d["location"], "modality": d["modality"],
                        "history": d["history"][:200]})
        return out


# --------------------------------------------------------------------------- knowledge graph
class KnowledgeGraph:
    """Reads a persisted llama-index SimpleGraphStore (graph_store.json = {"graph_dict": {subj: [[rel, obj], ...]}})
    and answers keyword queries WITHOUT an LLM: triplets whose subject/object share words with the query."""

    def __init__(self, graph_name=C.KG_GRAPH_NAME, graphs_dir=C.KG_GRAPHS_DIR):
        p = os.path.join(graphs_dir, graph_name, "graph_store.json")
        self.available = os.path.exists(p)
        self.path = p
        self.triplets = []
        if self.available:
            g = json.load(open(p)).get("graph_dict", {})
            for s, rels in g.items():
                for r, o in rels:
                    self.triplets.append((s, r, o, set(tokens(s)), set(tokens(o))))

    def facts_about(self, name, n=6):
        """Facts whose subject IS this entity (exact, case-insensitive), else the best token-overlap subject."""
        if not self.available or not name:
            return []
        nl = name.lower()
        hits = [t for t in self.triplets if t[0].lower() == nl]
        if not hits:
            q = set(tokens(name))
            best = max(((len(q & t[3]) / (len(q | t[3]) or 1), t[0]) for t in self.triplets), default=(0, None))
            if best[0] >= 0.6:
                hits = [t for t in self.triplets if t[0] == best[1]]
        return [{"subject": s, "relation": r, "object": o} for s, r, o, _, _ in hits[:n]]

    def query(self, terms, n=C.KG_MAX_TRIPLETS):
        if not self.available:
            return []
        q = set()
        for t in terms:
            q |= set(tokens(t))
        scored = []
        for s, r, o, ts, to in self.triplets:
            ov = len(q & ts) * 2 + len(q & to)  # subject matches weigh more
            if ov:
                scored.append((ov, s, r, o))
        scored.sort(key=lambda t: -t[0])
        seen, out = set(), []
        for ov, s, r, o in scored:
            key = (s.lower(), r.lower(), o.lower())
            if key not in seen:
                seen.add(key); out.append({"subject": s, "relation": r, "object": o, "match": ov})
            if len(out) >= n:
                break
        return out


# --------------------------------------------------------------------------- aggregation
def candidate_diagnoses(neighbours, kg=None, predicted_location=None, top=3):
    """Differential diagnosis = vote of retrieved neighbours (weighted by rank).
    If a KG is available, each candidate is looked up in it: when the KG says the disease occurs in the body
    region predicted from the image, the candidate gets a bonus (image + knowledge agree)."""
    votes = defaultdict(float); cat = {}
    for rank, n in enumerate(neighbours):
        d = n["diagnosis"]
        if not d:
            continue
        votes[d] += 1.0 / (rank + 1)
        cat.setdefault(d, n["category"])
    facts, loc_ok = {}, {}
    loc_words = set(tokens(predicted_location or ""))
    for d in votes:
        facts[d] = kg.facts_about(d) if kg is not None else []
        objs = set()
        for f in facts[d]:
            objs |= set(tokens(f["object"]))
        loc_ok[d] = bool(loc_words and loc_words <= objs)
        if loc_ok[d]:
            votes[d] += 0.5
    total = sum(votes.values()) or 1.0
    ranked = sorted(votes.items(), key=lambda t: -t[1])[:top]
    return [{"diagnosis": d, "category": cat.get(d, ""), "support": round(v / total, 3),
             "kg_facts": facts[d], "kg_location_consistent": loc_ok[d]} for d, v in ranked]
