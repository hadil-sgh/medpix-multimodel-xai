"""Deterministic 'structured' knowledge graph built directly from MedPix fields (no LLM).

Why: (1) lets the pipeline run before the Phi-3 KG finishes, (2) is a baseline to compare the
LLM-extracted KG against (does LLM extraction add knowledge beyond the structured fields?).
Saved in the same format as llama-index SimpleGraphStore, so KnowledgeGraph() reads either.

  python -m pipeline.kg_structured --split train
  -> MedPix-2-0/KG/graphs/structured-train/graph_store.json
"""
import argparse, json, os, re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from . import config as C
from .data import load_case_records, clean
from .evidence import short_dx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--symptoms_per_case", type=int, default=4)
    a = ap.parse_args()

    cases, demo = load_case_records(a.split)
    hist = [clean(c["Case"].get("History", "")) for c in cases]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(hist); names = vec.get_feature_names_out()
    skip = set("year old male female patient presents presented history yo man woman".split())

    g = defaultdict(set)
    for i, c in enumerate(cases):
        dx = short_dx(c["Case"].get("Title", ""))
        if not dx:
            continue
        age, sex, loc_cat, mod = demo.get(c["U_id"], ("", "", "", ""))
        g[dx].add(("located in", loc_cat))
        g[dx].add(("imaged with", mod))
        cat = clean(c.get("Topic", {}).get("Category", ""))
        if cat:
            g[dx].add(("is a", cat))
        acr = clean(c.get("Topic", {}).get("ACR Code", ""))
        if acr:
            g[dx].add(("ACR code", acr))
        row = X[i].toarray()[0]
        for j in row.argsort()[::-1][: a.symptoms_per_case * 2]:
            w = names[j]
            if row[j] > 0 and not (set(w.split()) & skip):
                g[dx].add(("presents with", w))
        g[dx].add(("seen in case", c["U_id"]))

    out_dir = os.path.join(C.KG_GRAPHS_DIR, f"structured-{a.split}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"graph_dict": {s: sorted([list(t) for t in rel]) for s, rel in g.items()}},
              open(os.path.join(out_dir, "graph_store.json"), "w"), indent=1)
    n = sum(len(v) for v in g.values())
    print(f"structured KG: {len(g)} diagnosis nodes, {n} triplets -> {out_dir}")
