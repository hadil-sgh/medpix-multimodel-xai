"""Agentic XAI pipeline - end-to-end orchestrator.

  Case input -> First-pass model -> Route decision -> (short) Vision tool
                                                   -> (long)  Vision tool + Evidence gathering (retrieval + KG)
             -> Synthesizer -> Verifier -(fail, once)-> long path again -> Output

Usage (from the repo root, the folder containing MedPix-2-0/):
  python -m pipeline.run_pipeline --split test --n 5                 # quick demo, prints 5 explanations
  python -m pipeline.run_pipeline --split test                       # full test split + metrics
  python -m pipeline.run_pipeline --split test --synth phi3          # LLM-written paragraphs (GPU)
  python -m pipeline.run_pipeline --split test --kg hybrid-train-10tripsphi3mini
"""
import argparse, json, os, time
from collections import Counter
import numpy as np
from . import config as C
from .data import load_split
from .first_pass import FirstPassModel
from .router import route
from .vision_tool import VisionTool
from .evidence import Retriever, KnowledgeGraph, candidate_diagnoses, short_dx
from .synthesizer import Synthesizer
from .verifier import verify


class Pipeline:
    def __init__(self, synth_backend=C.SYNTH_BACKEND, kg_name=C.KG_GRAPH_NAME, retrieval_backend=C.RETRIEVAL_BACKEND,
                 out_dir=C.OUTPUT_DIR, save_heatmaps=True):
        self.fp = FirstPassModel()
        self.vision = VisionTool(self.fp)
        self.retriever = Retriever(backend=retrieval_backend)
        self.kg = KnowledgeGraph(kg_name)
        self.synth = Synthesizer(synth_backend)
        self.out_dir = out_dir
        self.save_heatmaps = save_heatmaps
        self.known_dx = {d["diagnosis"].lower() for d in self.retriever.docs}
        print(f"[pipeline] device={self.fp.device} | retrieval={retrieval_backend} | synth={synth_backend} | "
              f"KG={'loaded ' + str(len(self.kg.triplets)) + ' triplets' if self.kg.available else 'NOT FOUND (' + self.kg.path + ')'}")

    def gather(self, case, fp, path):
        ev = {"text_words": self.fp.text_evidence(case, fp.location)}
        if path == "short":
            ev["neighbours"] = self.retriever.search(case, k=C.K_SHORT)
            ev["kg"] = []
        else:
            ev["neighbours"] = self.retriever.search(case, k=C.K_LONG, prefer_location=fp.location)
            terms = [case.history, fp.location] + [n["diagnosis"] for n in ev["neighbours"]]
            ev["kg"] = self.kg.query(terms)
        ev["candidates"] = candidate_diagnoses(ev["neighbours"], self.kg if path == "long" else None, fp.location)
        return ev

    def run(self, case):
        t0 = time.time()
        fp = self.fp.predict(case)
        path, why = route(fp)
        trace = [{"step": "route", "path": path, "why": why}]
        hm_dir = os.path.join(self.out_dir, "heatmaps") if self.save_heatmaps else None
        vision = self.vision.run(case, fp.location, hm_dir)
        for attempt in range(C.MAX_RETRIES + 1):
            evidence = self.gather(case, fp, path)
            paragraph = self.synth(case, fp, vision, evidence, path)
            ver = verify(paragraph, fp, vision, evidence, self.known_dx)
            trace.append({"step": f"verify#{attempt + 1}", "path": path, "passed": ver["passed"], "failed": ver["failed"]})
            if ver["passed"] or path == "long":
                break
            path = "long"  # verifier sends the case back through the long path once
        top = evidence["candidates"][0] if evidence["candidates"] else {"diagnosis": "undetermined", "category": ""}
        return {
            "image_id": case.image_id, "uid": case.uid,
            "prediction": {"modality": fp.modality, "location": fp.location,
                           "diagnosis": top["diagnosis"], "diagnosis_category": top["category"]},
            "confidence": {"modality": fp.p_modality, "location": fp.p_location,
                           "diagnosis_support": top.get("support", 0.0)},
            "differential": evidence["candidates"],
            "visual_explanation": vision,
            "text_explanation": paragraph,
            "supporting_evidence": {"history_words": evidence["text_words"], "similar_cases": evidence["neighbours"],
                                    "kg_facts": evidence["kg"]},
            "status": "released" if ver["passed"] else "flagged for clinician review",
            "verifier": ver, "path": path, "trace": trace, "first_pass": fp.to_dict(),
            "seconds": round(time.time() - t0, 2),
            "ground_truth": {"modality": case.gt_modality, "location": case.gt_location,
                             "diagnosis": case.gt_diagnosis, "category": case.gt_category},
        }


def summarize(results):
    n = len(results)
    g = lambda r, k: r["ground_truth"][k]
    mod = np.mean([r["prediction"]["modality"] == g(r, "modality") for r in results])
    loc = np.mean([r["prediction"]["location"] == g(r, "location") for r in results])
    joint = np.mean([r["prediction"]["modality"] == g(r, "modality") and r["prediction"]["location"] == g(r, "location") for r in results])
    cat1 = np.mean([r["prediction"]["diagnosis_category"].lower() == g(r, "category").lower() for r in results])
    cat3 = np.mean([any(c["category"].lower() == g(r, "category").lower() for c in r["differential"]) for r in results])
    dx3 = np.mean([any(c["diagnosis"].lower() == short_dx(g(r, "diagnosis")).lower() for c in r["differential"]) for r in results])
    paths = Counter(r["path"] for r in results)
    by_path = {p: round(100 * np.mean([r["prediction"]["location"] == g(r, "location") and r["prediction"]["modality"] == g(r, "modality")
                                       for r in results if r["path"] == p]), 1) for p in paths}
    fails = Counter(f for r in results for f in r["verifier"]["failed"])
    return {
        "n_images": n,
        "accuracy_modality_%": round(100 * mod, 1), "accuracy_location_%": round(100 * loc, 1), "accuracy_joint_%": round(100 * joint, 1),
        "disease_category_top1_%": round(100 * cat1, 1), "disease_category_in_top3_%": round(100 * cat3, 1),
        "exact_diagnosis_in_top3_%": round(100 * dx3, 1),
        "paths": dict(paths), "joint_accuracy_by_path_%": by_path,
        "released_%": round(100 * np.mean([r["status"] == "released" for r in results]), 1),
        "verifier_failures": dict(fails),
        "sent_back_by_verifier": sum(len(r["trace"]) > 2 for r in results),
        "grad_cam_faithful_%": round(100 * np.mean([r["visual_explanation"]["deletion"]["faithful"] for r in results]), 1),
        "mean_seconds_per_image": round(float(np.mean([r["seconds"] for r in results])), 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="only the first n images (0 = all)")
    ap.add_argument("--synth", default=C.SYNTH_BACKEND, choices=["template", "phi3"])
    ap.add_argument("--retrieval", default=C.RETRIEVAL_BACKEND, choices=["bm25", "faiss"])
    ap.add_argument("--kg", default=C.KG_GRAPH_NAME)
    ap.add_argument("--out", default=C.OUTPUT_DIR)
    ap.add_argument("--no_heatmaps", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cases = load_split(a.split)
    if a.n:
        cases = cases[: a.n]
    pipe = Pipeline(a.synth, a.kg, a.retrieval, a.out, save_heatmaps=not a.no_heatmaps)
    results = []
    with open(os.path.join(a.out, f"results-{a.split}.jsonl"), "w") as f:
        for i, case in enumerate(cases):
            r = pipe.run(case)
            results.append(r)
            f.write(json.dumps(r) + "\n")
            if a.n and a.n <= 10:
                print(f"\n=== {r['image_id']}  path={r['path']}  status={r['status']}")
                print("GT  :", r["ground_truth"])
                print("PRED:", r["prediction"], r["confidence"])
                print("TEXT:", r["text_explanation"])
            elif (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(cases)}")
    s = summarize(results)
    json.dump(s, open(os.path.join(a.out, f"summary-{a.split}.json"), "w"), indent=2)
    print("\nSUMMARY", json.dumps(s, indent=2))
