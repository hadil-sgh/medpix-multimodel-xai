# Agentic XAI pipeline (v0.1: working skeleton)

Every box in the thesis diagram is a module you can replace on its own:

```
Case input ─► First-pass model ─► Route decision ─┬─ short ─► Vision tool ───────────────────────┐
 data.py       first_pass.py       router.py       │           vision_tool.py                     ├─► Synthesizer ─► Verifier ─► Output
                                                   └─ long ──► Vision tool + Evidence gathering ──┘   synthesizer.py   verifier.py   run_pipeline.py
                                                                          evidence.py (retrieval + KG)        ▲             │
                                                                                                              └── fail: once more via long path
```

| Box | Module | Current backend | Swap for later |
|---|---|---|---|
| Case input | `data.py` | MedPix 2.0 split files | (none) |
| First-pass model | `first_pass.py` | small CNN (image) + TF-IDF/LR (history), late fusion | DR-Minerva, if the authors share the checkpoint |
| Route decision | `router.py` | confident **and** image/text agree → short | learned router |
| Vision tool | `vision_tool.py` | Grad-CAM + region description + faithfulness test | Grad-CAM on a stronger backbone |
| Retrieval | `evidence.py` | BM25 over train_1 histories | Week-3 FAISS / Linq-Embed-Mistral (`--retrieval faiss`) |
| Knowledge graph | `evidence.py` | reads any llama-index `graph_store.json` | Phi-3 KG (`--kg hybrid-train-10tripsphi3mini`) |
| Synthesizer | `synthesizer.py` | template (every sentence traceable) | Phi-3 (`--synth phi3`) |
| Verifier | `verifier.py` | consistency, grounding, faithfulness, confidence | NLI-based claim checking |

## Run (from the repo root, the folder containing `MedPix-2-0/`)

```bash
pip install rank_bm25 scikit-learn torch torchvision matplotlib
python -m pipeline.kg_structured --split train                           # structured KG (no LLM), ~5 s
python -m pipeline.run_pipeline --split test --n 3 --kg structured-train  # demo: prints 3 explanations
python -m pipeline.run_pipeline --split test --kg structured-train       # full test split + metrics
python -m pipeline.run_pipeline --split test --kg hybrid-train-10tripsphi3mini   # Phi-3 KG, once it is built
```

Outputs are saved to `outputs/pipeline/`: `results-test.jsonl` (one record per image), `summary-test.json`, and `heatmaps/*.png`.

`python -m pipeline.calibrate_router` picks the routing threshold on **dev** and saves it to `weights/router.json`.

## Output record (per image)
`prediction` (modality, location, diagnosis, category) · `confidence` · `differential` (top-3 with KG facts) ·
`visual_explanation` (Grad-CAM region, area, faithfulness) · `text_explanation` (paragraph) ·
`supporting_evidence` (history words, similar cases, KG facts) · `status` (released / flagged for clinician review) ·
`trace` (route + verifier decisions).

## v0.1 results, test split (180 images with files available, CPU, 0.11 s/image)

| Metric | Value |
|---|---|
| Modality / location / joint accuracy | 92.8 / 75.0 / 71.7 % |
| Routing | 64 short (joint acc 89.1 %) · 116 long (62.1 %) |
| Released by verifier | 65 % (rest flagged: 60 low confidence, 4 unfaithful heatmap) |
| Grad-CAM faithful (occluding the heatmap hurts more than occluding the same shape elsewhere) | 97.8 % |
| Disease category in top-3 | 11.7 % (with structured KG) / 12.2 % (no KG) |
| Exact diagnosis in top-3 | 0 % |

The weak link is the diagnosis: retrieval uses the history only. Only 20 of the 268 train_2 diagnoses appear anywhere
in train_1, so exact diagnosis is nearly impossible from retrieval. Next steps: image + history retrieval, the Phi-3 KG,
and evaluation by disease category.

## Known limitations
* The first-pass CNN is trained from scratch (no pretrained weights), with 128 px inputs. It is a stand-in, not the final model.
* The images come from the GitHub copy of MedPix 2.0; 261 of 2,050 images are missing there (test: 180/200).
* The Grad-CAM region is described in image coordinates, not anatomical ones.
