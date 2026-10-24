"""Box 1 - Case input: image + history + demographics for one MedPix case/image."""
import json, os, re
from dataclasses import dataclass, field
from typing import List, Optional
from . import config as C


def _jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def clean(t):
    t = "" if t is None else str(t)
    return "" if t.strip().upper() in {"N/A", "NA", "NONE"} else t.strip()


@dataclass
class CaseInput:
    image_id: str
    uid: str
    image_path: str
    age: str
    sex: str
    history: str
    # ground truth - used ONLY for evaluation, never by the pipeline components
    gt_modality: Optional[str] = None
    gt_location: Optional[str] = None
    gt_diagnosis: Optional[str] = None
    gt_category: Optional[str] = None

    @property
    def query_text(self):
        """Same query format as code-DRMinerva/inference_rag_gen.py."""
        return f"{self.age}{self.sex}patient.\n{self.history}"


def load_split(split: str, only_available_images: bool = True) -> List[CaseInput]:
    cases = {c["U_id"]: c for c in _jsonl(os.path.join(C.SPLITS_DIR, f"data_{split}.jsonl"))}
    out = []
    for d in _jsonl(os.path.join(C.SPLITS_DIR, f"descriptions_{split}.jsonl")):
        p = os.path.join(C.IMAGES_DIR, d["image"] + ".png")
        if only_available_images and not os.path.exists(p):
            continue
        c = cases[d["U_id"]]
        desc = d.get("Description", {})
        out.append(CaseInput(
            image_id=d["image"], uid=d["U_id"], image_path=p,
            age=clean(desc.get("Age", "")), sex=clean(desc.get("Sex", "")),
            history=clean(c["Case"].get("History", "")),
            gt_modality=d["Type"], gt_location=d["Location Category"],
            gt_diagnosis=re.sub(r"\s+", " ", clean(c["Case"].get("Title", ""))),
            gt_category=clean(c.get("Topic", {}).get("Category", "")),
        ))
    return out


def load_case_records(split: str):
    """Raw case records + per-case demographics, used to build the retrieval corpus."""
    cases = _jsonl(os.path.join(C.SPLITS_DIR, f"data_{split}.jsonl"))
    demo = {}
    for d in _jsonl(os.path.join(C.SPLITS_DIR, f"descriptions_{split}.jsonl")):
        demo.setdefault(d["U_id"], (clean(d["Description"].get("Age", "")), clean(d["Description"].get("Sex", "")),
                                    d["Location Category"], d["Type"]))
    return cases, demo
