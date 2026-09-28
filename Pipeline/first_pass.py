"""Box 2 - First-pass model: modality + location with confidence and an agreement check.

Stand-in for DR-Minerva (checkpoint not public): an image CNN + a text classifier on the history,
combined by late fusion. Implements the same interface a DR-Minerva backend would:
    predict(case) -> FirstPassResult
Week-4 test results (180 test images): image-only joint 63.3%, text-only 31.7%, fusion 72.2%.
"""
from dataclasses import dataclass, asdict
import re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from . import config as C
from .data import load_split

# late-fusion weight of the IMAGE model (chosen on dev in Week 4); text gets 1 - w
FUSION_W_IMAGE = {"modality": 0.9, "location": 0.25}

EVAL_TF = T.Compose([T.Resize((C.IMG_SIZE, C.IMG_SIZE)), T.ToTensor(), T.Normalize([0.5], [0.25])])


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class SmallCNN(nn.Module):
    """4-block CNN, two heads. `features()` returns the 16x16 map used by Grad-CAM."""
    def __init__(self):
        super().__init__()
        self.b1, self.b2, self.b3, self.b4 = _block(1, 24), _block(24, 48), _block(48, 96), _block(96, 160)
        self.drop = nn.Dropout(0.3)
        self.mod, self.loc = nn.Linear(160, 2), nn.Linear(160, 5)

    def features(self, x):
        x = F.max_pool2d(self.b1(x), 2); x = F.max_pool2d(self.b2(x), 2); x = F.max_pool2d(self.b3(x), 2)
        return self.b4(x)

    def forward(self, x):
        f = self.drop(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))
        return self.mod(f), self.loc(f)


# demographic / filler tokens that the text model uses but that are not clinical evidence
UNINFORMATIVE = set("""yo year years old male female patient man woman boy girl seen health prior history
presents presented developed day days week weeks month months ago""".split())


def text_for_classifier(case):
    # add spaces so "72" "male" "patient" are separate tokens (the raw RAG query glues them)
    return f"{case.age} {case.sex} patient. {case.history}"


@dataclass
class FirstPassResult:
    modality: str
    location: str
    p_modality: float
    p_location: float
    image_location: str       # what the image model alone says
    text_location: str        # what the text model alone says
    agree: bool               # image and text agree on location
    probs_location: dict

    def to_dict(self):
        return asdict(self)


class FirstPassModel:
    def __init__(self, device=None, weights=C.CNN_WEIGHTS, text_train_split="train"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn = SmallCNN().to(self.device).eval()
        self.cnn.load_state_dict(torch.load(weights, map_location=self.device))
        # text model is tiny: re-fit at start-up (avoids sklearn pickle version problems on Colab)
        tr = load_split(text_train_split, only_available_images=False)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2)
        X = self.vec.fit_transform([text_for_classifier(c) for c in tr])
        self.txt_mod = LogisticRegression(C=30, max_iter=3000, class_weight="balanced").fit(
            X, [C.MODALITIES.index(c.gt_modality) for c in tr])
        self.txt_loc = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit(
            X, [C.LOCATIONS.index(c.gt_location) for c in tr])

    def image_tensor(self, case):
        return EVAL_TF(Image.open(case.image_path).convert("L")).to(self.device)

    @torch.no_grad()
    def predict(self, case) -> FirstPassResult:
        x = self.image_tensor(case).unsqueeze(0)
        lm, ll = self.cnn(x)
        pim, pil = lm.softmax(1)[0].cpu().numpy(), ll.softmax(1)[0].cpu().numpy()
        xt = self.vec.transform([text_for_classifier(case)])
        ptm, ptl = self.txt_mod.predict_proba(xt)[0], self.txt_loc.predict_proba(xt)[0]
        wm, wl = FUSION_W_IMAGE["modality"], FUSION_W_IMAGE["location"]
        pm, pl = wm * pim + (1 - wm) * ptm, wl * pil + (1 - wl) * ptl
        return FirstPassResult(
            modality=C.MODALITIES[int(pm.argmax())], location=C.LOCATIONS[int(pl.argmax())],
            p_modality=round(float(pm.max()), 3), p_location=round(float(pl.max()), 3),
            image_location=C.LOCATIONS[int(pil.argmax())], text_location=C.LOCATIONS[int(ptl.argmax())],
            agree=bool(pil.argmax() == ptl.argmax()),
            probs_location={C.LOCATIONS[i]: round(float(p), 3) for i, p in enumerate(pl)},
        )

    def text_evidence(self, case, location, top=5):
        """Words in the history that pushed the text model towards `location` (tf-idf x coefficient)."""
        k = C.LOCATIONS.index(location)
        v = self.vec.transform([text_for_classifier(case)])
        names = self.vec.get_feature_names_out()
        contrib = [(names[j], float(v[0, j] * self.txt_loc.coef_[k, j])) for j in v.nonzero()[1]]
        contrib = [(w, round(s, 3)) for w, s in sorted(contrib, key=lambda t: -t[1])
                   if s > 0 and not (set(w.split()) & UNINFORMATIVE) and not re.search(r"\d", w)]
        return contrib[:top]
