"""Pick the route-decision threshold on the DEV split (never on test).
Rule: lowest threshold whose short-path cases reach the target joint accuracy (default 90%)
while still sending at least `min_coverage` of cases down the short path.

  python -m pipeline.calibrate_router --target 0.90
"""
import argparse, json, os
import numpy as np
from . import config as C
from .data import load_split
from .first_pass import FirstPassModel

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--min_coverage", type=float, default=0.10)
    a = ap.parse_args()
    fp = FirstPassModel()
    rows = []
    for c in load_split("dev"):
        r = fp.predict(c)
        rows.append((r.p_location, r.agree, r.location == c.gt_location and r.modality == c.gt_modality))
    p = np.array([x[0] for x in rows]); ag = np.array([x[1] for x in rows]); ok = np.array([x[2] for x in rows])
    print(f"dev images: {len(rows)} | overall joint acc {ok.mean():.3f} | agree rate {ag.mean():.3f} | "
          f"acc when agree {ok[ag].mean():.3f} / disagree {ok[~ag].mean():.3f}")
    table, chosen = [], None
    for t in np.round(np.arange(0.20, 0.96, 0.02), 2):
        m = (p >= t) & ag
        cov = m.mean(); acc = ok[m].mean() if m.any() else float("nan")
        table.append({"threshold": float(t), "short_coverage": round(float(cov), 3), "short_joint_acc": round(float(acc), 3),
                      "long_joint_acc": round(float(ok[~m].mean()), 3) if (~m).any() else None})
        if chosen is None and m.any() and acc >= a.target and cov >= a.min_coverage:
            chosen = float(t)
    for r in table[::3]:
        print(r)
    if chosen is None:
        chosen = max((r for r in table if r["short_coverage"] >= a.min_coverage), key=lambda r: r["short_joint_acc"])["threshold"]
        print("target not reachable - using the most accurate threshold with enough coverage")
    os.makedirs(os.path.dirname(C.CNN_WEIGHTS), exist_ok=True)
    json.dump({"threshold": chosen, "target": a.target, "picked_on": "dev", "table": table},
              open(os.path.join(os.path.dirname(C.CNN_WEIGHTS), "router.json"), "w"), indent=1)
    print("chosen threshold:", chosen)
