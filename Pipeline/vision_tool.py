"""Box 4 - Vision tool: Grad-CAM heatmap + a textual description of WHERE the model looked
+ a faithfulness check (does hiding the highlighted pixels actually change the prediction?)."""
import os
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from . import config as C


class VisionTool:
    def __init__(self, first_pass):
        self.fp = first_pass
        self.net = first_pass.cnn

    def gradcam(self, x, cls):
        self.net.eval()
        with torch.enable_grad():
            f = self.net.features(x.unsqueeze(0))
            f.retain_grad()
            logit = self.net.loc(F.adaptive_avg_pool2d(f, 1).flatten(1))
            self.net.zero_grad()
            logit[0, cls].backward()
            w = f.grad.mean((2, 3), keepdim=True)
            cam = F.relu((w * f).sum(1, keepdim=True))
            cam = F.interpolate(cam, (C.IMG_SIZE, C.IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)

    @staticmethod
    def describe(cam):
        """Turn the heatmap into words: position on a 3x3 grid (image orientation) + spread."""
        m = cam >= 0.5
        if m.sum() == 0:
            return {"region": "diffuse / no focal area", "area_pct": 0.0}
        ys, xs = np.nonzero(m)
        cy, cx = ys.mean() / cam.shape[0], xs.mean() / cam.shape[1]
        v = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
        h = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "central")
        region = "central" if (v, h) == ("middle", "central") else f"{v} {h}".replace("middle ", "")
        area = float(m.mean() * 100)
        spread = "focal" if area < 15 else ("broad" if area > 40 else "moderate")
        return {"region": f"{spread} activation, {region} part of the image (image orientation)",
                "area_pct": round(area, 1), "centroid_xy": [round(cx, 2), round(cy, 2)]}

    @torch.no_grad()
    def deletion_test(self, x, cam, cls, frac=C.DELETION_FRACTION, seed=0):
        """Faithfulness: occlude the top-`frac` Grad-CAM region and compare with occluding a region of the SAME
        shape and size moved to a random other place (a scattered random-pixel baseline is out-of-distribution
        for the CNN and would always look more damaging). Faithful = the Grad-CAM region matters more."""
        k = int(frac * cam.size)
        mask = np.zeros(cam.size, bool); mask[np.argsort(-cam.ravel())[:k]] = True
        mask = mask.reshape(cam.shape)
        rng = np.random.default_rng(seed)
        H, W = cam.shape
        shifted = [np.roll(mask, (int(rng.integers(H // 4, 3 * H // 4)), int(rng.integers(W // 4, 3 * W // 4))), (0, 1))
                   for _ in range(3)]
        p0 = self.net(x.unsqueeze(0))[1].softmax(1)[0, cls].item()

        def drop(m):
            xm = x.clone()
            xm[0][torch.as_tensor(m, device=xm.device)] = 0.0   # 0 = mid-grey after normalisation
            return p0 - self.net(xm.unsqueeze(0))[1].softmax(1)[0, cls].item()

        d_cam = drop(mask)
        d_ctrl = float(np.mean([drop(m) for m in shifted]))
        return {"p_before": round(p0, 3), "drop_gradcam": round(d_cam, 3),
                "drop_control": round(d_ctrl, 3), "faithful": bool(d_cam > d_ctrl)}

    def run(self, case, location, out_dir=None):
        cls = C.LOCATIONS.index(location)
        x = self.fp.image_tensor(case)
        cam = self.gradcam(x, cls)
        res = {"target": location, **self.describe(cam), "deletion": self.deletion_test(x, cam, cls)}
        if out_dir:
            res["heatmap_png"] = self.save_overlay(case, cam, out_dir)
        return res

    @staticmethod
    def save_overlay(case, cam, out_dir):
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(out_dir, exist_ok=True)
        img = np.array(Image.open(case.image_path).convert("L").resize((C.IMG_SIZE, C.IMG_SIZE)))
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title("input", fontsize=8)
        ax[1].imshow(img, cmap="gray"); ax[1].imshow(cam, cmap="jet", alpha=0.45); ax[1].set_title("Grad-CAM", fontsize=8)
        for a in ax: a.axis("off")
        p = os.path.join(out_dir, f"{case.image_id}_gradcam.png")
        plt.tight_layout(); plt.savefig(p, dpi=120); plt.close(fig)
        return p
