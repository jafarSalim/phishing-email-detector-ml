"""assets/scan.gif: the detector reading the synthetic phishing example, evidence by evidence."""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from phishguard import features, parse  # noqa: E402
from phishguard.model import Detector  # noqa: E402

BG, PANEL, INK, MUTE, RED, CYAN, GREEN = "#0b1020", "#121a33", "#e6ecff", "#8892b0", "#ff3b6b", "#38bdf8", "#22d39b"
raw = open(os.path.join(ROOT, "examples", "synthetic_phish.eml"), "rb").read()
m = parse.parse(raw)
row = pd.DataFrame([dict(subject=m.subject, text=m.text, **features.extract(m))])
det = Detector.load(os.path.join(ROOT, "models", "phishguard.joblib"))
ex = det.explain(row, k=6)
reasons = ex["reasons"]
base = 0.0
lines = ["From: \"PayPal Security\" <service@paypa1-secure.xyz>", "Reply-To: help@other-domain.ru",
         "Subject: URGENT: your account is suspended", "", "Dear customer, we detected unusual activity.",
         "Verify your account within 24 hours or it", "will be locked.", "", "[ https://www.paypal.com/signin ]  -> http://198.51.100.7/login",
         "<form> password: ____ </form>", "<div style=display:none> ... </div>"]
tmp = os.path.join(ROOT, "figures", "_frames")
os.makedirs(tmp, exist_ok=True)
frames = []
for k in range(len(reasons) + 1):
    fig = plt.figure(figsize=(11, 4.6), facecolor=BG)
    a1 = fig.add_axes([0.02, 0.06, 0.52, 0.86]); a1.set_facecolor(PANEL); a1.axis("off")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.text(0.03, 0.95, "synthetic test message", color=MUTE, fontsize=9, family="monospace", va="top")
    for i, l in enumerate(lines):
        a1.text(0.03, 0.86 - i * 0.075, l, color=INK, fontsize=9.2, family="monospace", va="top")
    a2 = fig.add_axes([0.62, 0.16, 0.35, 0.72]); a2.set_facecolor(PANEL)
    shown = reasons[:k]
    vals = [r["contribution"] for r in shown][::-1]
    labs = [r["description"][:30] for r in shown][::-1]
    a2.barh(range(len(vals)), vals, color=[RED if v > 0 else CYAN for v in vals])
    a2.set_yticks(range(len(vals))); a2.set_yticklabels(labs, color=INK, fontsize=8.5)
    a2.set_xlim(min(0, min([r["contribution"] for r in reasons])) - 0.5, max(r["contribution"] for r in reasons) + 1)
    a2.set_ylim(-0.6, len(reasons) - 0.4)
    a2.tick_params(colors=MUTE); [s.set_color("#1e2a4a") for s in a2.spines.values()]
    logit = sum(vals) + (np.log(ex["score_features"] / (1 - ex["score_features"])) - sum(r["contribution"] for r in reasons))
    p = 1 / (1 + np.exp(-logit)) if k else 0.5
    verdict = ("PHISHING" if p >= 0.5 else "clean") if k else "scanning…"
    a2.set_title(f"evidence {k}/{len(reasons)} · score {p:.2f} · {verdict}", color=RED if (k and p >= 0.5) else INK, fontsize=12, family="monospace", fontweight="bold")
    pth = os.path.join(tmp, f"f{k:02d}.png")
    fig.savefig(pth, dpi=85, facecolor=BG)
    plt.close(fig)
    im = Image.open(pth).convert("RGB")
    frames += [im] * (2 if k < len(reasons) else 6)
pal = frames[-1].quantize(colors=64)
q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
q[0].save(os.path.join(ROOT, "assets", "scan.gif"), save_all=True, append_images=q[1:], duration=450, loop=0, optimize=True)
for f in os.listdir(tmp):
    os.remove(os.path.join(tmp, f))
os.rmdir(tmp)
print("ok", ex["score"])
