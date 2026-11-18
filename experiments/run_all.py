"""All experiments of the phishing detector. Needs data/messages.parquet (build_dataset.py).

    python experiments/run_all.py           # about 5 minutes on a laptop

Writes figures/*.png, results/*.csv, results/summary.json, models/phishguard.joblib.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from lightgbm import LGBMClassifier                                     # noqa: E402
from sklearn.metrics import (auc, average_precision_score, f1_score, precision_recall_curve,  # noqa: E402
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold                     # noqa: E402

from phishguard.model import DESCRIPTIONS, Detector, feature_names      # noqa: E402

FIG, RES, MOD = (os.path.join(ROOT, d) for d in ("figures", "results", "models"))
for d in (FIG, RES, MOD):
    os.makedirs(d, exist_ok=True)
S = {}

# ---------------------------------------------------------------- style: "security console"
BG, PANEL, INK, MUTE, GRIDC = "#0b1020", "#121a33", "#e6ecff", "#8892b0", "#1e2a4a"
RED, GREEN, CYAN, AMBER, VIOLET = "#ff3b6b", "#22d39b", "#38bdf8", "#fbbf24", "#a78bfa"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG, "axes.edgecolor": GRIDC,
    "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "text.color": INK,
    "font.family": "DejaVu Sans Mono", "font.size": 11.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 170, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)


def boot_ci(y, p, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx1, idx0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n):
        b = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
        vals.append(metric(y[b], p[b]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load():
    df = pd.read_parquet(os.path.join(ROOT, "data", "messages.parquet"))
    return df


# ------------------------------------------------------------ 1. data and the shortcut test
def exp_data_and_shortcut(df):
    eng = df[df.english]
    counts = df.groupby(["source", "english"]).size().unstack(fill_value=0)
    counts.to_csv(os.path.join(RES, "dataset_counts.csv"))
    main = eng[eng.source.isin(["phishing", "ham_easy", "ham_hard"])].reset_index(drop=True)
    y = main.label.values
    sc_cols = [c for c in df.columns if c.startswith("sc_")]
    res = {}
    for name, cols in [("date year only", ["sc_date_year"]), ("header shortcuts (no date)", [c for c in sc_cols if c != "sc_date_year"]),
                       ("content features (this project)", feature_names())]:
        oof = np.zeros(len(main))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(main, y):
            m = LGBMClassifier(n_estimators=200, verbose=-1, random_state=0).fit(main.loc[tr, cols].values, y[tr])
            oof[te] = m.predict_proba(main.loc[te, cols].values)[:, 1]
        res[name] = dict(auc=roc_auc_score(y, oof), acc=float(((oof > 0.5) == y).mean()))
    S["shortcut"] = {k: {a: round(b, 4) for a, b in v.items()} for k, v in res.items()}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.8), gridspec_kw=dict(width_ratios=[1.2, 1]))
    yrs = df[df.sc_date_year > 1990]
    bins = np.arange(2001.5, 2027.5, 1)
    for src, c, lab in [("ham_easy", GREEN, "legitimate (SpamAssassin easy ham)"), ("ham_hard", CYAN, "legitimate, hard (SpamAssassin hard ham)"),
                        ("spam", AMBER, "spam (SpamAssassin spam_2)"), ("phishing", RED, "phishing (phishing_pot)")]:
        a1.hist(yrs[yrs.source == src].sc_date_year, bins=bins, color=c, alpha=0.85, label=lab)
    a1.set_yscale("log")
    a1.set_xlabel("year in the Date header")
    a1.set_ylabel("messages")
    a1.set_title("The two classes come from different decades")
    a1.legend(loc="upper center", fontsize=10)
    names = list(res)
    a2.barh(range(3)[::-1], [res[n]["auc"] for n in names], color=[RED, AMBER, GREEN], height=0.6)
    for i, n in enumerate(names):
        a2.text(0.52, 2 - i, n, va="center", ha="left", color=BG, fontweight="bold", fontsize=11)
        a2.text(res[n]["auc"] - 0.01, 2 - i, f"AUC {res[n]['auc']:.4f}", va="center", ha="right", color=BG, fontweight="bold")
    a2.set_yticks([])
    a2.set_xlim(0.5, 1.0)
    a2.set_xlabel("ROC AUC, 5-fold cross-validation")
    a2.set_title("A model can score ~1.0 for the wrong reason")
    save(fig, "fig1_shortcut_test.png")
    return main


# ----------------------------------------------------------------------- 2. main CV
def exp_cv(main):
    y = main.label.values
    oof = {k: np.zeros(len(main)) for k in ("features", "text", "combined")}
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(main, y)):
        det = Detector(seed=k).fit(main.iloc[tr])
        part = main.iloc[te]
        oof["features"][te] = det.proba_features(part)
        oof["text"][te] = det.proba_text(part)
        oof["combined"][te] = 0.5 * oof["features"][te] + 0.5 * oof["text"][te]
    rows = []
    for k, p in oof.items():
        yhat = (p >= 0.5).astype(int)
        fpr = float(((yhat == 1) & (y == 0)).sum() / (y == 0).sum())
        rows.append(dict(model=k, auc=roc_auc_score(y, p), auc_ci=boot_ci(y, p, roc_auc_score, 500),
                         ap=average_precision_score(y, p), f1=f1_score(y, yhat), precision=precision_score(y, yhat),
                         recall=recall_score(y, yhat), fpr=fpr,
                         fpr_hard=float(yhat[(main.source == "ham_hard").values].mean()),
                         fpr_easy=float(yhat[(main.source == "ham_easy").values].mean())))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "cv_metrics.csv"), index=False)
    pd.DataFrame({**{f"p_{k}": v for k, v in oof.items()}, "label": y, "source": main.source}).to_csv(os.path.join(RES, "cv_oof.csv"), index=False)
    S["cv"] = tab.round(4).to_dict("records")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 5.8))
    cols = {"features": CYAN, "text": VIOLET, "combined": RED}
    lab = {"features": "39 features + gradient boosting", "text": "TF-IDF words + logistic regression", "combined": "combined (average)"}
    for k, p in oof.items():
        f, t, _ = roc_curve(y, p)
        a1.plot(f, t, color=cols[k], lw=2.4, label=f"{lab[k]}  AUC {auc(f, t):.4f}")
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, color=cols[k], lw=2.4, label=f"AP {average_precision_score(y, p):.4f}")
    a1.set_xscale("log")
    a1.set_xlim(1e-3, 1)
    a1.set_xlabel("false-positive rate (log)")
    a1.set_ylabel("true-positive rate")
    a1.set_title("ROC, 5-fold cross-validation")
    a1.legend(loc="lower right", fontsize=9.5)
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a2.set_title("precision–recall")
    a2.set_ylim(0.8, 1.005)
    a2.legend(loc="lower left")
    p = oof["combined"]
    bins = np.linspace(0, 1, 41)
    for src, c, l in [("ham_easy", GREEN, "easy ham"), ("ham_hard", CYAN, "hard ham"), ("phishing", RED, "phishing")]:
        a3.hist(p[(main.source == src).values], bins=bins, color=c, alpha=0.8, label=l)
    a3.set_yscale("log")
    a3.axvline(0.5, color=INK, ls="--", lw=1)
    a3.set_xlabel("combined score")
    a3.set_ylabel("messages")
    a3.set_title("score distribution by source")
    a3.legend()
    save(fig, "fig2_cv_performance.png")
    return oof


# ------------------------------------------------------------------ 3. explanations
def exp_shap(main):
    det = Detector(seed=0).fit(main)
    det.save(os.path.join(MOD, "phishguard.joblib"))
    names = feature_names()
    X = main[names]
    sv = det.shap_values(main)
    imp = np.abs(sv).mean(0)
    order = np.argsort(-imp)[:16]
    pd.DataFrame(dict(feature=[names[i] for i in np.argsort(-imp)], mean_abs_shap=np.sort(imp)[::-1])).to_csv(os.path.join(RES, "shap_importance.csv"), index=False)
    fig, ax = plt.subplots(figsize=(13, 8.2))
    rng = np.random.default_rng(0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("v", [CYAN, "#5b6ea8", RED])
    for r, i in enumerate(order[::-1]):
        v = X.iloc[:, i].values.astype(float)
        q = (v - np.percentile(v, 2)) / (np.percentile(v, 98) - np.percentile(v, 2) + 1e-9)
        sel = rng.choice(len(v), min(2500, len(v)), replace=False)
        ax.scatter(sv[sel, i], r + rng.normal(0, 0.12, len(sel)), c=np.clip(q[sel], 0, 1), cmap=cmap, s=6, alpha=0.7, lw=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DESCRIPTIONS.get(names[i], names[i]) for i in order[::-1]], fontsize=10.5)
    ax.axvline(0, color=MUTE, lw=1)
    ax.set_xlabel("SHAP value: push towards phishing (right) or legitimate (left), log-odds")
    ax.set_title("What the feature model looks at (each dot is one message; colour = feature value, cyan low → red high)")
    save(fig, "fig3_shap_summary.png")
    S["top_features"] = [names[i] for i in order[:10]]
    # ablation by feature group
    groups = {"links": [n for n in names if n.startswith(("n_urls", "n_link", "url_", "links_", "anchor", "n_anchor"))],
              "HTML": ["html_only", "has_form", "has_script", "has_iframe", "n_images", "hidden_text", "html_text_ratio"],
              "wording": ["n_words", "urgent_words", "credential_words", "money_words", "click_here", "upper_ratio", "exclamations", "subject_len", "subject_has_re_fw"],
              "sender": ["brand_mentioned", "brand_not_in_sender_domain", "reply_to_differs", "return_path_differs", "display_name_has_address", "n_recipients", "from_freemail", "auth_fail"],
              "attachments": ["n_attachments", "risky_attachment", "attachment_pdf"]}
    y = main.label.values
    rows = []
    for g, cols in [("all", [])] + list(groups.items()):
        keep = [n for n in names if n not in cols]
        oof = np.zeros(len(main))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=2).split(main, y):
            m = LGBMClassifier(n_estimators=300, verbose=-1, random_state=0).fit(main.loc[tr, keep].values, y[tr])
            oof[te] = m.predict_proba(main.loc[te, keep].values)[:, 1]
        rows.append(dict(removed=g, auc=roc_auc_score(y, oof), recall_at_1pct_fpr=_tpr_at(y, oof, 0.01)))
    ab = pd.DataFrame(rows)
    ab.to_csv(os.path.join(RES, "ablation.csv"), index=False)
    S["ablation"] = ab.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    base = ab.iloc[0].recall_at_1pct_fpr
    vals = ab.iloc[1:]
    ax.barh(vals.removed, (base - vals.recall_at_1pct_fpr) * 100, color=[CYAN, VIOLET, AMBER, RED, GREEN])
    for i, r in enumerate(vals.itertuples()):
        ax.text((base - r.recall_at_1pct_fpr) * 100 + 0.1, i, f"recall {r.recall_at_1pct_fpr * 100:.1f} %", va="center", fontsize=10.5)
    ax.set_xlabel(f"recall lost at 1 % false positives when the group is removed (all groups: {base * 100:.1f} %)")
    ax.set_title("Which evidence matters: feature-group ablation")
    save(fig, "fig4_ablation.png")
    return det


def _tpr_at(y, p, fpr_target):
    f, t, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, f, t))


# --------------------------------------------------------- 4. drift and stress tests
def exp_drift_and_spam(df, main):
    """Train on the older half of the phishing samples, test on newer quarters; score spam."""
    ph = main[main.source == "phishing"].copy()
    ph["num"] = ph.file.str.extract(r"sample-(\d+)").astype(int)
    ph = ph.sort_values("num")
    ham = main[main.source != "phishing"].sample(frac=1, random_state=3)
    half = len(ph) // 2
    tr = pd.concat([ph.iloc[:half], ham.iloc[: len(ham) // 2]])
    det = Detector(seed=0).fit(tr)
    ham_te = ham.iloc[len(ham) // 2:]
    p_ham = det.predict_proba(ham_te)
    thr = float(np.quantile(p_ham, 0.99))          # threshold for 1 % false positives on held-out ham
    newer = ph.iloc[half:]
    q = np.array_split(np.arange(len(newer)), 4)
    rows = []
    for k, idx in enumerate(q):
        part = newer.iloc[idx]
        p = det.predict_proba(part)
        yrs = part.sc_date_year
        rows.append(dict(bucket=k + 1, n=len(part), year_median=float(np.median(yrs[yrs > 1990])), sample_from=int(part.num.min()),
                         sample_to=int(part.num.max()), recall_at_05=float((p >= 0.5).mean()), recall_at_1pct_fpr=float((p >= thr).mean())))
    dr = pd.DataFrame(rows)
    dr.to_csv(os.path.join(RES, "drift.csv"), index=False)
    S["drift"] = dr.round(4).to_dict("records")
    old = ph.iloc[:half]
    S["drift_train_years"] = [float(np.percentile(old.sc_date_year[old.sc_date_year > 1990], q)) for q in (5, 50, 95)]
    # spam stress test with the model trained on everything
    full = Detector(seed=0).fit(main)
    spam = df[(df.source == "spam") & df.english]
    ps = full.predict_proba(spam)
    S["spam_flagged_at_05"] = round(float((ps >= 0.5).mean()), 4)
    S["n_spam"] = int(len(spam))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    a1.plot(dr.bucket, dr.recall_at_05 * 100, "-o", color=RED, lw=2.6, ms=9, label="threshold 0.5")
    a1.plot(dr.bucket, dr.recall_at_1pct_fpr * 100, "-s", color=AMBER, lw=2.6, ms=9, label="threshold for 1 % false positives")
    a1.set_xticks(dr.bucket)
    a1.set_xticklabels([f"Q{r.bucket}\nsamples {r.sample_from}–{r.sample_to}\nmedian {int(r.year_median)}" for r in dr.itertuples()], fontsize=9.5)
    a1.set_ylabel("phishing detected (%)")
    a1.set_ylim(50, 101)
    a1.set_title("Trained on the older half of the phishing, tested on newer quarters")
    a1.legend(loc="lower left")
    a2.hist(ps, bins=np.linspace(0, 1, 31), color=AMBER, alpha=0.9)
    a2.axvline(0.5, color=INK, ls="--")
    a2.set_xlabel("score of 2002 spam (never seen in training)")
    a2.set_ylabel("messages")
    a2.set_title(f"Spam is not phishing: {S['spam_flagged_at_05'] * 100:.0f} % flagged")
    save(fig, "fig5_drift_spam.png")


def exp_example(main, det):
    """One phishing message explained (no addresses or links are shown)."""
    ph = main[(main.source == "phishing")]
    p = det.predict_proba(ph)
    pick = ph.iloc[[int(np.argsort(-np.abs(p - 0.97))[-1])]]
    ex = det.explain(pick, k=8)
    S["example"] = dict(score=round(ex["score"], 4), reasons=[(r["description"], round(r["value"], 3), round(r["contribution"], 3)) for r in ex["reasons"]],
                        words=[w["term"] for w in ex["words"]])
    fig, ax = plt.subplots(figsize=(12, 5))
    r = ex["reasons"][::-1]
    ax.barh(range(len(r)), [x["contribution"] for x in r], color=[RED if x["contribution"] > 0 else CYAN for x in r])
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels([f"{x['description']}  = {x['value']:g}" for x in r], fontsize=10.5)
    ax.axvline(0, color=MUTE)
    ax.set_xlabel("contribution to the feature model (log-odds)")
    ax.set_title(f"Why one message was flagged: score {ex['score']:.2f}; strongest words: {', '.join(S['example']['words'][:4])}")
    save(fig, "fig6_example_explanation.png")


def main_run():
    t0 = time.time()
    df = load()
    main = exp_data_and_shortcut(df)
    print("1 shortcut", S["shortcut"], flush=True)
    exp_cv(main)
    print("2 cv", flush=True)
    det = exp_shap(main)
    print("3 shap/ablation", flush=True)
    exp_drift_and_spam(df, main)
    print("4 drift/spam", flush=True)
    exp_example(main, det)
    S["n"] = {k: int(v) for k, v in main.source.value_counts().items()}
    S["run_time_s"] = round(time.time() - t0)
    json.dump(S, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main_run()
