"""The detector: gradient boosting on interpretable features + a TF-IDF text model.

    det = Detector().fit(df)            # df: rows from experiments/build_dataset.py
    det.predict_proba(df)               # combined probability of phishing
    det.explain(row)                    # top reasons (feature contributions from SHAP)
    det.save("models/phishguard.joblib"); Detector.load(...)
"""
from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import features as F

FEATURES = None


def feature_names() -> List[str]:
    global FEATURES
    if FEATURES is None:
        from .parse import Message
        FEATURES = list(F.extract(Message()).keys())
    return FEATURES


DESCRIPTIONS = {
    "n_urls": "number of links", "n_link_domains": "different link domains", "url_ip_host": "link to a bare IP address",
    "url_shortener": "URL shortener", "url_at_sign": "'@' inside a link host", "url_punycode": "punycode (look-alike) domain",
    "url_risky_tld": "link on a high-abuse top-level domain", "url_max_len": "longest link length",
    "url_max_dots": "most subdomain levels in a link", "links_off_sender_domain": "links that leave the sender's domain",
    "anchor_href_mismatch": "link text shows one domain, link goes to another", "n_anchor_links": "number of HTML links",
    "html_only": "HTML with no plain-text part", "has_form": "HTML form in the mail", "has_script": "script in the mail",
    "has_iframe": "iframe in the mail", "n_images": "number of images", "hidden_text": "hidden text (display:none / size 0)",
    "html_text_ratio": "HTML size relative to visible text", "n_words": "length in words",
    "urgent_words": "urgency wording", "credential_words": "asks about passwords / accounts", "money_words": "money and payment wording",
    "click_here": "'click here' style calls to action", "upper_ratio": "share of words in CAPITALS", "exclamations": "exclamation marks",
    "subject_len": "subject length", "subject_has_re_fw": "subject starts with Re:/Fwd:", "brand_mentioned": "a well-known brand in name or subject",
    "brand_not_in_sender_domain": "brand named, but not the sender's domain", "reply_to_differs": "Reply-To on another domain",
    "return_path_differs": "Return-Path on another domain", "display_name_has_address": "an e-mail address inside the display name",
    "n_recipients": "number of recipients", "from_freemail": "sent from a free-mail domain", "n_attachments": "number of attachments",
    "risky_attachment": "risky attachment type (html, zip, iso, exe, ...)", "attachment_pdf": "PDF attachment",
    "auth_fail": "SPF / DKIM / DMARC failure recorded",
}


class Detector:
    def __init__(self, w_text: float = 0.5, seed: int = 0):
        self.w_text = w_text
        self.gbm = LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=10,
                                  subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=60000, sublinear_tf=True,
                                   strip_accents="unicode", lowercase=True)
        self.lr = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
        self._explainer = None

    @staticmethod
    def _text(df: pd.DataFrame) -> pd.Series:
        return (df["subject"].fillna("") + " \n " + df["text"].fillna("")).str.slice(0, 20000)

    def fit(self, df: pd.DataFrame, y=None) -> "Detector":
        y = df["label"].values if y is None else y
        self.gbm.fit(df[feature_names()].values.astype(float), y)
        self.lr.fit(self.vec.fit_transform(self._text(df)), y)
        return self

    def proba_features(self, df):
        return self.gbm.predict_proba(df[feature_names()].values.astype(float))[:, 1]

    def proba_text(self, df):
        return self.lr.predict_proba(self.vec.transform(self._text(df)))[:, 1]

    def predict_proba(self, df):
        return (1 - self.w_text) * self.proba_features(df) + self.w_text * self.proba_text(df)

    def shap_values(self, df) -> np.ndarray:
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.gbm)
        sv = self._explainer.shap_values(df[feature_names()].values.astype(float))
        return sv[1] if isinstance(sv, list) else sv

    def explain(self, row: pd.DataFrame, k: int = 6) -> Dict:
        sv = self.shap_values(row)[0]
        names = feature_names()
        order = np.argsort(-np.abs(sv))[:k]
        reasons = [dict(feature=names[i], description=DESCRIPTIONS.get(names[i], names[i]),
                        value=float(row[names[i]].iloc[0]), contribution=float(sv[i])) for i in order]
        coef = self.lr.coef_[0]
        x = self.vec.transform(self._text(row))
        contrib = x.multiply(coef).toarray()[0]
        voc = np.array(self.vec.get_feature_names_out())
        top = np.argsort(-contrib)[:5]
        words = [dict(term=voc[i], contribution=float(contrib[i])) for i in top if contrib[i] > 0]
        return dict(score=float(self.predict_proba(row)[0]), score_features=float(self.proba_features(row)[0]),
                    score_text=float(self.proba_text(row)[0]), reasons=reasons, words=words)

    def save(self, path: str):
        e, self._explainer = self._explainer, None
        joblib.dump(self, path, compress=3)
        self._explainer = e

    @staticmethod
    def load(path: str) -> "Detector":
        return joblib.load(path)
