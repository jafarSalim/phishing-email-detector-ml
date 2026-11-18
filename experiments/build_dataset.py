"""Parse the raw messages once and store features, text and metadata (not the raw mail).

Sources (downloaded by scripts/fetch_data.sh, not redistributed here):
  phishing  rf-peixoto/phishing_pot (CC BY-NC 4.0): real phishing and scam e-mails, 2022-2026
  ham       Apache SpamAssassin public corpus (via stdlib-js/datasets-spam-assassin, CC0):
            easy_ham, hard_ham (legitimate), and spam_2 (unsolicited, used only as a stress test)

Output: data/messages.parquet with one row per message.
"""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from phishguard import features, lang, parse  # noqa: E402

RAW = os.environ.get("PHISHGUARD_RAW", os.path.join(ROOT, "data", "raw"))
SOURCES = [("phishing", "phishing_pot/email/*.eml"), ("ham_easy", "spamassassin/easy-ham-*/*.txt"),
           ("ham_hard", "spamassassin/hard-ham-1/*.txt"), ("spam", "spamassassin/spam-2/*.txt")]


def main():
    rows = []
    for src, pat in SOURCES:
        files = sorted(glob.glob(os.path.join(RAW, pat)))
        for f in files:
            raw = open(f, "rb").read()
            try:
                m = parse.parse(raw)
            except Exception:
                continue
            x = features.extract(m)
            s = features.shortcut_features(raw)
            body = (m.subject or "") + "\n" + (m.text or "")
            rows.append(dict(source=src, file=os.path.basename(f), english=lang.is_english(body),
                             subject=m.subject or "", text=(m.text or "")[:20000], **x,
                             **{"sc_" + k: v for k, v in s.items()}))
        print(f"{src}: {len(files)} files", flush=True)
    df = pd.DataFrame(rows)
    df["label"] = (df.source == "phishing").astype(int)
    df.to_parquet(os.path.join(ROOT, "data", "messages.parquet"))
    print(df.groupby(["source", "english"]).size())


if __name__ == "__main__":
    main()
