"""Interpretable features of a parsed message.

Every feature describes *what the message asks the reader to do and how it is built*, not
when it was sent or which mail servers carried it. That choice matters for the benchmark in
this repository, whose legitimate mail is older than its phishing (see README, "the shortcut
test"); it also makes the model harder to fool with header tricks.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Dict, List
from urllib.parse import urlparse

from .parse import Message

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
              "rb.gy", "shorturl.at", "t.ly", "tiny.cc", "s.id", "lnkd.in"}
RISKY_TLDS = {"xyz", "top", "click", "online", "site", "shop", "live", "icu", "buzz", "monster", "rest", "cfd",
              "sbs", "quest", "cyou", "lol", "beauty", "hair", "skin", "bond", "ru", "cn", "tk", "ml", "ga", "cf", "gq"}
BRANDS = ["paypal", "microsoft", "office365", "outlook", "apple", "icloud", "amazon", "netflix", "dhl", "fedex",
          "ups", "usps", "bank", "wells fargo", "chase", "coinbase", "binance", "metamask", "docusign", "adobe",
          "dropbox", "wetransfer", "facebook", "instagram", "whatsapp", "linkedin", "google", "gmail", "yahoo",
          "mcafee", "norton", "geek squad", "irs", "correos", "santander", "itau", "bradesco", "caixa", "nubank"]
URGENT = ["urgent", "immediately", "verify", "verification", "suspend", "suspended", "locked", "unusual", "confirm",
          "expire", "expired", "expiration", "within 24", "within 48", "final notice", "last warning", "action required",
          "restricted", "deactivat", "security alert", "update your", "validate", "unauthorized"]
CREDENTIAL = ["password", "login", "log in", "sign in", "username", "account", "credentials", "ssn", "social security",
              "card number", "cvv", "pin", "bank details"]
MONEY = ["invoice", "payment", "refund", "transfer", "wire", "bitcoin", "btc", "crypto", "wallet", "prize", "winner",
         "lottery", "inheritance", "$", "€", "£", "usd"]
RISKY_EXT = (".html", ".htm", ".shtml", ".zip", ".rar", ".7z", ".iso", ".img", ".exe", ".scr", ".js", ".vbs",
             ".docm", ".xlsm", ".lnk", ".hta", ".svg")


def _host(u: str) -> str:
    try:
        return (urlparse(u).hostname or "").lower()
    except Exception:
        return ""


def _domain(addr_or_host: str) -> str:
    h = addr_or_host.split("@")[-1].lower().strip(".")
    parts = h.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else h


def _is_ip(h: str) -> bool:
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _count(text: str, words: List[str]) -> int:
    t = text.lower()
    return sum(t.count(w) for w in words)


FEATURE_NAMES: List[str] = []


def extract(m: Message) -> Dict[str, float]:
    text = m.text or ""
    tl = text.lower()
    subj = (m.subject or "").lower()
    hosts = [h for h in (_host(u) for u in m.urls) if h]
    doms = {_domain(h) for h in hosts}
    from_dom = _domain(m.from_addr) if m.from_addr else ""
    words = re.findall(r"[A-Za-z]{2,}", text)
    f: Dict[str, float] = {}
    # links
    f["n_urls"] = len(m.urls)
    f["n_link_domains"] = len(doms)
    f["url_ip_host"] = float(any(_is_ip(h) for h in hosts))
    f["url_shortener"] = float(any(h in SHORTENERS for h in hosts))
    f["url_at_sign"] = float(any("@" in (urlparse(u).netloc or "") for u in m.urls))
    f["url_punycode"] = float(any("xn--" in h for h in hosts))
    f["url_risky_tld"] = float(any(h.rsplit(".", 1)[-1] in RISKY_TLDS for h in hosts))
    f["url_max_len"] = max((len(u) for u in m.urls), default=0)
    f["url_max_dots"] = max((h.count(".") for h in hosts), default=0)
    f["links_off_sender_domain"] = (sum(_domain(h) != from_dom for h in hosts) / len(hosts)) if hosts and from_dom else 0.0
    mism = 0
    for href, anchor in m.links:
        shown = re.search(r"([a-z0-9-]+\.)+[a-z]{2,}", anchor.lower())
        if shown and _host(href) and _domain(shown.group(0)) != _domain(_host(href)):
            mism += 1
    f["anchor_href_mismatch"] = mism
    f["n_anchor_links"] = len(m.links)
    # HTML
    h = m.html.lower()
    f["html_only"] = float(bool(m.html) and not m.has_plain)
    f["has_form"] = float("<form" in h)
    f["has_script"] = float("<script" in h)
    f["has_iframe"] = float("<iframe" in h)
    f["n_images"] = h.count("<img")
    f["hidden_text"] = float("display:none" in h.replace(" ", "") or "font-size:0" in h.replace(" ", ""))
    f["html_text_ratio"] = (len(m.html) / (len(text) + 1)) if m.html else 0.0
    # wording
    f["n_words"] = len(words)
    f["urgent_words"] = _count(tl, URGENT) + 2 * _count(subj, URGENT)
    f["credential_words"] = _count(tl, CREDENTIAL) + 2 * _count(subj, CREDENTIAL)
    f["money_words"] = _count(tl, MONEY) + 2 * _count(subj, MONEY)
    f["click_here"] = _count(tl, ["click here", "click below", "click the link", "tap here", "open the attached"])
    f["upper_ratio"] = (sum(w.isupper() for w in words) / len(words)) if words else 0.0
    f["exclamations"] = text.count("!") + 2 * (m.subject or "").count("!")
    f["subject_len"] = len(m.subject or "")
    f["subject_has_re_fw"] = float(subj.startswith(("re:", "fw:", "fwd:")))
    # sender consistency
    name = (m.from_name or "").lower()
    brand_in_name = [b for b in BRANDS if b in name or b in subj]
    f["brand_mentioned"] = float(bool(brand_in_name))
    f["brand_not_in_sender_domain"] = float(bool(brand_in_name) and not any(b.replace(" ", "") in from_dom for b in brand_in_name))
    f["reply_to_differs"] = float(bool(m.reply_to) and _domain(m.reply_to) != from_dom)
    f["return_path_differs"] = float(bool(m.return_path) and _domain(m.return_path) != from_dom)
    f["display_name_has_address"] = float("@" in name)
    f["n_recipients"] = m.n_recipients
    f["from_freemail"] = float(from_dom in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com", "mail.ru", "gmx.com"})
    # attachments
    f["n_attachments"] = len(m.attachments)
    f["risky_attachment"] = float(any(fn.endswith(RISKY_EXT) for fn, _ in m.attachments))
    f["attachment_pdf"] = float(any(fn.endswith(".pdf") or ct == "application/pdf" for fn, ct in m.attachments))
    # authentication results, when present (fail is informative; absence is not used)
    f["auth_fail"] = float(bool(re.search(r"(spf|dkim|dmarc)=(fail|softfail)", m.auth)))
    global FEATURE_NAMES
    if not FEATURE_NAMES:
        FEATURE_NAMES = list(f)
    return f


# Fields that only tell *when* or *through which systems* a message travelled. Used only in the
# shortcut test, to show how a model can score well on a biased benchmark for the wrong reason.
SHORTCUT_HEADERS = ["DKIM-Signature", "ARC-Seal", "X-MS-Exchange-Organization-AuthAs", "X-Microsoft-Antispam",
                    "X-Google-Smtp-Source", "Feedback-ID", "List-Unsubscribe-Post", "X-Mailer", "X-Spam-Status"]


def shortcut_features(raw: bytes) -> Dict[str, float]:
    import email
    import email.policy
    from email.utils import parsedate_to_datetime
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    f = {("has_" + h.replace("-", "_").lower()): float(msg.get(h) is not None) for h in SHORTCUT_HEADERS}
    try:
        f["date_year"] = float(parsedate_to_datetime(msg.get("Date")).year)
    except Exception:
        f["date_year"] = 0.0
    f["n_received"] = float(len(msg.get_all("Received") or []))
    return f
