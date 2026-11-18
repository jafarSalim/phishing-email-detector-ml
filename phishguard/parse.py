"""Parse a raw e-mail into the parts the detector looks at."""
from __future__ import annotations

import email
import email.policy
import html
import re
from dataclasses import dataclass, field
from email.utils import getaddresses, parseaddr
from typing import List, Tuple

URL_RE = re.compile(r"""https?://[^\s<>"'()\[\]{}]+""", re.I)
HREF_RE = re.compile(r"""<a\b[^>]*?href\s*=\s*["']?([^"' >]+)[^>]*>(.*?)</a>""", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")
STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)


@dataclass
class Message:
    subject: str = ""
    from_name: str = ""
    from_addr: str = ""
    reply_to: str = ""
    return_path: str = ""
    n_recipients: int = 0
    text: str = ""                    # visible text (plain part, or HTML with tags removed)
    html: str = ""                    # raw HTML part(s)
    links: List[Tuple[str, str]] = field(default_factory=list)   # (href, anchor text)
    urls: List[str] = field(default_factory=list)                # every URL seen in text or HTML
    attachments: List[Tuple[str, str]] = field(default_factory=list)  # (filename, content type)
    auth: str = ""                    # Authentication-Results / Received-SPF text, lower case
    has_plain: bool = False


def _decode(part) -> str:
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True) or b""
        return payload.decode(part.get_content_charset() or "latin-1", errors="replace")


def html_to_text(h: str) -> str:
    h = STYLE_RE.sub(" ", h)
    h = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", h, flags=re.I)
    return html.unescape(TAG_RE.sub(" ", h))


def parse(raw: bytes) -> Message:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    m = Message()
    try:
        m.subject = str(msg.get("Subject", "") or "")
    except Exception:
        m.subject = ""
    name, addr = parseaddr(str(msg.get("From", "") or ""))
    m.from_name, m.from_addr = name, addr.lower()
    m.reply_to = parseaddr(str(msg.get("Reply-To", "") or ""))[1].lower()
    m.return_path = parseaddr(str(msg.get("Return-Path", "") or ""))[1].lower()
    rcpts = []
    for h in ("To", "Cc"):
        try:
            rcpts += getaddresses([str(v) for v in msg.get_all(h, [])])
        except Exception:
            pass
    m.n_recipients = len(rcpts)
    m.auth = " ".join(str(v) for h in ("Authentication-Results", "Received-SPF", "ARC-Authentication-Results")
                      for v in (msg.get_all(h, []) or [])).lower()
    texts, htmls = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        disp = (part.get("Content-Disposition") or "").lower()
        fname = part.get_filename()
        if fname or "attachment" in disp:
            m.attachments.append(((fname or "").lower(), ctype))
            continue
        if ctype == "text/plain":
            texts.append(_decode(part))
            m.has_plain = True
        elif ctype == "text/html":
            htmls.append(_decode(part))
    m.html = "\n".join(htmls)
    m.text = "\n".join(texts) if texts else html_to_text(m.html)
    m.links = [(href, html_to_text(a).strip()) for href, a in HREF_RE.findall(m.html)]
    m.urls = list(dict.fromkeys(URL_RE.findall(m.text) + [h for h, _ in m.links if h.lower().startswith("http")]))
    return m
