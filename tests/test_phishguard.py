import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phishguard import features, lang, parse  # noqa: E402

PHISH = b"""From: "PayPal Security" <service@paypa1-secure.xyz>
Reply-To: help@other-domain.ru
To: victim@example.org
Subject: URGENT: your account is suspended
Content-Type: text/html; charset=utf-8

<html><body><p>Dear customer, we detected unusual activity. Verify your account within 24 hours or it will be locked.</p>
<a href="http://198.51.100.7/login">https://www.paypal.com/signin</a>
<form action="http://198.51.100.7/p"><input name="password"></form>
<div style="display:none">hidden</div></body></html>
"""

HAM = b"""From: Alice <alice@example.org>
To: bob@example.org
Subject: Re: meeting notes
Content-Type: text/plain; charset=utf-8

Hi Bob, the notes from today are below. See you on Thursday.
https://example.org/wiki/notes
"""


def test_parse_and_features_flag_the_obvious_signs():
    m = parse.parse(PHISH)
    f = features.extract(m)
    assert m.subject.startswith("URGENT")
    assert f["url_ip_host"] == 1 and f["anchor_href_mismatch"] == 1 and f["has_form"] == 1 and f["hidden_text"] == 1
    assert f["brand_not_in_sender_domain"] == 1 and f["reply_to_differs"] == 1 and f["urgent_words"] >= 4
    assert f["url_risky_tld"] == 0           # the link is a bare IP; the risky TLD is only in the sender


def test_benign_message_has_quiet_features():
    f = features.extract(parse.parse(HAM))
    assert f["anchor_href_mismatch"] == 0 and f["has_form"] == 0 and f["urgent_words"] == 0 and f["subject_has_re_fw"] == 1


def test_feature_set_is_stable_and_has_no_time_fields():
    names = list(features.extract(parse.parse(HAM)).keys())
    assert len(names) == 39
    assert not any("date" in n or "received" in n for n in names)


def test_language_filter():
    assert lang.is_english("Please verify your account and confirm the payment to us")
    assert not lang.is_english("Bitte bestätigen Sie Ihr Konto innerhalb von 24 Stunden")
