"""phishguard: explainable phishing-email detection from raw .eml messages.

parse      raw RFC 822 message -> structured fields (text, HTML, links, attachments, auth results)
features   39 interpretable features (links, HTML, wording, sender consistency, attachments),
           deliberately free of fields that only encode *when* or *through which servers* a
           message travelled (dates, Received chains, Message-ID formats)
text       TF-IDF model of the visible words
model      gradient boosting on the features + text model, combined; explanations per message
"""
__version__ = "1.0.0"
