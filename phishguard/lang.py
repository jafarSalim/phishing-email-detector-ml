"""A tiny English detector (share of common English function words). Used to keep the
benchmark monolingual, so that the models cannot separate the classes by language."""
import re

EN = set("the and to of a in is you for that your this it on with be are we have as at or our from by will not "
         "can if an please all more has was this here".split())


def english_share(text: str) -> float:
    w = re.findall(r"[a-zA-Z']+", text.lower())[:400]
    return sum(x in EN for x in w) / len(w) if w else 0.0


def is_english(text: str, threshold: float = 0.12) -> bool:
    return english_share(text) >= threshold
