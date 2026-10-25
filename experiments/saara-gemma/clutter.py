"""Deterministic web-clutter injection for the cleaning skill (AUDIT ITEM, spec §2).

Clean paragraphs from fineweb-edu are wrapped and interleaved with typical scraping debris.
The target is always the untouched original text, so the model only learns to remove clutter.
"""
from __future__ import annotations

import random

NAV = [
    "Home | About | Contact | Blog | Login",
    "Skip to main content",
    "Menu ☰  Search 🔍",
    "You are here: Home > Articles > Education",
    "Share on Facebook | Tweet | Pin it | Email",
]
BANNERS = [
    "We use cookies to improve your experience. By continuing you agree to our Cookie Policy. [Accept] [Decline]",
    "Subscribe to our newsletter! Enter your email: ________ [Sign up]",
    "ADVERTISEMENT",
    "Sponsored content — Click here to learn more",
    "This article is for subscribers only. Already a member? Log in.",
]
FOOTERS = [
    "© 2024 All rights reserved. Privacy Policy | Terms of Use",
    "Related posts: 10 Tips You Need to Know | Read more »",
    "Comments (0) — Leave a reply. Your email address will not be published.",
    "Tags: education, learning, study",
]


def _link(rng: random.Random) -> str:
    word = rng.choice(["Read more", "Click here", "Source", "Next page", "Download PDF"])
    return f"[{word}](https://example.{rng.choice(['com', 'org', 'net'])}/{rng.randint(1000, 9999)})"


def _image(rng: random.Random) -> str:
    return f"![image-{rng.randint(1, 99)}](https://cdn.example.com/img/{rng.randint(10000, 99999)}.jpg)"


def _html_noise(text: str, rng: random.Random) -> str:
    tags = [("<div class=\"content\">", "</div>"), ("<p>", "</p>"), ("<span style=\"font-size:14px\">", "</span>")]
    open_tag, close_tag = rng.choice(tags)
    return f"{open_tag}{text}{close_tag}"


def add_clutter(text: str, seed: int) -> tuple[str, list[str]]:
    """Return (cluttered_text, list_of_clutter_kinds). The same seed always gives the same output."""
    rng = random.Random(seed)
    kinds: list[str] = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    if rng.random() < 0.8:
        out.extend(rng.sample(NAV, k=rng.randint(1, 2)))
        kinds.append("navigation")
    for index, para in enumerate(paragraphs):
        if rng.random() < 0.3:
            para = _html_noise(para, rng)
            kinds.append("html")
        out.append(para)
        if rng.random() < 0.35:
            out.append(rng.choice([_link(rng), _image(rng)]))
            kinds.append("link_or_image")
        if index == 0 and rng.random() < 0.5:
            out.append(rng.choice(BANNERS))
            kinds.append("banner")
    if rng.random() < 0.8:
        out.extend(rng.sample(FOOTERS, k=rng.randint(1, 2)))
        kinds.append("footer")
    joined = "\n\n".join(out)
    if rng.random() < 0.3:
        joined = joined.replace("\n\n", "\n\n\n   ")
        kinds.append("whitespace")
    return joined, sorted(set(kinds))


CLUTTER_MARKERS = [*NAV, *BANNERS, *FOOTERS, "](https://", "![", "<div", "<p>", "<span"]


def clutter_left(text: str) -> int:
    """Count injected clutter markers still present in a model output."""
    return sum(marker in text for marker in CLUTTER_MARKERS)
