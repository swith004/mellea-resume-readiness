"""Check that extracted items really appear in the source text.

A small model sometimes returns a skill the resume never mentions (in testing it
copied the example skills out of its own instructions). These checks are plain
Python, so the result does not depend on the model being honest.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# A word is a run of letters and digits, plus "+" and "#" so that "C++" and "C#"
# stay distinct from "C".
_WORD = re.compile(r"[a-z0-9+#]+")


def normalize(text: str) -> str:
    """Lowercase and reduce to words separated by single spaces, with a space at each end.

    Punctuation and spacing differences disappear ("Node.js" and "node js" compare
    equal), accents are removed ("Naïve" matches "Naive"), and the padding lets a
    whole-word search be a plain substring test.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " " + " ".join(_WORD.findall(plain.casefold())) + " "


def is_grounded(item: str, source: str) -> bool:
    """True if `item` appears in `source` as whole words, ignoring case, spacing, and punctuation.

    Whole words matter: "Java" is not grounded by a resume that only says "JavaScript".
    """
    return not ungrounded([item], source)


def ungrounded(items: list[str], source: str) -> list[str]:
    """The items that do not appear in `source`. An empty list means everything is grounded."""
    haystack = normalize(source)
    return [item for item in items if not normalize(item).strip() or normalize(item) not in haystack]


# How many leading words `ground` may drop as a category label.
MAX_LABEL_WORDS = 2

# Words a name does not start with. A remainder that starts with one is a fragment.
_JOINING_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


def _is_name(text: str) -> bool:
    """True if `text` can stand as a name on its own once a label is dropped from it.

    Two or more words can ("project management"). One word can only if it is written
    like a name, with a capital, a digit, or a symbol ("MySQL", "HTTP", "C#"). A lone
    ordinary word ("management", "standards") says nothing by itself.
    """
    words = text.split()
    if not words or words[0].casefold().strip("(") in _JOINING_WORDS:
        return False
    return len(words) > 1 or not text.islower()


def ground(item: str, source: str) -> str | None:
    """Return `item` if it appears in `source`, else the part after a leading label if that does, else None.

    The model sometimes puts a category label in front of a real entry, turning the
    line "Database: MySQL" into the skill "Database MySQL". Dropping the label
    recovers "MySQL" instead of losing the skill.

    Only a label is dropped: at most `MAX_LABEL_WORDS` words, and what is left must
    still be a name. A phrase the model put together itself is not rescued by
    cutting it down to its last word. "database management" is not turned into
    "management" when the source never says "database management"; it is reported
    as not found.
    """
    words = item.split()
    for start in range(min(len(words), MAX_LABEL_WORDS + 1)):
        candidate = " ".join(words[start:])
        if start and not _is_name(candidate):
            continue
        if is_grounded(candidate, source):
            return candidate
    return None


def word_overlap(text: str, source: str) -> float:
    """Fraction of the words in `text` that also appear somewhere in `source` (0.0 to 1.0).

    Used for values that are reworded rather than copied, such as a description or
    a date written in a different layout. 1.0 means every word came from the source.
    """
    words = normalize(text).split()
    if not words:
        return 0.0
    source_words = set(normalize(source).split())
    return sum(word in source_words for word in words) / len(words)


# A line such as "Python - Experience - 24 months" or "SQL: Experience: Less than 1 year".
# "Exprience" is how the sample resumes spell it.
_SKILL_DURATION = re.compile(
    r"^[ \t•*-]*(?P<skill>[^\n]{1,200}?)\s*[-–:]\s*Exp[a-z]*ence\s*[-–:]\s*"
    r"(?P<duration>(?:less than\s+)?\d+\+?\s*(?:years?|yrs?|months?)(?:\s+months?)?)",
    re.IGNORECASE | re.MULTILINE,
)


def stated_skill_durations(source: str) -> list[tuple[str, str]]:
    """Find lines that state how long a skill has been used, as (skill, duration) pairs.

    Reads lines of the form "Skill - Experience - 24 months" directly from the text.
    The model caught only about half of these, and a fixed layout does not need a model.
    A line naming several skills ("Java, Javascript - Experience - 6 months") gives
    one pair per skill.
    """
    pairs = []
    for match in _SKILL_DURATION.finditer(source):
        duration = " ".join(match["duration"].split())
        for skill in match["skill"].split(","):
            skill = re.sub(r"^and\s+", "", skill.strip(" \t;:-").rstrip("."), flags=re.IGNORECASE)
            if skill:
                pairs.append((skill, duration))
    return pairs


def closest_span(item: str, source: str, cutoff: float = 0.85) -> str | None:
    """The run of words in `source` that most resembles `item`, if it is close enough.

    The model sometimes "fixes" a spelling ("Softwere" becomes "Software"), which
    makes a real value look invented. This returns the resume's own wording so the
    value can be kept as written. Returns None when nothing is at least `cutoff`
    similar (0.0 to 1.0).
    """
    size = len(item.split())
    target = normalize(item)
    words = source.split()
    best, best_score = None, cutoff
    for start in range(len(words) - size + 1):
        candidate = " ".join(words[start : start + size])
        score = SequenceMatcher(None, target, normalize(candidate)).ratio()
        if score >= best_score:
            best, best_score = candidate, score
    return best.strip(" ,;:") if best else None


_AMOUNT = r"\d+(?:\.\d+)?\+?[ \t]*(?:years?|yrs?)(?:[ \t]+(?:and[ \t]+)?\d+[ \t]*months?)?|\d+\+?[ \t]*months?"

# "5 years of experience", "3+ yrs of IT experience", "total experience: 18 months",
# "experience of over 7 years". Up to four words may sit between the amount and "experience".
_TOTAL_EXPERIENCE = [
    re.compile(rf"(?P<amount>{_AMOUNT})[ \t]+(?:of[ \t]+)?(?:[\w/&.-]+[ \t]+){{0,4}}?experience\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:total|overall)[ \t]+(?:[\w/&.-]+[ \t]+){{0,2}}?experience[ \t]*(?:of|is|:|-|–)?[ \t]*(?P<amount>{_AMOUNT})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bexperience[ \t]+of[ \t]+(?:over|about|around|more[ \t]+than|nearly)?[ \t]*(?P<amount>{_AMOUNT})",
        re.IGNORECASE,
    ),
]


def stated_total_experience(source: str) -> str | None:
    """The total work experience the resume states, such as "5 years", or None.

    Only an outright statement counts ("5 years of experience"). Lines that give
    the time spent on one skill ("Python - Experience - 24 months") are ignored,
    and nothing is added up from dates.
    """
    text = _SKILL_DURATION.sub(" ", source)
    for pattern in _TOTAL_EXPERIENCE:
        match = pattern.search(text)
        if match:
            return " ".join(match["amount"].split())
    return None
