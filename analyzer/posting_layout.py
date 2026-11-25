"""Read the parts of a job posting that follow a fixed layout, without a model.

USAJOBS postings repeat the same wording from one announcement to the next:

    You must be a U.S. Citizen to apply for this job
    You must have one year of specialized experience equivalent to ... GS-11
    Attention to Detail - Is thorough when performing work and ...

A fixed phrase does not need a model to find it, and reading it here means the
value is in the posting by construction. The model is only asked for the parts
that are written differently in every posting.
"""

from __future__ import annotations

import re

# The posting fields that hold requirements, in the order they are joined, with
# the label written in front of each. Title, agency, salary, and locations are not
# requirements, so they are left out.
POSTING_SECTIONS = (
    ("summary", "Summary"),
    ("duties", "Duties"),
    ("qualifications", "Qualifications"),
    ("education", "Education"),
    ("requirements", "Requirements"),
)


def posting_text(posting: dict) -> str:
    """Join the requirement fields of a saved posting into one labelled text.

    The result is what the model reads and what every extracted value is checked
    against. A field that is missing, empty, or a list (`duties`) is handled.
    """
    parts = []
    for key, label in POSTING_SECTIONS:
        value = posting.get(key)
        if isinstance(value, list):
            value = "\n".join(str(item) for item in value if item)
        if value and str(value).strip():
            parts.append(f"{label}:\n{str(value).strip()}")
    return "\n\n".join(parts)


# "U.S. Citizen", "US Citizens", "United States Citizenship", "U.S. Citizen or U.S. National".
_CITIZENSHIP = re.compile(
    r"\b(?:U\.\s?S\.|US|United States)[ \t]+(?:Citizen(?:ship|s)?|National)"
    r"(?:[ \t]+or[ \t]+(?:U\.\s?S\.|US|United States)[ \t]+Nationals?)?",
    re.IGNORECASE,
)

# A security clearance the posting asks for: "Secret clearance", "Top Secret/SCI security clearance".
_CLEARANCE = re.compile(
    r"\b(?:Top[ \t]+Secret(?:[ \t]*/[ \t]*SCI)?|Secret)[ \t]+(?:security[ \t]+)?clearance\b",
    re.IGNORECASE,
)


def stated_work_authorization(text: str) -> str | None:
    """The citizenship (or, failing that, clearance) the posting requires, as written, or None."""
    for pattern in (_CITIZENSHIP, _CLEARANCE):
        match = pattern.search(text)
        if match:
            return " ".join(match.group(0).split())
    return None


_NUMBER = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"

# "one year of specialized experience", "one (1) full year of specialized experience",
# "2 years of experience developing ...". Group "amount" is the part kept, e.g. "one year".
_MIN_EXPERIENCE = re.compile(
    rf"\b(?P<amount>{_NUMBER}(?:[ \t]*\(\d+\))?(?:[ \t]+full)?[ \t]+years?)[ \t]+of[ \t]+"
    r"(?P<kind>specialized[ \t]+|progressively[ \t]+responsible[ \t]+|professional[ \t]+|general[ \t]+)?experience\b"
    r"(?P<after>[ \t]+refers[ \t]+to)?",
    re.IGNORECASE,
)


def stated_min_experience(text: str) -> str | None:
    """The minimum experience the posting requires, such as "one year", or None.

    "Specialized experience" is preferred, because that is the requirement federal
    postings grade on. Sentences that only define a term ("One year of experience
    refers to full-time work") are ignored.
    """
    matches = [match for match in _MIN_EXPERIENCE.finditer(text) if not match["after"]]
    specialized = [match for match in matches if match["kind"] and "specialized" in match["kind"].casefold()]
    chosen = (specialized or matches or [None])[0]
    return " ".join(chosen["amount"].split()) if chosen else None


# An OPM competency line: "Attention to Detail - Is thorough when ...",
# "Customer Service - Works with clients ...". The name starts a sentence (after a
# line break, ". ", ": ", or a list marker such as "a) "), is two to five words, each capitalised except small
# joining words, and is followed by " - " and a verb ending in "s".
_COMPETENCY = re.compile(
    r"(?:(?<=^)|(?<=\n)|(?<=[.:;)•] ))"
    r"(?P<name>[A-Z][a-z]+(?:[ \t]+(?:(?:to|of|and|&)[ \t]+)?[A-Z][a-z]+){1,4})"
    r"[ \t]+[-–][ \t]+(?:Is|[A-Z][a-z]+s)\b",
    re.MULTILINE,
)


def section(text: str, label: str) -> str:
    """The part of a `posting_text` under `label` ("Qualifications"), or all of `text` if it has no such label."""
    labels = "|".join(re.escape(name) for _, name in POSTING_SECTIONS)
    match = re.search(rf"^{re.escape(label)}:\n(?P<body>.*?)(?=^(?:{labels}):\n|\Z)", text, re.MULTILINE | re.DOTALL)
    return match["body"] if match else text


def stated_competencies(text: str) -> list[str]:
    """Competency names the Qualifications section lists as "Name - definition", in order, each once.

    These are the skills federal postings require of every applicant (Attention to
    Detail, Customer Service, Oral Communication, Problem Solving, and others).
    Only the Qualifications section is read: elsewhere the same layout introduces
    duties ("Accident Investigations - Conducts ...") or rules ("Selective Service
    Registration - Males born after ..."), which are not skills.
    """
    found: list[str] = []
    for match in _COMPETENCY.finditer(section(text, "Qualifications")):
        name = " ".join(match["name"].split())
        if name.casefold() not in {item.casefold() for item in found}:
            found.append(name)
    return found
