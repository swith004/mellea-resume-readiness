"""Read the parts of a resume that follow a fixed layout, without a model.

Some resumes label each employer, for example:

    company - Acme Corp
    description - Built the billing system. Jan 2019 to Mar 2021.

Asked to find every employer in a long resume, the model missed about four in
ten. A labelled line does not need a model to find it, so the labels are read
here and the model is only asked to fill in the details of each job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# "company - Acme", "Company: Acme", "Employer: Acme", "Organisation name - Acme"
_EMPLOYER_LABEL = re.compile(
    r"^[ \t•*-]*(?:company|employer|organi[sz]ation)(?:[ \t]+name)?[ \t]*[-–:][ \t]*(?P<name>[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)

# The label that introduces the text about the job: "description - ..."
_BODY_LABEL = re.compile(r"^\s*(?:description|details|responsibilities)[ \t]*[-–:][ \t]*", re.IGNORECASE)

# A line that starts a different section, which ends the text about the last job.
_SECTION_HEADING = re.compile(
    r"^[ \t]*(?:technical[ \t]+)?(?:skills?|education|projects?|certifications?|summary|objective|achievements?"
    r"|awards?|references?|languages?|interests|hobbies)(?:[ \t]+details)?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# An employer name is a name, not a sentence.
MAX_EMPLOYER_WORDS = 10


@dataclass(frozen=True)
class LabeledJob:
    """One employer the resume labels, with the text that follows it."""

    organization: str | None  # None when the label is there but the name is blank
    body: str  # the text about this job, up to the next employer label


def labeled_jobs(text: str) -> list[LabeledJob]:
    """Find every labelled employer in `text`, in order, with the text that belongs to it.

    Returns an empty list when the resume does not label its employers. An employer
    listed twice in a row with the same text is returned once.
    """
    labels = list(_EMPLOYER_LABEL.finditer(text))
    jobs: list[LabeledJob] = []
    seen: set[tuple[str, str]] = set()
    for position, label in enumerate(labels):
        end = labels[position + 1].start() if position + 1 < len(labels) else len(text)
        body = text[label.end() : end]
        heading = _SECTION_HEADING.search(body)
        if heading:
            body = body[: heading.start()]
        body = _BODY_LABEL.sub("", body, count=1).strip()
        name = " ".join(label["name"].split()).strip(" ,;:-") or None
        if name and len(name.split()) > MAX_EMPLOYER_WORDS:
            name = None
        if name is None and not body:
            continue
        key = ((name or "").casefold(), body)
        if key not in seen:
            seen.add(key)
            jobs.append(LabeledJob(organization=name, body=body))
    return jobs
