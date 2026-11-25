"""Turn a job posting into JobRequirements with Mellea @generative functions.

Use it from code:

    from analyzer.extract_posting import extract_requirements
    requirements = extract_requirements(posting)       # a posting dict from data/postings_*.json
    requirements = extract_requirements(posting_text)  # or the posting as plain text

Or from the command line, run from the project root:

    uv run python -m analyzer.extract_posting --index 0
    uv run python -m analyzer.extract_posting --postings data/postings_construction_miami-fl.json --index 3
    uv run python -m analyzer.extract_posting --file tests/fixtures/sample_posting.json

Built the same way as `extract_resume`: focused passes, fixed wording read in
plain Python, and every value checked against the posting before it is returned.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from mellea import MelleaSession, generative, start_session
from pydantic import Field, ValidationError

# Shared with the resume extractor so both halves use the same model settings and
# the same rules. (Candidates to move into a shared module once PR #18 is merged.)
from analyzer.extract_resume import (
    _WORK_AUTHORIZATION,
    MAX_SKILL_WORDS,
    MIN_DESCRIPTION_OVERLAP,
    MODEL_OPTIONS,
    SPOKEN_LANGUAGES,
    ExtractionError,
)
from analyzer.grounding import closest_span, ground, is_grounded, normalize, word_overlap
from analyzer.posting_layout import (
    posting_text,
    section,
    stated_competencies,
    stated_min_experience,
    stated_work_authorization,
)
from analyzer.schemas import JobRequirements, Schema

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSTINGS = PROJECT_ROOT / "data" / "postings_information-technology_miami-fl.json"

# An education requirement must name a degree, a diploma, or schooling. The postings'
# boilerplate "There is no educational substitution at this grade level" does not.
_EDUCATION = re.compile(
    r"\b(?:degrees?|bachelor'?s?|master'?s?|associate'?s?|doctora(?:te|l)|ph\.?\s?d|diploma|high[ \t]+school|ged"
    r"|b\.?\s?[as]\.?|m\.?\s?[as]\.?|college|university|coursework|course[ \t]+work|semester[ \t]+hours"
    r"|education)\b",
    re.IGNORECASE,
)
_NO_EDUCATION = re.compile(r"\bno[ \t]+educational[ \t]+substitution\b|\bnot[ \t]+required\b", re.IGNORECASE)

# The posting is read in focused passes, as in extract_resume: asked for
# everything at once, a small model mixes sections up. Education gets a pass of
# its own, over only the sections that state it.


class _SkillsSection(Schema):
    required_skills: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 40})
    preferred_skills: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 25})


class _EducationSection(Schema):
    education: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 5})


class _QualificationsSection(Schema):
    qualifications: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 15})
    education: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 5})
    certifications: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 10})
    languages: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 5})


@generative
def extract_posting_skills(posting_text: str) -> _SkillsSection:
    """List the skills the job posting asks for.

    Copy from the posting. Never add a skill that is not written there. Ignore
    text about pay, benefits, telework, probation, identification, background
    checks, and physical demands: none of it is a skill.

    - required_skills: every skill, tool, technology, programming language,
      and competency the posting requires, such as a named software package,
      a method, or a competency like "Customer Service". One skill per item,
      a few words at most, spelled as in the posting. Split lists apart: "Alpha,
      Beta, and Gamma" becomes "Alpha", "Beta", "Gamma". Never repeat a skill.
      Do not include sentences, job titles, grades, or durations.
    - preferred_skills: only skills the posting calls preferred, desired,
      desirable, or a plus. Use an empty list if it calls none of them that.
    """


@generative
def extract_posting_qualifications(posting_text: str) -> _QualificationsSection:
    """List the qualifications the job posting requires.

    Copy from the posting. Never guess or add anything that is not written
    there. Use an empty list when the posting states nothing for an item.
    Ignore text about pay, benefits, telework, probation, identification,
    background checks, and physical demands.

    - qualifications: each experience or qualification the applicant must
      have, as a short phrase in the posting's own words, one per item. For
      "specialized experience" defined as several duties, give each duty as its
      own item. At most one sentence per item.
    - education: each degree, diploma, or schooling the posting requires, as
      written. A sentence saying there is no educational substitution is not
      an education requirement.
    - certifications: each certification or license the posting requires,
      as written.
    - languages: spoken languages the posting requires, such as Spanish.
      Programming languages are not spoken languages.
    """


@generative
def extract_posting_education(posting_text: str) -> _EducationSection:
    """List the education the job posting asks for.

    Copy from the posting. Never add anything that is not written there.

    education: each degree, diploma, or amount of schooling an applicant must
    have or may use to qualify, such as a bachelor's degree in a named field,
    a master's degree, or a number of semester hours. One per item, a short
    phrase in the posting's own words. Leave out rules about transcripts,
    accreditation, and foreign education: they are not requirements a degree
    meets. Use an empty list if the posting names no degree or schooling.
    """


def _education_text(text: str) -> str:
    """The parts of a posting that state education: its Qualifications and Education sections.

    Asked about education with the whole posting in front of it, the model
    returned nothing for 20 of the 21 saved postings that name a degree.
    """
    parts = [section(text, "Qualifications")]
    if "\nEducation:\n" in f"\n{text}":
        parts.append(section(text, "Education"))
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _source_text(posting: dict | str) -> str:
    if isinstance(posting, dict):
        return posting_text(posting)
    return posting or ""


def extract_requirements(posting: dict | str, m: MelleaSession | None = None) -> JobRequirements:
    """Extract JobRequirements from a posting dict (as saved in data/) or from posting text.

    The result is checked against the posting before it is returned, so nothing in
    it is invented: see `ground_requirements` for the rules.

    Pass an existing session as `m` to reuse it across calls; otherwise a new
    default session (Ollama + Granite) is started.
    """
    text = _source_text(posting)
    if not text.strip():
        raise ValueError("posting text is empty")
    session = m if m is not None else start_session()
    try:
        skills = extract_posting_skills(session, posting_text=text, model_options=MODEL_OPTIONS)
        quals = extract_posting_qualifications(session, posting_text=text, model_options=MODEL_OPTIONS)
        schooling = extract_posting_education(session, posting_text=_education_text(text), model_options=MODEL_OPTIONS)
    except ValidationError as e:
        # Usually the answer was cut off at the token cap, leaving incomplete JSON.
        raise ExtractionError(f"model returned an unreadable answer: {e.errors()[0]['msg']}") from e

    requirements = JobRequirements(
        # Competencies written as "Name - definition" are read straight from the text;
        # the model's list adds the skills written in prose. Repeats are dropped.
        required_skills=[*stated_competencies(text), *skills.required_skills],
        preferred_skills=skills.preferred_skills,
        qualifications=quals.qualifications,
        education=[*schooling.education, *quals.education],
        certifications=quals.certifications,
        languages=quals.languages,
        # Fixed wording, read from the text rather than asked of the model.
        min_experience=stated_min_experience(text),
        work_authorization=stated_work_authorization(text),
    )
    return ground_requirements(requirements, text)


def ground_requirements(
    requirements: JobRequirements, text: str, removed: list[str] | None = None
) -> JobRequirements:
    """Return a copy of `requirements` with everything that is not in `text` removed.

    - Skills must appear in the posting as whole words. A skill with a label in
      front ("Software Python") is trimmed to the part that appears, and a "skill"
      longer than a few words is dropped (reported as "a sentence, not a skill").
    - Certifications and languages must appear word for word; if the model only
      changed a spelling, the posting's wording is used. A language must also be
      a known spoken language.
    - A qualification or education entry is dropped when too few of its words come
      from the posting, and an education entry must name a degree or schooling.
    - Minimum experience and work authorization must come word for word from the
      posting, and work authorization must mention citizenship, a visa, or a clearance.

    Everything removed is logged as a warning and, if `removed` is given, appended
    to it, so a caller can count what the model made up. An entry ending in a reason
    in parentheses, such as "(a sentence, not a skill)", was in the posting but in
    the wrong field; an entry with no reason was not in the posting at all.
    """
    dropped: list[str] = []

    def exact(value: str | None, label: str) -> str | None:
        if value is None or is_grounded(value, text):
            return value
        as_written = closest_span(value, text)
        if as_written is None:
            dropped.append(f"{label} {value!r}")
        return as_written

    def loose(value: str | None, label: str, minimum: float = 1.0) -> str | None:
        if value is None or word_overlap(value, text) >= minimum:
            return value
        dropped.append(f"{label} {value!r}")
        return None

    def belongs(value: str | None, pattern: re.Pattern[str], label: str) -> str | None:
        if value is None or pattern.search(value):
            return value
        dropped.append(f"{label} {value!r} (not a {label})")
        return None

    def skills(values: list[str], label: str) -> list[str]:
        kept = []
        for value in values:
            found = ground(value, text)
            if found is None:
                dropped.append(f"{label} {value!r}")
            elif len(found.split()) > MAX_SKILL_WORDS:
                # In the posting, but a sentence rather than a skill (usually a duty
                # the model also listed under qualifications). Not an invented value.
                dropped.append(f"{label} {value!r} (a sentence, not a skill)")
            else:
                kept.append(found)
        return kept

    def education(value: str) -> str | None:
        value = loose(value, "education", MIN_DESCRIPTION_OVERLAP)
        if value and _NO_EDUCATION.search(value):
            dropped.append(f"education {value!r} (not a requirement)")
            return None
        return belongs(value, _EDUCATION, "education")

    def language(value: str) -> str | None:
        value = exact(value, "language")
        if value and normalize(value).strip() not in SPOKEN_LANGUAGES:
            dropped.append(f"language {value!r} (not a spoken language)")
            return None
        return value

    data = {
        "required_skills": skills(requirements.required_skills, "required skill"),
        "preferred_skills": skills(requirements.preferred_skills, "preferred skill"),
        "qualifications": [
            q for q in (loose(item, "qualification", MIN_DESCRIPTION_OVERLAP) for item in requirements.qualifications) if q
        ],
        "education": [e for e in (education(item) for item in requirements.education) if e],
        "certifications": [c for c in (exact(item, "certification") for item in requirements.certifications) if c],
        "languages": [name for name in (language(item) for item in requirements.languages) if name],
        "min_experience": loose(requirements.min_experience, "minimum experience"),
        "work_authorization": belongs(
            loose(requirements.work_authorization, "work authorization"), _WORK_AUTHORIZATION, "work authorization"
        ),
    }
    # A spoken language belongs under languages, not skills.
    spoken = {normalize(name) for name in data["languages"]}
    for key in ("required_skills", "preferred_skills"):
        data[key] = [skill for skill in data[key] if normalize(skill) not in spoken]

    if dropped:
        logger.warning("Removed %d value(s) the posting does not support: %s", len(dropped), "; ".join(dropped))
        if removed is not None:
            removed.extend(dropped)
    # Validating again drops blanks and repeats, and keeps required and preferred apart.
    return JobRequirements.model_validate(data)


def _read_posting(args: argparse.Namespace) -> dict | str:
    if args.file is not None:
        if args.file.suffix.lower() == ".json":
            posting = json.loads(args.file.read_text(encoding="utf-8"))
            if not isinstance(posting, dict):
                raise ValueError(f"{args.file} does not hold a single posting object")
            return posting
        return args.file.read_text(encoding="utf-8")
    payload = json.loads(args.postings.read_text(encoding="utf-8"))
    postings = payload.get("postings") if isinstance(payload, dict) else None
    if not isinstance(postings, list):
        raise ValueError(f"{args.postings} has no \"postings\" list")
    if not 0 <= args.index < len(postings):
        raise IndexError(f"--index {args.index} out of range (0-{len(postings) - 1})")
    return postings[args.index]


def main(argv: list[str] | None = None) -> int:
    """Extract one posting and print it as JSON. Exit code 0 = fully populated, 1 = not."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="Extract JobRequirements from one job posting.")
    parser.add_argument(
        "--postings", type=Path, default=DEFAULT_POSTINGS, metavar="PATH",
        help="a saved postings file (default: the IT postings in data/)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, default=0, metavar="I", help="posting #I in --postings (default 0)")
    source.add_argument("--file", type=Path, metavar="PATH", help="one posting: a .json posting object or a text file")
    args = parser.parse_args(argv)

    try:
        posting = _read_posting(args)
    except (OSError, IndexError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    if isinstance(posting, dict) and posting.get("title"):
        print(f"Posting: {posting['title']} ({posting.get('organization') or 'unknown agency'})", file=sys.stderr)
    try:
        requirements = extract_requirements(posting)
    except (ExtractionError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print(requirements.model_dump_json(indent=2))

    missing = requirements.missing_sections()
    if missing:
        print(f"Not fully populated; empty sections: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(
        f"Fully populated: {len(requirements.required_skills)} required skills, "
        f"{len(requirements.preferred_skills)} preferred skills, {len(requirements.qualifications)} qualifications, "
        f"{len(requirements.certifications)} certifications",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
