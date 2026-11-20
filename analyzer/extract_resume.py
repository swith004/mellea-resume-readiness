"""Turn raw resume text into a ResumeProfile with Mellea @generative functions.

Use it from code:

    from analyzer.extract_resume import extract_resume
    profile = extract_resume(resume_text)

Or from the command line, run from the project root:

    uv run python -m analyzer.extract_resume --index 0
    uv run python -m analyzer.extract_resume --file path/to/resume.txt
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from mellea import MelleaSession, generative, start_session
from mellea.backends import ModelOption
from pydantic import Field, ValidationError

from analyzer.grounding import (
    closest_span,
    ground,
    is_grounded,
    normalize,
    stated_skill_durations,
    stated_total_experience,
    word_overlap,
)
from analyzer.layout import LabeledJob, labeled_jobs
from analyzer.schemas import Education, Experience, Project, ResumeProfile, Schema, SkillExperience

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """The model's answer could not be read as a ResumeProfile."""


# Settings for every extraction call:
# - Temperature 0 keeps runs repeatable.
# - Thinking is off. Granite 4.2 thinks by default, and on real resumes it spent
#   tens of thousands of tokens reasoning without ever writing the answer.
# - The context window is raised because the longest sample resumes (about
#   14,000 characters) plus the answer do not fit in Ollama's default.
# - The token cap stops a runaway generation instead of letting it run for minutes.
MODEL_OPTIONS = {
    ModelOption.TEMPERATURE: 0.0,
    ModelOption.THINKING: False,
    ModelOption.CONTEXT_WINDOW: 16384,
    ModelOption.MAX_NEW_TOKENS: 4000,
}

# A description is kept only if at least this share of its words come from the resume.
MIN_DESCRIPTION_OVERLAP = 0.8

# A skill is a name, not a sentence: anything longer than this many words is dropped.
MAX_SKILL_WORDS = 6

# The same goes for a job title.
MAX_TITLE_WORDS = 10

# The checks below keep a value out of a field it does not belong in. The model
# returned programming languages as spoken languages, job duties as achievements,
# and place names as grades, all of them real text from the resume.

# Spoken languages the extractor recognises. Anything else offered as a language is dropped.
SPOKEN_LANGUAGES = {
    "afrikaans", "albanian", "amharic", "arabic", "armenian", "assamese", "azerbaijani", "bengali", "bhojpuri",
    "bosnian", "bulgarian", "burmese", "cantonese", "catalan", "chinese", "creole", "croatian", "czech", "danish",
    "dari", "dutch", "english", "estonian", "farsi", "filipino", "finnish", "french", "georgian", "german", "greek",
    "gujarati", "haitian creole", "hausa", "hebrew", "hindi", "hungarian", "icelandic", "igbo", "indonesian",
    "irish", "italian", "japanese", "kannada", "kashmiri", "kazakh", "khmer", "konkani", "korean", "kurdish",
    "lao", "latin", "latvian", "lithuanian", "malay", "malayalam", "mandarin", "marathi", "mongolian", "nepali",
    "norwegian", "odia", "oriya", "pashto", "persian", "polish", "portuguese", "punjabi", "romanian", "russian",
    "sanskrit", "serbian", "sindhi", "sinhala", "slovak", "slovenian", "somali", "spanish", "swahili", "swedish",
    "tagalog", "tamil", "telugu", "thai", "tibetan", "tulu", "turkish", "ukrainian", "urdu", "uzbek",
    "vietnamese", "welsh", "yoruba", "zulu",
}

_ACHIEVEMENT = re.compile(
    r"\b(?:awards?|awarded|prizes?|winners?|won|medals?|medalist|ranked|rank|scholarships?|honou?rs?|honou?red"
    r"|dean'?s list|toppers?|published|publications?|patents?|fellowship|recognition|recogni[sz]ed"
    r"|certificate of (?:merit|appreciation|excellence)|(?:first|second|third|1st|2nd|3rd) (?:place|prize|rank)"
    r"|best \w+|employee of the)\b",
    re.IGNORECASE,
)

# A grade is a percentage, a labelled figure, a class, or a bare score such as "3.7" or
# "8.2/10". A bare four-digit number is a year, not a grade.
_GRADE = re.compile(
    r"^\d{1,2}(?:\.\d+)?(?:[ \t]*/[ \t]*\d+(?:\.\d+)?)?$"
    r"|\d+(?:\.\d+)?[ \t]*%|\b(?:c?gpa|percentage|percentile|percent|marks?|grade)\b"
    r"|\b(?:first|second|third)[ \t]+(?:class|division)\b|\bdistinction\b|\bcum laude\b|\bhonou?rs\b",
    re.IGNORECASE,
)

_WORK_AUTHORIZATION = re.compile(
    r"\b(?:citizen|citizenship|national|visa|clearance|permanent resident|green card|work permit|authori[sz]ed to work"
    r"|work authori[sz]ation|h-?1b|opt|ead)\b",
    re.IGNORECASE,
)

# Section headings the model sometimes returns as a job title.
_HEADINGS = {
    "roles and responsibilities",
    "role and responsibilities",
    "responsibilities",
    "job responsibilities",
    "key responsibilities",
    "job description",
    "description",
    "role",
    "roles",
    "designation",
    "work experience",
    "experience",
}


# The resume is read in focused passes instead of one. Asked for everything at
# once, the small model mixed sections up (education dates on jobs, degrees in the
# wrong field). Each pass over the whole resume returns one of these section models.
#
# Employers get extra care, because the model alone missed about four in ten:
# - If the resume labels its employers ("company - Acme"), the labels are read
#   directly and the model describes each job from that job's own text.
# - If it does not, a second pass asks only for employer names and adds any the
#   work-history pass left out.


class _SkillsSection(Schema):
    skills: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 60})
    skill_experience: list[SkillExperience] = Field(default_factory=list, json_schema_extra={"maxItems": 40})


class _WorkSection(Schema):
    experience: list[Experience] = Field(default_factory=list, json_schema_extra={"maxItems": 12})
    projects: list[Project] = Field(default_factory=list, json_schema_extra={"maxItems": 12})


class _EducationSection(Schema):
    education: list[Education] = Field(default_factory=list, json_schema_extra={"maxItems": 8})
    certifications: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 15})


class _OverviewSection(Schema):
    work_authorization: str | None = None
    languages: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 10})
    achievements: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 10})


class _EmployerList(Schema):
    employers: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": 20})


@generative
def extract_skills(resume_text: str) -> _SkillsSection:
    """List the skills in the resume text.

    Copy from the resume. Never add a skill that is not written there.

    - skills: every skill, tool, programming language, framework, and
      technology the resume names, anywhere in it. One skill per item, a few
      words at most, spelled as in the resume. Split lists apart: the line
      "Tools: Alpha, Beta, Gamma" becomes "Alpha", "Beta", "Gamma". Never
      repeat a skill. Do not include category labels, sentences, job titles,
      or durations.
    - skill_experience: only where the resume states how long a skill has
      been used, such as "Alpha - Experience - 24 months": the skill and the
      duration as written. Use an empty list if the resume gives no durations.
    """


@generative
def extract_work_history(resume_text: str) -> _WorkSection:
    """List the jobs and the projects in the resume text.

    Copy from the resume. Never guess or add anything that is not written
    there. Use null for a value the resume does not state, and an empty list
    when there are no entries.

    - experience: every employer the person has worked for, one entry per
      employer. Employers are often introduced by a label such as "Company:"
      or "company -"; include every one of them. The title is the job title
      held there and the organization is the employer's name. Give start and
      end dates only when the resume
      states them for that job; otherwise use null. Ignore the education
      section: its dates and schools never belong to a job. The description
      is what the person did there, in the resume's own words, two sentences
      at most.
    - projects: every named project, whether academic, personal, or done at
      work, with its name and a description of two sentences at most. A
      project is not a job, so do not list it under experience.
    """


@generative
def extract_job(job_text: str) -> Experience:
    """Describe one job from the text about it.

    The text is the part of a resume about a single job. Copy from it. Never
    guess or add anything that is not written there, and use null for a value
    the text does not state.

    - title: the job title or designation the person held, if the text
      states one.
    - organization: the employer's name, if the text states it.
    - start and end: the dates of this job, if the text states them.
    - description: what the person did in this job, in the text's own words,
      two sentences at most.
    """


@generative
def list_employers(resume_text: str) -> _EmployerList:
    """List every employer in the resume text.

    employers: the name of every company, organization, or institution the
    person has worked for or interned at, spelled as in the resume. Include
    every one, however briefly it is mentioned. Do not include schools the
    person only studied at, clients, products, or project names. Use an empty
    list if the resume names no employer.
    """


@generative
def extract_overview(resume_text: str) -> _OverviewSection:
    """Pick out a few facts the resume states about the person.

    Copy from the resume. Never guess or work anything out: use null, or an
    empty list, unless the resume states it outright.

    - work_authorization: citizenship, visa status, or security clearance,
      only if the resume states it.
    - languages: the languages the person speaks, such as English or
      Spanish. Programming languages are not spoken languages.
    - achievements: awards, honors, scholarships, and publications the
      resume names, each as a short phrase copied from the resume.
    """


@generative
def extract_education(resume_text: str) -> _EducationSection:
    """List the education and the certifications in the resume text.

    Copy from the resume. Never guess or add anything that is not written
    there. Use null for a value the resume does not state, and an empty list
    when there are no entries.

    - education: every degree, diploma, or course of study, one entry per
      line of the education section. The degree is its name as written, such
      as "Bachelor of Science", "Master of Arts", or "Diploma". The field of
      study is the subject. The institution is the school, college,
      university, or board. The year is the year or dates on that line, as
      written. The grade is the GPA, percentage, or class, if the resume
      states one. Jobs and employers are not education.
    - certifications: every certification, license, or completed training
      course the resume names.
    """


def extract_resume(text: str, m: MelleaSession | None = None) -> ResumeProfile:
    """Extract a ResumeProfile from resume text.

    The result is checked against `text` before it is returned, so nothing in
    it is invented: see `ground_profile` for the rules.

    Pass an existing session as `m` to reuse it across calls; otherwise a new
    default session (Ollama + Granite) is started.
    """
    if not text or not text.strip():
        raise ValueError("resume text is empty")
    session = m if m is not None else start_session()
    labeled = labeled_jobs(text)
    try:
        skills = extract_skills(session, resume_text=text, model_options=MODEL_OPTIONS)
        work = extract_work_history(session, resume_text=text, model_options=MODEL_OPTIONS)
        education = extract_education(session, resume_text=text, model_options=MODEL_OPTIONS)
        overview = extract_overview(session, resume_text=text, model_options=MODEL_OPTIONS)
        if labeled:
            jobs = _jobs_from_labels(session, labeled, work.experience)
        else:
            named = list_employers(session, resume_text=text, model_options=MODEL_OPTIONS)
            jobs = _add_missed_employers(work.experience, named.employers)
    except ValidationError as e:
        # Usually the answer was cut off at the token cap, leaving incomplete JSON.
        raise ExtractionError(f"model returned an unreadable answer: {e.errors()[0]['msg']}") from e

    # Durations written as "Skill - Experience - 24 months" are read straight from the
    # text; the model's own list only adds skills that this did not find.
    durations = [SkillExperience(skill=skill, duration=duration) for skill, duration in stated_skill_durations(text)]
    found = {normalize(item.skill) for item in durations}
    durations += [item for item in skills.skill_experience if item.skill and normalize(item.skill) not in found]

    profile = ResumeProfile(
        # A skill the resume gives a duration for is a skill, even if the model's list missed it.
        skills=[*skills.skills, *(item.skill for item in durations if item.skill)],
        skill_experience=durations,
        experience=jobs,
        projects=work.projects,
        education=education.education,
        certifications=education.certifications,
        achievements=overview.achievements,
        languages=overview.languages,
        # Read straight from the text; the model mistook single-skill durations for totals.
        total_experience=stated_total_experience(text),
        work_authorization=overview.work_authorization,
    )
    return ground_profile(profile, text)


def _same_employer(a: str | None, b: str | None) -> bool:
    """True if two employer names are the same, nearly the same, or one contains the other.

    "Acme" matches "Acme Corp, Pune", and "Wab It Software" matches "Wab It Softwere".
    """
    if not a or not b:
        return False
    first, second = normalize(a), normalize(b)
    return first in second or second in first or SequenceMatcher(None, first, second).ratio() >= 0.85


def _is_heading(title: str) -> bool:
    """True for a section heading the model mistook for a job title."""
    return normalize(title).strip() in _HEADINGS


def _jobs_from_labels(session: MelleaSession, labeled: list[LabeledJob], model_jobs: list[Experience]) -> list[Experience]:
    """Build the job list for a resume that labels its employers.

    Every labelled employer becomes a job, so none can be missed. The model reads
    each job's own text for the title, dates, and description, and those are
    checked against that text alone, so a date from another part of the resume
    cannot end up on the job.
    """
    jobs: list[Experience] = []
    for item in labeled:
        detail = extract_job(session, job_text=item.body, model_options=MODEL_OPTIONS) if item.body else Experience()
        body = item.body

        def stated(value: str | None, minimum: float = 1.0) -> str | None:
            return value if value and word_overlap(value, body) >= minimum else None

        title = _without_employer(detail.title, item.organization)
        if title and not is_grounded(title, body):
            title = closest_span(title, body)
        if title and (_same_employer(title, item.organization) or len(title.split()) > MAX_TITLE_WORDS or _is_heading(title)):
            title = None
        if title is None:
            # The title is often written elsewhere ("Analyst - Acme" at the top of the
            # resume); the whole-resume pass may have found it for this employer.
            title = next((job.title for job in model_jobs if job.title and _same_employer(job.organization, item.organization)), None)
        jobs.append(
            Experience(
                title=title,
                organization=item.organization,
                start=stated(detail.start),
                end=stated(detail.end),
                description=stated(detail.description, MIN_DESCRIPTION_OVERLAP),
            )
        )

    # Employers the model found that have no label of their own (an internship
    # mentioned in a paragraph, for example) are kept as well.
    for job in model_jobs:
        if job.organization and not any(_same_employer(job.organization, known.organization) for known in jobs):
            jobs.append(job)
    return jobs


def _add_missed_employers(jobs: list[Experience], employers: list[str]) -> list[Experience]:
    """Add a job for each employer the second pass named that the first pass left out."""
    jobs = list(jobs)
    for employer in employers:
        if not any(_same_employer(employer, job.organization) for job in jobs):
            jobs.append(Experience(organization=employer))
    return jobs


def _without_employer(title: str | None, employer: str | None) -> str | None:
    """ "Project Manager - Acme" with employer "Acme" is just "Project Manager"."""
    if not title or not employer or normalize(title) == normalize(employer):
        return title
    if not normalize(title).endswith(normalize(employer)):
        return title
    cut = title.casefold().rfind(employer.casefold())
    return title[:cut].rstrip(" -–—,@|") if cut > 0 else title


def _job_title(job: Experience) -> str | None:
    """The job's title, unless it is really the employer's name, a heading, or a sentence."""
    title = _without_employer(job.title, job.organization)
    if not title or _is_heading(title) or len(title.split()) > MAX_TITLE_WORDS:
        return None
    if job.organization and normalize(title) == normalize(job.organization):
        return None
    return title


def _merge_repeated_jobs(jobs: list[dict]) -> list[dict]:
    """Combine entries for the same job, keeping every value either one has.

    Two entries are the same job when the employer and title match and their start
    dates do not disagree. Two stints at one employer with different dates stay apart.
    """

    def same_job(a: dict, b: dict) -> bool:
        if normalize(a["organization"] or "") != normalize(b["organization"] or ""):
            return False
        if normalize(a["title"] or "") != normalize(b["title"] or ""):
            return False
        return not (a["start"] and b["start"] and normalize(a["start"]) != normalize(b["start"]))

    merged: list[dict] = []
    for job in jobs:
        kept = next((other for other in merged if same_job(other, job)), None)
        if kept is None:
            merged.append(job)
        else:
            for field, value in job.items():
                kept[field] = kept[field] or value
    return merged


def ground_profile(profile: ResumeProfile, text: str) -> ResumeProfile:
    """Return a copy of `profile` with everything that is not in `text` removed.

    - Skills and certifications must appear in the resume. A skill with a category
      label in front ("Database MySQL") is trimmed to the part that appears, and a
      "skill" that is really a sentence is dropped.
    - Names (job title, employer, degree, field, school, project name) must appear
      word for word. If the model only changed a spelling, the resume's wording is
      used; otherwise the field is cleared.
    - Dates and durations may be laid out differently, but every word in them must
      come from the resume; otherwise the field is cleared.
    - A description or achievement is cleared when too few of its words come from
      the resume.

    - A date the resume states only once is not allowed on both a school and a job;
      the job loses it.
    - A value must suit its field: a spoken language must be a known one, an
      achievement must mention an award, rank, or publication, a grade must look
      like one, and work authorization must mention citizenship, a visa, or a
      clearance.

    An entry left with nothing to identify it is dropped, and a repeated job or
    education entry is kept once. Everything removed is logged as a warning.
    """
    removed: list[str] = []

    def exact(value: str | None, label: str) -> str | None:
        if value is None or is_grounded(value, text):
            return value
        # Not in the resume as written. If the resume has nearly the same words
        # (the model changed a spelling), use the resume's wording instead.
        as_written = closest_span(value, text)
        if as_written is None:
            removed.append(f"{label} {value!r}")
        return as_written

    def loose(value: str | None, label: str, minimum: float = 1.0) -> str | None:
        if value is None or word_overlap(value, text) >= minimum:
            return value
        removed.append(f"{label} {value!r}")
        return None

    def belongs(value: str | None, pattern: re.Pattern[str], label: str) -> str | None:
        """Keep a value only if it looks like what the field is for."""
        if value is None or pattern.search(value):
            return value
        removed.append(f"{label} {value!r} (not a {label})")
        return None

    # Durations read straight from the resume are already known to be in it.
    stated = {(normalize(skill), normalize(duration)) for skill, duration in stated_skill_durations(text)}

    def duration_of(item: SkillExperience) -> str | None:
        if item.skill and item.duration and (normalize(item.skill), normalize(item.duration)) in stated:
            return item.duration
        return loose(item.duration, "duration")

    skills = []
    for skill in profile.skills:
        found = ground(skill, text)
        if found is None or len(found.split()) > MAX_SKILL_WORDS:
            removed.append(f"skill {skill!r}")
        else:
            skills.append(found)

    data = {
        "skills": skills,
        "skill_experience": [
            {"skill": exact(item.skill, "skill"), "duration": duration_of(item)}
            for item in profile.skill_experience
        ],
        "experience": [
            {
                "title": exact(_job_title(job), "job title"),
                "organization": exact(job.organization, "employer"),
                "start": loose(job.start, "start date"),
                "end": loose(job.end, "end date"),
                "description": loose(job.description, "description", MIN_DESCRIPTION_OVERLAP),
            }
            for job in profile.experience
        ],
        "projects": [
            {
                "name": exact(project.name, "project"),
                "description": loose(project.description, "description", MIN_DESCRIPTION_OVERLAP),
            }
            for project in profile.projects
        ],
        "education": [
            {
                "degree": exact(entry.degree, "degree"),
                "field_of_study": exact(entry.field_of_study, "field of study"),
                "institution": exact(entry.institution, "school"),
                "year": loose(entry.year, "year"),
                "grade": belongs(loose(entry.grade, "grade"), _GRADE, "grade"),
            }
            for entry in profile.education
        ],
        "certifications": [c for c in (exact(cert, "certification") for cert in profile.certifications) if c],
        "achievements": [
            item
            for item in (
                belongs(loose(item, "achievement", MIN_DESCRIPTION_OVERLAP), _ACHIEVEMENT, "achievement")
                for item in profile.achievements
            )
            if item
        ],
        "languages": [
            name
            for name in (exact(item, "language") for item in profile.languages)
            if name and (normalize(name).strip() in SPOKEN_LANGUAGES or removed.append(f"language {name!r}"))
        ],
        "total_experience": loose(profile.total_experience, "total experience"),
        "work_authorization": belongs(
            loose(profile.work_authorization, "work authorization"), _WORK_AUTHORIZATION, "work authorization"
        ),
    }
    # A spoken language belongs under languages, not skills.
    spoken = {normalize(name) for name in data["languages"]}
    data["skills"] = [skill for skill in data["skills"] if normalize(skill) not in spoken]

    # A date the resume states once cannot belong to both a school and a job. The
    # model sometimes copies an education date onto a job, so the job loses it.
    education_dates = normalize(" ".join(entry["year"] or "" for entry in data["education"]))
    for job in data["experience"]:
        for key in ("start", "end"):
            date = job[key]
            if date and normalize(date) in education_dates and normalize(text).count(normalize(date)) < 2:
                removed.append(f"{key} date {date!r} (belongs to education)")
                job[key] = None

    data["experience"] = _merge_repeated_jobs(data["experience"])

    for entry in data["education"]:
        # A school name is not a degree, and neither one is a field of study.
        if entry["degree"] and entry["institution"] and normalize(entry["degree"]) == normalize(entry["institution"]):
            entry["degree"] = None
        for other in ("degree", "institution"):
            if entry["field_of_study"] and entry[other] and normalize(entry["field_of_study"]) == normalize(entry[other]):
                entry["field_of_study"] = None

    # The same degree listed twice (the model sometimes repeats a line) is kept once,
    # with every value either copy has. Entries at different schools stay apart.
    def same_entry(a: dict, b: dict) -> bool:
        if any(normalize(a[name] or "") != normalize(b[name] or "") for name in ("degree", "field_of_study", "year")):
            return False
        return not (a["institution"] and b["institution"]) or _same_employer(a["institution"], b["institution"])

    unique_education: list[dict] = []
    for entry in data["education"]:
        kept = next((other for other in unique_education if same_entry(other, entry)), None)
        if kept is None:
            unique_education.append(entry)
        else:
            for name, value in entry.items():
                kept[name] = kept[name] or value
    data["education"] = unique_education

    if removed:
        logger.warning("Removed %d value(s) the resume does not support: %s", len(removed), "; ".join(removed))
    # Validating again drops entries that were left empty and removes duplicates.
    return ResumeProfile.model_validate(data)


def _read_resume(args: argparse.Namespace) -> str:
    if args.file is not None:
        return args.file.read_text(encoding="utf-8")
    # Group 9's loader; importable because this module is run from the project root.
    from scripts.load_resumes import load_resumes

    resumes = load_resumes()
    if args.index >= len(resumes):
        raise IndexError(f"--index {args.index} out of range (0-{len(resumes) - 1})")
    return resumes[args.index]["resume"]


def main(argv: list[str] | None = None) -> int:
    """Extract one resume and print it as JSON. Exit code 0 = fully populated, 1 = not."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="Extract a ResumeProfile from one resume.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, default=0, metavar="I", help="resume #I in data/resumes.json (default 0)")
    source.add_argument("--file", type=Path, metavar="PATH", help="a text file holding one resume")
    args = parser.parse_args(argv)

    try:
        text = _read_resume(args)
    except (OSError, IndexError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    try:
        profile = extract_resume(text)
    except ExtractionError as e:
        print(e, file=sys.stderr)
        return 1
    print(profile.model_dump_json(indent=2))

    missing = profile.missing_sections()
    if missing:
        print(f"Not fully populated; empty sections: {', '.join(missing)}", file=sys.stderr)
        return 1
    print(
        f"Fully populated: {len(profile.skills)} skills, {len(profile.experience)} jobs, "
        f"{len(profile.education)} education entries, {len(profile.projects)} projects, "
        f"{len(profile.certifications)} certifications",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
