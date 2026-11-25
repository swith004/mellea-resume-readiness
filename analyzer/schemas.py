"""Typed shapes the analyzer passes between its steps.

Only data classes live here: field names, types, and light cleanup. No LLM
calls, so other groups can import these and build test objects by hand.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# What the model writes when a resume does not state something. Treated as "not stated".
_PLACEHOLDERS = {"n/a", "na", "none", "null", "unknown", "not specified", "not stated", "not provided", "not mentioned", "-"}


# Separates the two ends of a date range: "May 2021 to June 2022", "2019 - 2021", "2019 – Present".
_DATE_RANGE = re.compile(r"\s+(?:to|-|–|—)\s+", re.IGNORECASE)


def _clean(value: str | None) -> str | None:
    """Collapse whitespace; turn an empty string or a placeholder like "N/A" into None."""
    if value is None:
        return None
    value = " ".join(value.split())
    if not value or value.casefold() in _PLACEHOLDERS:
        return None
    return value


def _require_every_field(schema: dict) -> None:
    schema["required"] = list(schema.get("properties", {}))


class Schema(BaseModel):
    """Base class for the analyzer's data classes.

    The JSON schema given to the model lists every field as required. Without
    that, the model skips any field that has a default and whole sections come
    back empty. The Python defaults still apply, so objects built by hand can
    leave fields out.
    """

    model_config = ConfigDict(json_schema_extra=_require_every_field)


class Experience(Schema):
    """One job or internship listed on a resume. Projects go in `Project`."""

    title: str | None = Field(default=None, description="Job title or role, as written on the resume")
    organization: str | None = Field(default=None, description="Employer or organization name")
    start: str | None = Field(default=None, description="Start date of this job as written, e.g. 'May 2021'")
    end: str | None = Field(default=None, description="End date of this job as written, or 'Present'")
    description: str | None = Field(default=None, description="What the person did in this role, in the resume's words")

    @field_validator("title", "organization", "start", "end", "description", mode="before")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return _clean(value)

    @model_validator(mode="after")
    def _split_date_range(self) -> Experience:
        """Turn a start of "June 2015 to February 2018" with no end into separate start and end."""
        if self.start and not self.end:
            parts = _DATE_RANGE.split(self.start, maxsplit=1)
            if len(parts) == 2 and all(part.strip() for part in parts):
                self.start, self.end = parts[0].strip(), parts[1].strip()
        return self


class Education(Schema):
    """One degree, diploma, or course of study listed on a resume."""

    degree: str | None = Field(default=None, description="Degree or diploma, e.g. 'B.S.' or 'Bachelor of Science'")
    field_of_study: str | None = Field(default=None, description="Field of study or major")
    institution: str | None = Field(default=None, description="School, college, or university name")
    year: str | None = Field(default=None, description="Graduation year or date range as written")
    grade: str | None = Field(default=None, description="GPA, percentage, or class as written, e.g. '3.6 GPA' or 'First Class'")

    @field_validator("degree", "field_of_study", "institution", "year", "grade", mode="before")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return _clean(value)


class Project(Schema):
    """One named project (academic, personal, or done at work) listed on a resume."""

    name: str | None = Field(default=None, description="Project name or title, as written on the resume")
    description: str | None = Field(default=None, description="What the project was, in the resume's words")

    @field_validator("name", "description", mode="before")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return _clean(value)


class SkillExperience(Schema):
    """How long the person has used a skill, when the resume says so."""

    skill: str | None = Field(default=None, description="Skill name, as written on the resume")
    duration: str | None = Field(default=None, description="How long, as written, e.g. '24 months' or '3 years'")

    @field_validator("skill", "duration", mode="before")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return _clean(value)


def _list_key(value: str) -> str:
    """What two entries are compared on: letters and digits only, so "Node.js" equals "node js"."""
    return " ".join(re.findall(r"[a-z0-9+#]+", value.casefold()))


def _clean_list(values: list[str] | None) -> list[str]:
    """Tidy a list of names: trim each entry, drop blanks, placeholders, and repeats.

    Punctuation left hanging on the end of an entry ("SQL," or "Routing and Switching.")
    is removed. Two entries are repeats when they differ only in case, spacing, or
    punctuation; the first spelling is kept.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in values or []:
        value = _clean(raw)
        if value:
            # A leading "." is kept for names like ".NET"; "+" and "#" for "C++" and "C#".
            value = _clean(value.rstrip(".,;:").lstrip(",;:"))
        key = _list_key(value) if value else ""
        if value and key and key not in seen:
            seen.add(key)
            cleaned.append(value)
    return cleaned


class ResumeProfile(Schema):
    """Structured view of a resume: what the person knows, has done, and has studied.

    `skills`, `experience`, and `education` are the core sections every resume is
    expected to have. Everything else is filled only when the resume has it.

    Left out on purpose: name, contact details, date of birth, gender, and marital
    status. A skills-gap check does not need them, and they should not influence it.
    """

    # maxItems is a limit for the model only (it is not enforced on objects built
    # by hand). Without it the model can get stuck repeating list entries until it
    # runs out of tokens, which leaves the JSON cut off and unreadable.
    skills: list[str] = Field(
        default_factory=list,
        description="Skills, tools, and technologies named on the resume, one per item",
        json_schema_extra={"maxItems": 60},
    )
    skill_experience: list[SkillExperience] = Field(
        default_factory=list,
        description="How long each skill has been used, only where the resume states it",
        json_schema_extra={"maxItems": 40},
    )
    experience: list[Experience] = Field(
        default_factory=list,
        description="Jobs and internships, one entry per employer",
        json_schema_extra={"maxItems": 12},
    )
    projects: list[Project] = Field(
        default_factory=list,
        description="Named projects: academic, personal, or done at work",
        json_schema_extra={"maxItems": 12},
    )
    education: list[Education] = Field(
        default_factory=list,
        description="Degrees, diplomas, and courses of study",
        json_schema_extra={"maxItems": 8},
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Certifications, licenses, and completed training courses",
        json_schema_extra={"maxItems": 15},
    )
    achievements: list[str] = Field(
        default_factory=list,
        description="Awards, honors, scholarships, and publications",
        json_schema_extra={"maxItems": 10},
    )
    languages: list[str] = Field(
        default_factory=list,
        description="Spoken languages, such as English or Spanish (not programming languages)",
        json_schema_extra={"maxItems": 10},
    )
    total_experience: str | None = Field(
        default=None,
        description="Total work experience when the resume states it, as written, e.g. '5 years'",
    )
    work_authorization: str | None = Field(
        default=None,
        description="Citizenship, visa status, or security clearance when the resume states it, as written",
    )

    @field_validator("total_experience", "work_authorization", mode="before")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        return _clean(value)

    @field_validator("total_experience", mode="after")
    @classmethod
    def _keep_only_the_duration(cls, value: str | None) -> str | None:
        """ "5 years of experience" is recorded as "5 years"."""
        if value is None:
            return None
        return _clean(re.sub(r"\s+(?:of\s+(?:\w+\s+)?)?experience\b.*$", "", value, flags=re.IGNORECASE))

    @field_validator("skills", "certifications", "achievements", "languages", mode="before")
    @classmethod
    def _clean_string_lists(cls, value: list[str] | None) -> list[str]:
        return _clean_list(value)

    @field_validator("experience", mode="after")
    @classmethod
    def _drop_empty_experience(cls, value: list[Experience]) -> list[Experience]:
        """A job needs at least a title or an employer."""
        return [item for item in value if item.title or item.organization]

    @field_validator("education", mode="after")
    @classmethod
    def _drop_empty_education(cls, value: list[Education]) -> list[Education]:
        """An education entry needs at least a degree, a field, or a school."""
        return [item for item in value if item.degree or item.field_of_study or item.institution]

    @field_validator("projects", mode="after")
    @classmethod
    def _drop_unnamed_projects(cls, value: list[Project]) -> list[Project]:
        return [item for item in value if item.name]

    @field_validator("skill_experience", mode="after")
    @classmethod
    def _drop_incomplete_skill_experience(cls, value: list[SkillExperience]) -> list[SkillExperience]:
        return [item for item in value if item.skill and item.duration]

    def missing_sections(self) -> list[str]:
        """Names of the core sections that came back empty. An empty list means fully populated."""
        return [name for name in ("skills", "experience", "education") if not getattr(self, name)]

    @property
    def is_fully_populated(self) -> bool:
        return not self.missing_sections()
