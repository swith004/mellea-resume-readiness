from __future__ import annotations
from analyzer.schemas import ResumeProfile, JobRequirements
from analyzer.grounding import normalize 
from analyzer.extract_resume import MODEL_OPTIONS
from pydantic import BaseModel, Field
from mellea import MelleaSession, generative, start_session

class Match(BaseModel):
    score: float
    missing: list[str]

def same_skill(required: str , has: str) -> bool:
    a,b = normalize(required), normalize(has)
    if not a.strip() or not b.strip():
        return False
    return a==b or a in b or b in a

def _covered(required: str, resume_skills: list[str]) -> bool:
    return any(same_skill(required, s) for s in resume_skills)

class _Equals(BaseModel):
    required: str
    resume_skill: str | None = None

class _EqualsList(BaseModel):
    pairs: list[_Equals] = Field(default_factory=list)


@generative
def find_equal_skills(required_skills: list[str], resume_skills: list[str]) -> _EqualsList:
     """For each required skill, name the resume skill that means the same thing:
    an abbreviation, another name for it, or a more specific version of it.
    Use null when no resume skill means the same thing.
    Only use resume skills copied exactly from the list.
    """


def _llm_covered(missing: list[str], resume_skills: list[str], m:MelleaSession) -> set[str]:
    answer = find_equal_skills(m, required_skills=missing, resume_skills=resume_skills, model_options=MODEL_OPTIONS)
    real = {normalize(s) for s in resume_skills}
    asked = {normalize(s) for s in missing}
    covered = set()
    for pair in answer.pairs:
        if pair.resume_skill and normalize(pair.resume_skill) in real and normalize(pair.required) in asked:
            covered.add(normalize(pair.required))
    return covered


def match(resume: ResumeProfile, job: JobRequirements, use_llm: bool = False, m: MelleaSession | None = None) -> Match:
    required = job.required_skills
    missing = [s for s in required if not _covered(s, resume.skills)]
    if use_llm and missing and resume.skills:
        session = m if m is not None else start_session()
        covered = _llm_covered(missing, resume.skills, session)
        missing = [s for s in missing if normalize(s) not in covered]

    score = 100 * (len(required) - len(missing)) / len(required) if required else 100.0
    return Match(score=round(score, 1), missing=missing)

if __name__ == "__main__":
    import sys

    resume = ResumeProfile(skills=["Python programming", "SQL", "JS", "Node.js", "Git"])
    job = JobRequirements(required_skills=["Python", "JavaScript", "node js", "SQL databases", "Linux"])

    result = match(resume, job, use_llm="--llm" in sys.argv)
    print(f"Match score: {result.score}%")
    for item in result.missing:
        print(f"Missing: {item}")
