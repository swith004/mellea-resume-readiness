"""End-to-end tests for the resume extractor. They call the model, so they are opt-in:

    RUN_LLM_TESTS=1 uv run python -m unittest tests.test_extract_resume -v

Needs Ollama running with granite4.2:3b pulled.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from analyzer.extract_resume import extract_resume
from analyzer.grounding import is_grounded, ungrounded

SAMPLE = Path(__file__).parent / "fixtures" / "sample_resume.txt"
RUN_LLM_TESTS = os.environ.get("RUN_LLM_TESTS") == "1"


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to call the model")
class SampleResumeTests(unittest.TestCase):
    """The hand-written sample has every section, so every section should come back."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SAMPLE.read_text(encoding="utf-8")
        cls.profile = extract_resume(cls.text)

    def test_core_sections_are_fully_populated(self) -> None:
        self.assertEqual(self.profile.missing_sections(), [])

    def test_finds_every_listed_skill(self) -> None:
        found = {skill.casefold() for skill in self.profile.skills}
        for skill in ("python", "javascript", "sql", "react", "flask", "git", "docker", "postgresql"):
            self.assertIn(skill, found)

    def test_every_skill_appears_in_the_resume(self) -> None:
        self.assertEqual(ungrounded(self.profile.skills, self.text), [])

    def test_finds_stated_skill_durations(self) -> None:
        durations = {item.skill: item.duration for item in self.profile.skill_experience}
        self.assertEqual(durations.get("Python"), "36 months")
        self.assertEqual(durations.get("SQL"), "24 months")

    def test_finds_both_jobs_with_their_own_dates(self) -> None:
        jobs = {job.organization: job for job in self.profile.experience}
        self.assertEqual(set(jobs), {"Sunshine Health Systems", "Florida International University"})
        intern = jobs["Sunshine Health Systems"]
        self.assertEqual((intern.title, intern.start, intern.end), ("Software Engineering Intern", "May 2025", "August 2025"))

    def test_finds_the_project_separately_from_the_jobs(self) -> None:
        self.assertEqual([project.name for project in self.profile.projects], ["Campus Shuttle Tracker"])

    def test_finds_both_degrees(self) -> None:
        schools = {entry.institution: entry for entry in self.profile.education}
        self.assertEqual(set(schools), {"Florida International University", "Miami Dade College"})
        self.assertIn("Associate in Arts", schools["Miami Dade College"].degree or "")

    def test_finds_both_certifications(self) -> None:
        self.assertEqual(set(self.profile.certifications), {"CompTIA Security+", "AWS Certified Cloud Practitioner"})

    def test_finds_the_stated_facts(self) -> None:
        self.assertEqual(self.profile.total_experience, "2 years")
        self.assertIn("Citizen", self.profile.work_authorization or "")
        self.assertEqual(set(self.profile.languages), {"English", "Spanish"})

    def test_spoken_languages_are_not_listed_as_skills(self) -> None:
        skills = {skill.casefold() for skill in self.profile.skills}
        self.assertFalse({"english", "spanish"} & skills)

    def test_finds_the_awards(self) -> None:
        awards = " | ".join(self.profile.achievements)
        self.assertIn("Dean's List", awards)
        self.assertIn("ShellHacks", awards)

    def test_finds_the_grade(self) -> None:
        grades = [entry.grade for entry in self.profile.education if entry.grade]
        self.assertTrue(any("3.7" in grade for grade in grades), grades)


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to call the model")
class DatasetResumeTests(unittest.TestCase):
    """Resume #1 in data/resumes.json: short, real, and it used to get education dates on its job."""

    @classmethod
    def setUpClass(cls) -> None:
        from scripts.load_resumes import load_resumes

        cls.text = load_resumes()[1]["resume"]
        cls.profile = extract_resume(cls.text)

    def test_core_sections_are_fully_populated(self) -> None:
        self.assertEqual(self.profile.missing_sections(), [])

    def test_finds_the_employer(self) -> None:
        self.assertIn("Matelabs", [job.organization for job in self.profile.experience])

    def test_the_job_does_not_take_the_degree_dates(self) -> None:
        for job in self.profile.experience:
            self.assertNotIn("2013", f"{job.start} {job.end}")

    def test_finds_all_seven_stated_skill_durations(self) -> None:
        self.assertEqual(len(self.profile.skill_experience), 7)

    def test_every_name_appears_in_the_resume(self) -> None:
        names = [*self.profile.skills, *self.profile.certifications]
        names += [value for job in self.profile.experience for value in (job.title, job.organization) if value]
        names += [value for entry in self.profile.education for value in (entry.degree, entry.institution) if value]
        self.assertEqual([name for name in names if not is_grounded(name, self.text)], [])


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to call the model")
class EmployerRecallTests(unittest.TestCase):
    """Real resumes where the model alone used to miss employers."""

    @classmethod
    def setUpClass(cls) -> None:
        from mellea import start_session

        from scripts.load_resumes import load_resumes

        cls.resumes = load_resumes()
        cls.session = start_session()

    def employers(self, index: int) -> list[str]:
        profile = extract_resume(self.resumes[index]["resume"], self.session)
        return [job.organization for job in profile.experience if job.organization]

    def test_finds_an_employer_whose_description_is_blank(self) -> None:
        self.assertIn("Mumbai Monorail", self.employers(42))

    def test_keeps_the_resumes_spelling_and_lists_the_employer_once(self) -> None:
        self.assertEqual(self.employers(319), ["Wab It Softwere Pvt. Ltd."])

    def test_finds_all_five_employers_on_the_longest_resume(self) -> None:
        found = self.employers(514)
        for employer in (
            "Delta Controls, Dubai FZCO",
            "HONEYWELL AUTOMATION INDIA LTD",
            "FOX CONTROLS",
            "Fox Controls, Dubai",
            "ENERCON (INDIA) LIMITED",
        ):
            self.assertIn(employer, found)


class ExtractResumeInputTests(unittest.TestCase):
    def test_empty_text_is_rejected_before_calling_the_model(self) -> None:
        with self.assertRaises(ValueError):
            extract_resume("   \n ")


if __name__ == "__main__":
    unittest.main()
