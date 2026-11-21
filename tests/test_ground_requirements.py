"""Tests for the checks applied to JobRequirements after the model returns them. No model calls."""

from __future__ import annotations

import logging
import unittest

from analyzer.extract_posting import ground_requirements
from analyzer.schemas import JobRequirements

POSTING = """Qualifications:
Attention to Detail - Is thorough when performing work.
Specialized experience is defined as: diagnosing and resolving complex hardware, software, and network incidents;
installing and configuring IT equipment. Software: ServiceNow, Active Directory.
Knowledge of Python is desired. Must hold a CompTIA Security+ certification. Spanish speakers are preferred.

Education:
There is no educational substitution at this grade level. A bachelor's degree in computer science may substitute.

Requirements:
You must be a U.S. Citizen to apply for this job. You need one year of specialized experience.
"""


class _quiet:
    """Silence the removal warnings while a test runs."""

    def __enter__(self) -> None:
        logging.disable(logging.WARNING)

    def __exit__(self, *exc: object) -> None:
        logging.disable(logging.NOTSET)


def check(requirements: JobRequirements) -> tuple[JobRequirements, list[str]]:
    """Ground `requirements` against POSTING and return the result and what was removed."""
    removed: list[str] = []
    with _quiet():
        result = ground_requirements(requirements, POSTING, removed)
    return result, removed


class SkillTests(unittest.TestCase):
    def test_keeps_skills_the_posting_names(self) -> None:
        result, removed = check(JobRequirements(required_skills=["ServiceNow", "Active Directory", "Attention to Detail"]))
        self.assertEqual(result.required_skills, ["ServiceNow", "Active Directory", "Attention to Detail"])
        self.assertEqual(removed, [])

    def test_removes_an_invented_skill_and_reports_it(self) -> None:
        result, removed = check(JobRequirements(required_skills=["ServiceNow", "Kubernetes"]))
        self.assertEqual(result.required_skills, ["ServiceNow"])
        self.assertEqual(removed, ["required skill 'Kubernetes'"])

    def test_trims_a_label_in_front_of_a_skill(self) -> None:
        result, _ = check(JobRequirements(required_skills=["Ticketing ServiceNow"]))
        self.assertEqual(result.required_skills, ["ServiceNow"])

    def test_whole_words_only(self) -> None:
        # "Java" is not in the posting, even though "Python" and other words are.
        result, _ = check(JobRequirements(preferred_skills=["Java", "Python"]))
        self.assertEqual(result.preferred_skills, ["Python"])

    def test_drops_a_sentence_offered_as_a_skill(self) -> None:
        long_skill = "diagnosing and resolving complex hardware, software, and network incidents"
        result, removed = check(JobRequirements(required_skills=[long_skill]))
        self.assertEqual(result.required_skills, [])
        self.assertEqual(removed, [f"required skill {long_skill!r} (a sentence, not a skill)"])

    def test_a_spoken_language_is_not_a_skill(self) -> None:
        result, _ = check(JobRequirements(required_skills=["Spanish", "ServiceNow"], languages=["Spanish"]))
        self.assertEqual(result.required_skills, ["ServiceNow"])
        self.assertEqual(result.languages, ["Spanish"])


class QualificationTests(unittest.TestCase):
    def test_keeps_a_qualification_in_the_postings_words(self) -> None:
        item = "diagnosing and resolving complex hardware, software, and network incidents"
        result, _ = check(JobRequirements(qualifications=[item]))
        self.assertEqual(result.qualifications, [item])

    def test_removes_an_invented_qualification(self) -> None:
        result, removed = check(JobRequirements(qualifications=["five years leading cloud migration teams"]))
        self.assertEqual(result.qualifications, [])
        self.assertEqual(len(removed), 1)

    def test_removes_the_no_substitution_boilerplate_from_education(self) -> None:
        result, removed = check(
            JobRequirements(education=["There is no educational substitution at this grade level"])
        )
        self.assertEqual(result.education, [])
        self.assertIn("not a requirement", removed[0])

    def test_keeps_a_real_degree_requirement(self) -> None:
        result, _ = check(JobRequirements(education=["bachelor's degree in computer science"]))
        self.assertEqual(result.education, ["bachelor's degree in computer science"])

    def test_restores_the_postings_spelling_of_a_certification(self) -> None:
        result, _ = check(JobRequirements(certifications=["CompTIA Securty+"]))
        self.assertEqual(result.certifications, ["CompTIA Security+"])

    def test_removes_an_invented_certification(self) -> None:
        result, _ = check(JobRequirements(certifications=["CISSP"]))
        self.assertEqual(result.certifications, [])

    def test_a_programming_language_is_not_a_spoken_language(self) -> None:
        result, removed = check(JobRequirements(languages=["Python"]))
        self.assertEqual(result.languages, [])
        self.assertIn("not a spoken language", removed[0])


class StatedFactTests(unittest.TestCase):
    def test_keeps_stated_experience_and_citizenship(self) -> None:
        result, _ = check(JobRequirements(min_experience="one year", work_authorization="U.S. Citizen"))
        self.assertEqual((result.min_experience, result.work_authorization), ("one year", "U.S. Citizen"))

    def test_removes_experience_the_posting_does_not_state(self) -> None:
        result, _ = check(JobRequirements(min_experience="5 years"))
        self.assertIsNone(result.min_experience)

    def test_work_authorization_must_be_about_citizenship_visa_or_clearance(self) -> None:
        # The words are in the posting, but they are not a work authorization.
        result, removed = check(JobRequirements(work_authorization="computer science"))
        self.assertIsNone(result.work_authorization)
        self.assertIn("not a work authorization", removed[0])


class ReportingTests(unittest.TestCase):
    def test_nothing_removed_means_an_empty_report(self) -> None:
        _, removed = check(JobRequirements(required_skills=["ServiceNow"]))
        self.assertEqual(removed, [])

    def test_removals_are_logged_as_a_warning(self) -> None:
        with self.assertLogs("analyzer.extract_posting", level=logging.WARNING) as logs:
            ground_requirements(JobRequirements(required_skills=["Kubernetes"]), POSTING)
        self.assertIn("Kubernetes", logs.output[0])

    def test_the_removed_argument_is_optional(self) -> None:
        with self.assertLogs("analyzer.extract_posting", level=logging.WARNING):
            result = ground_requirements(JobRequirements(required_skills=["Kubernetes", "ServiceNow"]), POSTING)
        self.assertEqual(result.required_skills, ["ServiceNow"])


if __name__ == "__main__":
    unittest.main()
