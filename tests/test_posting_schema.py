"""Tests for the JobRequirements data class. No model calls.

    uv run python -m unittest tests.test_posting_schema -v
"""

from __future__ import annotations

import unittest

from analyzer import JobRequirements


class JobRequirementsTests(unittest.TestCase):
    def test_hand_built_requirements_only_need_the_fields_you_care_about(self) -> None:
        requirements = JobRequirements(required_skills=["Python"], qualifications=["one year of IT experience"])
        self.assertEqual(requirements.preferred_skills, [])
        self.assertEqual(requirements.certifications, [])
        self.assertIsNone(requirements.min_experience)
        self.assertIsNone(requirements.work_authorization)
        self.assertTrue(requirements.is_fully_populated)

    def test_empty_requirements_report_both_core_sections_missing(self) -> None:
        requirements = JobRequirements()
        self.assertEqual(requirements.missing_sections(), ["required_skills", "qualifications"])
        self.assertFalse(requirements.is_fully_populated)

    def test_preferred_skills_are_optional(self) -> None:
        # Most USAJOBS postings mark nothing as preferred; that is not a failed extraction.
        requirements = JobRequirements(required_skills=["SQL"], qualifications=["one year of experience"])
        self.assertEqual(requirements.missing_sections(), [])

    def test_lists_are_trimmed_and_deduplicated_ignoring_case(self) -> None:
        requirements = JobRequirements(required_skills=["  Python ", "python", "", "N/A", "SQL"])
        self.assertEqual(requirements.required_skills, ["Python", "SQL"])

    def test_placeholders_become_none(self) -> None:
        requirements = JobRequirements(min_experience="Not specified", work_authorization=" n/a ")
        self.assertIsNone(requirements.min_experience)
        self.assertIsNone(requirements.work_authorization)

    def test_a_skill_listed_as_required_and_preferred_counts_once_as_required(self) -> None:
        requirements = JobRequirements(required_skills=["Python"], preferred_skills=["python", "Docker"])
        self.assertEqual(requirements.required_skills, ["Python"])
        self.assertEqual(requirements.preferred_skills, ["Docker"])

    def test_json_schema_marks_every_field_required_and_caps_lists(self) -> None:
        # The model skips fields that look optional and can loop on uncapped lists.
        schema = JobRequirements.model_json_schema()
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        for name in ("required_skills", "preferred_skills", "qualifications", "education", "certifications", "languages"):
            self.assertIn("maxItems", schema["properties"][name], name)

    def test_round_trips_through_json(self) -> None:
        requirements = JobRequirements(
            required_skills=["Customer Service"], qualifications=["one year of specialized experience"],
            min_experience="one year", work_authorization="U.S. Citizen",
        )
        self.assertEqual(JobRequirements.model_validate_json(requirements.model_dump_json()), requirements)


if __name__ == "__main__":
    unittest.main()
