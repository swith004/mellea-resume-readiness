"""Tests for reading fixed USAJOBS wording straight from a posting. No model calls."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from analyzer.posting_layout import (
    posting_text,
    section,
    stated_competencies,
    stated_min_experience,
    stated_work_authorization,
)

SAMPLE = Path(__file__).parent / "fixtures" / "sample_posting.json"


class PostingTextTests(unittest.TestCase):
    def test_joins_every_requirement_field_with_a_label(self) -> None:
        text = posting_text(json.loads(SAMPLE.read_text(encoding="utf-8")))
        for label in ("Summary:", "Duties:", "Qualifications:", "Education:", "Requirements:"):
            self.assertIn(label, text)

    def test_skips_missing_and_empty_fields_and_joins_lists(self) -> None:
        text = posting_text({"summary": "", "duties": ["Fix laptops.", None, "Train staff."], "education": None})
        self.assertEqual(text, "Duties:\nFix laptops.\nTrain staff.")

    def test_leaves_out_fields_that_are_not_requirements(self) -> None:
        text = posting_text({"title": "IT Specialist", "salary": {"min": "1"}, "qualifications": "Know SQL."})
        self.assertEqual(text, "Qualifications:\nKnow SQL.")

    def test_section_returns_one_labelled_part(self) -> None:
        text = posting_text({"duties": "Fix laptops.", "qualifications": "Know SQL.", "requirements": "Be a citizen."})
        self.assertEqual(section(text, "Qualifications").strip(), "Know SQL.")

    def test_section_of_unlabelled_text_is_the_whole_text(self) -> None:
        self.assertEqual(section("Know SQL.", "Qualifications"), "Know SQL.")


class WorkAuthorizationTests(unittest.TestCase):
    def test_reads_the_citizenship_sentence(self) -> None:
        self.assertEqual(stated_work_authorization("You must be a U.S. Citizen to apply for this job"), "U.S. Citizen")

    def test_keeps_the_or_national_part(self) -> None:
        self.assertEqual(
            stated_work_authorization("You must be a US Citizen or US National."), "US Citizen or US National"
        )

    def test_reads_who_may_apply_and_citizenship_required(self) -> None:
        self.assertEqual(stated_work_authorization("Who May Apply: US Citizens"), "US Citizens")
        self.assertEqual(stated_work_authorization("U.S. Citizenship is Required"), "U.S. Citizenship")

    def test_falls_back_to_a_clearance(self) -> None:
        self.assertEqual(
            stated_work_authorization("Must be able to obtain a Top Secret/SCI security clearance."),
            "Top Secret/SCI security clearance",
        )

    def test_none_when_not_stated(self) -> None:
        self.assertIsNone(stated_work_authorization("Applicants must pass a drug test."))


class MinExperienceTests(unittest.TestCase):
    def test_reads_specialized_experience(self) -> None:
        text = "You must have one year of specialized experience equivalent to at least the next lower grade GS-11"
        self.assertEqual(stated_min_experience(text), "one year")

    def test_keeps_the_amount_as_written(self) -> None:
        self.assertEqual(stated_min_experience("One (1) year of specialized experience at the GS-09"), "One (1) year")
        self.assertEqual(stated_min_experience("one (1) full year of specialized experience"), "one (1) full year")
        self.assertEqual(stated_min_experience("2 years of experience developing training"), "2 years")

    def test_ignores_a_definition(self) -> None:
        text = "One year of experience refers to full-time work. You need 3 years of professional experience."
        self.assertEqual(stated_min_experience(text), "3 years")

    def test_prefers_specialized_experience(self) -> None:
        text = "2 years of general experience, including one year of specialized experience."
        self.assertEqual(stated_min_experience(text), "one year")

    def test_none_when_not_stated(self) -> None:
        self.assertIsNone(stated_min_experience("Experience with Excel is helpful."))


class CompetencyTests(unittest.TestCase):
    def test_reads_the_competencies_from_the_sample_posting(self) -> None:
        text = posting_text(json.loads(SAMPLE.read_text(encoding="utf-8")))
        self.assertEqual(
            stated_competencies(text),
            ["Attention to Detail", "Customer Service", "Oral Communication", "Problem Solving"],
        )

    def test_reads_competencies_after_a_list_marker(self) -> None:
        text = "Qualifications:\nthe following: a) Attention to Detail - Is thorough; b) Customer Service - Works with clients"
        self.assertEqual(stated_competencies(text), ["Attention to Detail", "Customer Service"])

    def test_ignores_the_same_layout_outside_qualifications(self) -> None:
        text = posting_text(
            {
                "duties": "Accident Investigations - Conducts the investigation of failures.",
                "requirements": "Proof of citizenship. Selective Service Registration - Males born after 1959.",
                "qualifications": "Problem Solving - Identifies problems.",
            }
        )
        self.assertEqual(stated_competencies(text), ["Problem Solving"])

    def test_ignores_a_name_that_does_not_start_a_sentence(self) -> None:
        # "...Principles of the American Founding - This core..." is the tail of a longer name.
        text = "ECQ 1: Commitment to the Rule of Law and the Principles of the American Founding - This requires"
        self.assertEqual(stated_competencies(text), [])

    def test_lists_each_competency_once(self) -> None:
        text = "Customer Service - Works with clients. Customer Service - Works with clients."
        self.assertEqual(stated_competencies(text), ["Customer Service"])


if __name__ == "__main__":
    unittest.main()
