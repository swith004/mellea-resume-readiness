"""End-to-end tests for the posting extractor. They call the model, so they are opt-in:

    RUN_LLM_TESTS=1 uv run python -m unittest tests.test_extract_posting -v

Needs Ollama running with granite4.2:3b pulled.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from analyzer.extract_posting import extract_requirements, main
from analyzer.grounding import ungrounded
from analyzer.posting_layout import posting_text

SAMPLE = Path(__file__).parent / "fixtures" / "sample_posting.json"
CONSTRUCTION = Path(__file__).parent.parent / "data" / "postings_construction_miami-fl.json"
RUN_LLM_TESTS = os.environ.get("RUN_LLM_TESTS") == "1"


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to call the model")
class SamplePostingTests(unittest.TestCase):
    """The VA "IT Specialist" posting (#0 in the IT postings file)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.posting = json.loads(SAMPLE.read_text(encoding="utf-8"))
        cls.text = posting_text(cls.posting)
        cls.requirements = extract_requirements(cls.posting)

    def test_core_sections_are_fully_populated(self) -> None:
        self.assertEqual(self.requirements.missing_sections(), [])

    def test_finds_the_four_competencies(self) -> None:
        for skill in ("Attention to Detail", "Customer Service", "Oral Communication", "Problem Solving"):
            self.assertIn(skill, self.requirements.required_skills)

    def test_every_skill_appears_in_the_posting(self) -> None:
        skills = [*self.requirements.required_skills, *self.requirements.preferred_skills]
        self.assertEqual(ungrounded(skills, self.text), [])

    def test_finds_the_stated_facts(self) -> None:
        self.assertEqual(self.requirements.min_experience, "one year")
        self.assertEqual(self.requirements.work_authorization, "U.S. Citizen")

    def test_the_no_substitution_sentence_is_not_an_education_requirement(self) -> None:
        self.assertEqual(self.requirements.education, [])


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to call the model")
class ConstructionPostingTests(unittest.TestCase):
    """A posting from the other saved search, to check nothing is IT-specific."""

    @classmethod
    def setUpClass(cls) -> None:
        posting = json.loads(CONSTRUCTION.read_text(encoding="utf-8"))["postings"][3]
        cls.text = posting_text(posting)
        cls.requirements = extract_requirements(posting)

    def test_core_sections_are_fully_populated(self) -> None:
        self.assertEqual(self.requirements.missing_sections(), [])

    def test_every_value_appears_in_the_posting(self) -> None:
        names = [*self.requirements.required_skills, *self.requirements.certifications, *self.requirements.languages]
        self.assertEqual(ungrounded(names, self.text), [])


class InputTests(unittest.TestCase):
    """No model calls: bad input is rejected before a session is started."""

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_requirements(" \n ")

    def test_a_posting_with_no_requirement_fields_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_requirements({"title": "IT Specialist", "summary": "", "duties": []})

    def test_cli_rejects_an_index_out_of_range(self) -> None:
        self.assertEqual(main(["--index", "999"]), 1)


if __name__ == "__main__":
    unittest.main()
