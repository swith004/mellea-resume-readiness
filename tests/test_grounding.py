"""Tests for the grounding checks. No model calls."""

from __future__ import annotations

import unittest

from analyzer.grounding import (
    closest_span,
    ground,
    is_grounded,
    stated_skill_durations,
    stated_total_experience,
    ungrounded,
    word_overlap,
)

RESUME = "Skills: Naïve Bayes, JavaScript, C++ and Node.js. Extreme Programming(XP). Database: MySQL"


class GroundingTests(unittest.TestCase):
    def test_case_spacing_and_punctuation_are_ignored(self) -> None:
        for skill in ("javascript", "Node JS", "Extreme Programming (XP)", "c++"):
            self.assertTrue(is_grounded(skill, RESUME), skill)

    def test_accents_are_ignored(self) -> None:
        self.assertTrue(is_grounded("naive bayes", RESUME))

    def test_only_whole_words_count(self) -> None:
        # "Java" must not pass because of "JavaScript", nor "C" because of "C++".
        self.assertFalse(is_grounded("Java", RESUME))
        self.assertFalse(is_grounded("C", RESUME))
        self.assertFalse(is_grounded("SQL", RESUME))

    def test_ungrounded_lists_what_is_missing_in_order(self) -> None:
        self.assertEqual(ungrounded(["MySQL", "Python", "C++", "Docker", ""], RESUME), ["Python", "Docker", ""])

    def test_ground_keeps_an_item_that_is_present(self) -> None:
        self.assertEqual(ground("Node.js", RESUME), "Node.js")

    def test_ground_strips_a_leading_category_label(self) -> None:
        self.assertEqual(ground("Version control MySQL", RESUME), "MySQL")

    def test_ground_returns_none_when_nothing_matches(self) -> None:
        self.assertIsNone(ground("Python", RESUME))
        self.assertIsNone(ground("", RESUME))


class WordOverlapTests(unittest.TestCase):
    def test_reordered_words_from_the_source_fully_overlap(self) -> None:
        self.assertEqual(word_overlap("2013-2017", "May 2013 to May 2017"), 1.0)

    def test_partial_overlap_is_a_fraction(self) -> None:
        self.assertEqual(word_overlap("built four dashboards", "Built dashboards in React"), 2 / 3)

    def test_empty_text_has_no_overlap(self) -> None:
        self.assertEqual(word_overlap("", "anything"), 0.0)


class StatedSkillDurationTests(unittest.TestCase):
    def test_reads_the_sample_resume_layout(self) -> None:
        text = "Skill Details \nPython- Exprience - 24 months\nAWS- Exprience - Less than 1 year months\n"
        self.assertEqual(
            stated_skill_durations(text),
            [("Python", "24 months"), ("AWS", "Less than 1 year months")],
        )

    def test_reads_common_variations(self) -> None:
        text = "• Python: Experience: 3 years\nJava - Experience - 10+ years"
        self.assertEqual(stated_skill_durations(text), [("Python", "3 years"), ("Java", "10+ years")])

    def test_a_line_naming_several_skills_gives_one_pair_each(self) -> None:
        self.assertEqual(
            stated_skill_durations("Java, Javascript,- Exprience - 6 months"),
            [("Java", "6 months"), ("Javascript", "6 months")],
        )

    def test_a_long_list_of_skills_on_one_line_is_split(self) -> None:
        line = "Basic Computer, Electrical Machine, Electrical wiring, Solar power Plant, and Distribution Substation- Exprience - 12 months"
        self.assertEqual(
            [skill for skill, _ in stated_skill_durations(line)],
            ["Basic Computer", "Electrical Machine", "Electrical wiring", "Solar power Plant", "Distribution Substation"],
        )

    def test_stops_at_the_duration_when_the_next_heading_is_glued_on(self) -> None:
        # The sample resumes run the last duration into the next heading.
        self.assertEqual(stated_skill_durations("SQL- Exprience - 6 monthsCompany Details"), [("SQL", "6 months")])

    def test_ordinary_sentences_are_not_matched(self) -> None:
        self.assertEqual(stated_skill_durations("I have experience - mostly in design - of 3 systems."), [])


class StatedTotalExperienceTests(unittest.TestCase):
    def test_reads_an_outright_statement(self) -> None:
        for text, expected in [
            ("Junior developer with 2 years of experience building apps", "2 years"),
            ("5+ years of professional software development experience", "5+ years"),
            ("Having 9yrs of IT experience in testing", "9yrs"),
            ("Total experience: 18 months", "18 months"),
            ("Overall work experience of 5.5 years", "5.5 years"),
            ("I have experience of over 7 years in sales", "7 years"),
            ("3 years 6 months of experience", "3 years 6 months"),
        ]:
            self.assertEqual(stated_total_experience(text), expected, text)

    def test_time_spent_on_one_skill_is_not_a_total(self) -> None:
        self.assertIsNone(stated_total_experience("Python- Exprience - 24 months\nSQL- Exprience - 6 months"))

    def test_nothing_is_worked_out_from_dates_or_job_lengths(self) -> None:
        self.assertIsNone(stated_total_experience("Worked for 2 years at Acme. Jan 2019 to Mar 2021."))


class ClosestSpanTests(unittest.TestCase):
    TEXT = "company - Wab It Softwere Pvt.  Ltd.\ndescription - Jr. Java Developer"

    def test_returns_the_resume_wording_for_a_corrected_spelling(self) -> None:
        self.assertEqual(closest_span("Wab It Software Pvt. Ltd.", self.TEXT), "Wab It Softwere Pvt. Ltd.")

    def test_returns_none_when_nothing_is_close(self) -> None:
        self.assertIsNone(closest_span("Google Inc", self.TEXT))


if __name__ == "__main__":
    unittest.main()
