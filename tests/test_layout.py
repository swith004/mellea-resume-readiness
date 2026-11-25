"""Tests for reading labelled employers straight from the text. No model calls."""

from __future__ import annotations

import unittest

from analyzer.layout import labeled_jobs

DATASET_STYLE = """Education Details
January 2015 B.E Pune University
Skill Details
Python- Exprience - 24 monthsCompany Details
company - Acme Corp
description - Built the billing system.
Jan 2019 to Mar 2021
company - Globex Pvt. Ltd.
description - Jr. Analyst
company - Mumbai Monorail
description - """


class LabeledJobsTests(unittest.TestCase):
    def test_finds_every_labelled_employer_in_order(self) -> None:
        self.assertEqual(
            [job.organization for job in labeled_jobs(DATASET_STYLE)],
            ["Acme Corp", "Globex Pvt. Ltd.", "Mumbai Monorail"],
        )

    def test_each_job_gets_only_its_own_text(self) -> None:
        jobs = labeled_jobs(DATASET_STYLE)
        self.assertEqual(jobs[0].body, "Built the billing system.\nJan 2019 to Mar 2021")
        self.assertEqual(jobs[1].body, "Jr. Analyst")

    def test_an_employer_with_no_text_is_still_found(self) -> None:
        self.assertEqual(labeled_jobs(DATASET_STYLE)[2].body, "")

    def test_other_label_styles_are_read(self) -> None:
        text = "EXPERIENCE\nEmployer: Acme Corp\nResponsibilities: Built things.\n\nOrganisation Name - Globex\nDid stuff"
        self.assertEqual(
            [(job.organization, job.body) for job in labeled_jobs(text)],
            [("Acme Corp", "Built things."), ("Globex", "Did stuff")],
        )

    def test_the_last_job_stops_at_the_next_section_heading(self) -> None:
        text = "Company: Acme Corp\nBuilt things.\n\nSKILLS\nPython, SQL\n\nEDUCATION\nB.S."
        self.assertEqual(labeled_jobs(text)[0].body, "Built things.")

    def test_a_resume_without_labels_gives_nothing(self) -> None:
        text = "Software Engineer, Acme Corp\nMay 2020 - Present\nMy company - a startup - was great."
        self.assertEqual(labeled_jobs(text), [])

    def test_a_blank_label_keeps_its_text_but_has_no_name(self) -> None:
        jobs = labeled_jobs("company - \ndescription - Ran the front desk.")
        self.assertEqual([(job.organization, job.body) for job in jobs], [(None, "Ran the front desk.")])

    def test_a_blank_label_with_no_text_is_skipped(self) -> None:
        self.assertEqual(labeled_jobs("company - \ndescription - "), [])

    def test_a_sentence_after_the_label_is_not_an_employer_name(self) -> None:
        text = "company - Monitoring multiple database servers and web servers and cloud based services every day\ndescription - Ops."
        self.assertIsNone(labeled_jobs(text)[0].organization)

    def test_an_exact_repeat_is_returned_once(self) -> None:
        text = "company - Acme Corp\ndescription - Built things.\ncompany - Acme Corp\ndescription - Built things."
        self.assertEqual(len(labeled_jobs(text)), 1)


if __name__ == "__main__":
    unittest.main()
