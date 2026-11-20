"""Tests for how jobs are assembled so that no employer is missed.

The model is replaced by a stand-in, so these make no model calls.
"""

from __future__ import annotations

import unittest
from unittest import mock

from analyzer import extract_resume as module
from analyzer.layout import LabeledJob
from analyzer.schemas import Experience


def fake_extract_job(answers: dict[str, Experience]):
    """A stand-in for the model: returns the prepared answer for each job's text."""

    def extract_job(session, *, job_text: str, model_options: dict) -> Experience:
        return answers.get(job_text, Experience())

    return extract_job


class JobsFromLabelsTests(unittest.TestCase):
    def build(self, labeled, answers=None, model_jobs=()):
        with mock.patch.object(module, "extract_job", fake_extract_job(answers or {})):
            return module._jobs_from_labels(None, labeled, list(model_jobs))

    def test_every_labelled_employer_becomes_a_job_even_if_the_model_finds_nothing(self) -> None:
        labeled = [LabeledJob("Acme Corp", "Built things."), LabeledJob("Globex", ""), LabeledJob("Initech", "Did stuff.")]
        jobs = self.build(labeled)
        self.assertEqual([job.organization for job in jobs], ["Acme Corp", "Globex", "Initech"])

    def test_details_come_from_the_jobs_own_text(self) -> None:
        body = "Senior Analyst\nJan 2019 to Mar 2021\nBuilt the billing system for retail clients."
        answer = Experience(title="Senior Analyst", start="Jan 2019", end="Mar 2021", description="Built the billing system for retail clients.")
        job = self.build([LabeledJob("Acme Corp", body)], {body: answer})[0]
        self.assertEqual((job.title, job.start, job.end), ("Senior Analyst", "Jan 2019", "Mar 2021"))
        self.assertEqual(job.description, "Built the billing system for retail clients.")

    def test_a_date_that_is_not_in_the_jobs_own_text_is_not_kept(self) -> None:
        body = "Built the billing system."
        answer = Experience(title="Analyst", start="May 2013", end="May 2017", description="Built the billing system.")
        job = self.build([LabeledJob("Acme Corp", body)], {body: answer})[0]
        self.assertEqual((job.start, job.end), (None, None))

    def test_an_invented_title_or_description_is_not_kept(self) -> None:
        body = "Built the billing system."
        answer = Experience(title="Chief Wizard", description="Led forty engineers across three continents.")
        job = self.build([LabeledJob("Acme Corp", body)], {body: answer})[0]
        self.assertEqual((job.title, job.description), (None, None))

    def test_a_heading_or_the_employer_name_is_not_a_title(self) -> None:
        for wrong_title in ("Roles and Responsibilities", "Acme Corp"):
            body = f"{wrong_title}\nBuilt things."
            job = self.build([LabeledJob("Acme Corp", body)], {body: Experience(title=wrong_title)})[0]
            self.assertIsNone(job.title, wrong_title)

    def test_a_missing_title_is_taken_from_the_whole_resume_pass(self) -> None:
        model_jobs = [Experience(title="Data Scientist", organization="Matelabs")]
        job = self.build([LabeledJob("Matelabs", "ML platform for business users.")], model_jobs=model_jobs)[0]
        self.assertEqual(job.title, "Data Scientist")

    def test_the_whole_resume_pass_does_not_duplicate_a_labelled_employer(self) -> None:
        # Same employer, spelled three ways by the model.
        model_jobs = [
            Experience(title="Developer", organization="Wab It Software Pvt. Ltd."),
            Experience(title="Developer", organization="Wab It"),
            Experience(title="Developer", organization="WAB IT SOFTWERE PVT LTD"),
        ]
        jobs = self.build([LabeledJob("Wab It Softwere Pvt. Ltd.", "")], model_jobs=model_jobs)
        self.assertEqual(len(jobs), 1)

    def test_an_employer_only_the_model_found_is_added(self) -> None:
        model_jobs = [Experience(title="Intern", organization="Hindustan Aeronautics Limited"), Experience(title="No employer")]
        jobs = self.build([LabeledJob("Acme Corp", "")], model_jobs=model_jobs)
        self.assertEqual([job.organization for job in jobs], ["Acme Corp", "Hindustan Aeronautics Limited"])


class AddMissedEmployersTests(unittest.TestCase):
    def test_adds_only_the_employers_that_are_not_already_listed(self) -> None:
        jobs = [Experience(title="Intern", organization="Sunshine Health Systems")]
        named = ["Sunshine Health Systems", "Florida International University", "sunshine health"]
        result = module._add_missed_employers(jobs, named)
        self.assertEqual(
            [(job.title, job.organization) for job in result],
            [("Intern", "Sunshine Health Systems"), (None, "Florida International University")],
        )

    def test_leaves_the_original_list_unchanged(self) -> None:
        jobs = [Experience(organization="Acme Corp")]
        module._add_missed_employers(jobs, ["Globex"])
        self.assertEqual(len(jobs), 1)


class SameEmployerTests(unittest.TestCase):
    def test_matches_the_same_name_written_differently(self) -> None:
        for a, b in [("Acme", "Acme Corp, Pune"), ("Wab It Software", "Wab It Softwere"), ("AT&T", "at&t")]:
            self.assertTrue(module._same_employer(a, b), (a, b))

    def test_different_employers_and_missing_names_do_not_match(self) -> None:
        self.assertFalse(module._same_employer("Acme Corp", "Globex Industries"))
        self.assertFalse(module._same_employer(None, "Acme Corp"))


if __name__ == "__main__":
    unittest.main()
