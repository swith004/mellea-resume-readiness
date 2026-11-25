"""Tests for the checks applied to a profile after the model returns it. No model calls."""

from __future__ import annotations

import logging
import unittest

from analyzer.extract_resume import ground_profile
from analyzer.schemas import Education, Experience, Project, ResumeProfile, SkillExperience

RESUME = """Education Details
May 2013 to May 2017 B.E UIT-RGPV
January 2010 Diploma Electronics Government Polytechnic

Skill Details
Python- Exprience - 24 months
Database: MySQL, Oracle

Company Details
company - Wab It Softwere Pvt. Ltd.
description - Worked as a java developer building billing software for retail clients.
company - Acme Corp
description - Jr. Analyst from March 2018 to June 2019.

Project: Campus Shuttle Tracker - live shuttle map.
Certified Scrum Master
Total experience: 4 years. Indian citizen. CGPA 8.2.
Languages known: English, Hindi
Won first prize in the state level coding contest.
"""


def setUpModule() -> None:
    logging.disable(logging.WARNING)  # the removals under test are logged as warnings


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


class GroundProfileTests(unittest.TestCase):
    def test_a_profile_copied_from_the_resume_is_unchanged(self) -> None:
        profile = ResumeProfile(
            skills=["Python", "MySQL"],
            skill_experience=[SkillExperience(skill="Python", duration="24 months")],
            experience=[Experience(title="Jr. Analyst", organization="Acme Corp", start="March 2018", end="June 2019")],
            projects=[Project(name="Campus Shuttle Tracker", description="live shuttle map")],
            education=[Education(degree="B.E", institution="UIT-RGPV", year="May 2013 to May 2017")],
            certifications=["Certified Scrum Master"],
        )
        self.assertEqual(ground_profile(profile, RESUME), profile)

    def test_invented_skills_and_certifications_are_dropped(self) -> None:
        profile = ResumeProfile(skills=["Python", "Kubernetes"], certifications=["Certified Scrum Master", "PMP"])
        grounded = ground_profile(profile, RESUME)
        self.assertEqual(grounded.skills, ["Python"])
        self.assertEqual(grounded.certifications, ["Certified Scrum Master"])

    def test_a_category_label_is_trimmed_off_a_skill(self) -> None:
        grounded = ground_profile(ResumeProfile(skills=["Tools MySQL", "Tools Oracle"]), RESUME)
        self.assertEqual(grounded.skills, ["MySQL", "Oracle"])

    def test_a_sentence_is_not_a_skill(self) -> None:
        profile = ResumeProfile(skills=["Worked as a java developer building billing software for retail clients"])
        self.assertEqual(ground_profile(profile, RESUME).skills, [])

    def test_a_corrected_spelling_is_put_back_as_the_resume_wrote_it(self) -> None:
        profile = ResumeProfile(experience=[Experience(organization="Wab It Software Pvt. Ltd.")])
        self.assertEqual(ground_profile(profile, RESUME).experience[0].organization, "Wab It Softwere Pvt. Ltd.")

    def test_an_invented_employer_is_cleared_and_an_empty_job_is_dropped(self) -> None:
        profile = ResumeProfile(
            experience=[
                Experience(title="Jr. Analyst", organization="Globex Industries"),
                Experience(title="Chief Wizard", organization="Globex Industries"),
            ]
        )
        jobs = ground_profile(profile, RESUME).experience
        self.assertEqual([(j.title, j.organization) for j in jobs], [("Jr. Analyst", None)])

    def test_an_education_date_is_not_allowed_on_a_job(self) -> None:
        # The model copied the degree's dates onto the job; the resume states them once.
        profile = ResumeProfile(
            experience=[Experience(organization="Acme Corp", start="May 2013", end="May 2017")],
            education=[Education(degree="B.E", year="May 2013 to May 2017")],
        )
        job = ground_profile(profile, RESUME).experience[0]
        self.assertEqual((job.start, job.end), (None, None))

    def test_a_job_keeps_dates_the_resume_gives_it(self) -> None:
        profile = ResumeProfile(
            experience=[Experience(organization="Acme Corp", start="March 2018", end="June 2019")],
            education=[Education(degree="B.E", year="May 2013 to May 2017")],
        )
        job = ground_profile(profile, RESUME).experience[0]
        self.assertEqual((job.start, job.end), ("March 2018", "June 2019"))

    def test_a_date_with_words_not_in_the_resume_is_cleared(self) -> None:
        profile = ResumeProfile(experience=[Experience(organization="Acme Corp", start="April 2020")])
        self.assertIsNone(ground_profile(profile, RESUME).experience[0].start)

    def test_an_invented_description_is_cleared_but_a_copied_one_is_kept(self) -> None:
        profile = ResumeProfile(
            experience=[
                Experience(organization="Acme Corp", description="Led a global team of forty engineers across continents."),
                Experience(
                    organization="Wab It Softwere Pvt. Ltd.",
                    description="Worked as a java developer building billing software.",
                ),
            ]
        )
        jobs = ground_profile(profile, RESUME).experience
        self.assertIsNone(jobs[0].description)
        self.assertEqual(jobs[1].description, "Worked as a java developer building billing software.")

    def test_the_employer_is_removed_from_the_end_of_a_title(self) -> None:
        profile = ResumeProfile(experience=[Experience(title="Jr. Analyst - Acme Corp", organization="Acme Corp")])
        self.assertEqual(ground_profile(profile, RESUME).experience[0].title, "Jr. Analyst")

    def test_a_school_is_not_a_degree_or_a_field_of_study(self) -> None:
        profile = ResumeProfile(
            education=[Education(degree="UIT-RGPV", field_of_study="UIT-RGPV", institution="UIT-RGPV", year="May 2017")]
        )
        entry = ground_profile(profile, RESUME).education[0]
        self.assertEqual((entry.degree, entry.field_of_study, entry.institution), (None, None, "UIT-RGPV"))

    def test_a_repeated_education_entry_is_kept_once(self) -> None:
        profile = ResumeProfile(
            education=[
                Education(degree="Diploma", field_of_study="Electronics", institution="Government Polytechnic", year="January 2010"),
                Education(degree="Diploma", field_of_study="Electronics", institution="Polytechnic", year="January 2010"),
            ]
        )
        self.assertEqual(len(ground_profile(profile, RESUME).education), 1)

    def test_stated_facts_are_kept_and_invented_ones_are_cleared(self) -> None:
        stated = ResumeProfile(total_experience="4 years", work_authorization="Indian citizen")
        self.assertEqual(ground_profile(stated, RESUME), stated)
        invented = ResumeProfile(total_experience="12 years", work_authorization="Top Secret clearance")
        grounded = ground_profile(invented, RESUME)
        self.assertEqual((grounded.total_experience, grounded.work_authorization), (None, None))

    def test_spoken_languages_must_be_in_the_resume_and_are_moved_out_of_skills(self) -> None:
        profile = ResumeProfile(skills=["Python", "English", "Hindi"], languages=["English", "Hindi", "French"])
        grounded = ground_profile(profile, RESUME)
        self.assertEqual(grounded.languages, ["English", "Hindi"])
        self.assertEqual(grounded.skills, ["Python"])

    def test_a_job_title_is_not_also_a_skill(self) -> None:
        profile = ResumeProfile(
            skills=["Python", "Jr. Analyst"], experience=[Experience(title="Jr. Analyst", organization="Acme Corp")]
        )
        self.assertEqual(ground_profile(profile, RESUME).skills, ["Python"])

    def test_a_programming_language_is_not_a_spoken_language(self) -> None:
        # "Python" and "Oracle" are in the resume, but nobody speaks them.
        profile = ResumeProfile(languages=["English", "Python", "Oracle"])
        self.assertEqual(ground_profile(profile, RESUME).languages, ["English"])

    def test_a_job_duty_is_not_an_achievement(self) -> None:
        profile = ResumeProfile(
            achievements=["Worked as a java developer building billing software", "Won first prize in the state level coding contest"]
        )
        self.assertEqual(ground_profile(profile, RESUME).achievements, ["Won first prize in the state level coding contest"])

    def test_a_place_or_a_school_is_not_a_grade(self) -> None:
        profile = ResumeProfile(
            education=[
                Education(degree="B.E", grade="UIT-RGPV"),
                Education(degree="Diploma", grade="Government Polytechnic"),
                Education(degree="B.E", year="May 2017", grade="CGPA 8.2"),
            ]
        )
        self.assertEqual([entry.grade for entry in ground_profile(profile, RESUME).education], [None, None, "CGPA 8.2"])

    def test_a_repeated_education_entry_keeps_values_from_both_copies(self) -> None:
        profile = ResumeProfile(
            education=[Education(degree="B.E", institution="UIT-RGPV"), Education(degree="B.E", grade="CGPA 8.2")]
        )
        entries = ground_profile(profile, RESUME).education
        self.assertEqual([(e.degree, e.institution, e.grade) for e in entries], [("B.E", "UIT-RGPV", "CGPA 8.2")])

    def test_the_same_degree_at_two_schools_stays_as_two_entries(self) -> None:
        profile = ResumeProfile(
            education=[Education(degree="Diploma", institution="UIT-RGPV"), Education(degree="Diploma", institution="Government Polytechnic")]
        )
        self.assertEqual(len(ground_profile(profile, RESUME).education), 2)

    def test_a_bare_score_is_a_grade_but_a_year_is_not(self) -> None:
        text = RESUME + "GPA 3.7 in 2017. Scored 8.2/10."
        profile = ResumeProfile(
            education=[
                Education(degree="B.E", year="May 2013", grade="3.7"),
                Education(degree="B.E", year="May 2017", grade="8.2/10"),
                Education(degree="Diploma", grade="2017"),
            ]
        )
        self.assertEqual([entry.grade for entry in ground_profile(profile, text).education], ["3.7", "8.2/10", None])

    def test_work_authorization_must_mention_citizenship_a_visa_or_a_clearance(self) -> None:
        self.assertIsNone(ground_profile(ResumeProfile(work_authorization="Acme Corp"), RESUME).work_authorization)
        self.assertEqual(
            ground_profile(ResumeProfile(work_authorization="Indian citizen"), RESUME).work_authorization, "Indian citizen"
        )

    def test_an_invented_achievement_is_dropped(self) -> None:
        profile = ResumeProfile(achievements=["first prize in the state level coding contest", "Nobel Prize in Physics"])
        self.assertEqual(ground_profile(profile, RESUME).achievements, ["first prize in the state level coding contest"])

    def test_a_grade_must_come_from_the_resume(self) -> None:
        profile = ResumeProfile(education=[Education(degree="B.E", grade="CGPA 8.2"), Education(degree="Diploma", grade="4.0 GPA")])
        self.assertEqual([entry.grade for entry in ground_profile(profile, RESUME).education], ["CGPA 8.2", None])

    def test_a_heading_or_a_sentence_is_not_a_job_title(self) -> None:
        profile = ResumeProfile(
            experience=[
                Experience(title="Roles and Responsibilities", organization="Acme Corp"),
                Experience(title="Acme Corp", organization="Acme Corp"),
            ]
        )
        jobs = ground_profile(profile, RESUME).experience
        self.assertEqual([(job.title, job.organization) for job in jobs], [(None, "Acme Corp")])

    def test_the_same_job_listed_twice_is_merged(self) -> None:
        profile = ResumeProfile(
            experience=[
                Experience(title="Jr. Analyst", organization="Acme Corp"),
                Experience(title="Jr. Analyst", organization="Acme Corp", start="March 2018", end="June 2019"),
            ]
        )
        jobs = ground_profile(profile, RESUME).experience
        self.assertEqual([(job.title, job.start, job.end) for job in jobs], [("Jr. Analyst", "March 2018", "June 2019")])

    def test_two_stints_at_one_employer_stay_separate(self) -> None:
        profile = ResumeProfile(
            experience=[
                Experience(title="Jr. Analyst", organization="Acme Corp", start="March 2018"),
                Experience(title="Jr. Analyst", organization="Acme Corp", start="June 2019"),
            ]
        )
        self.assertEqual(len(ground_profile(profile, RESUME).experience), 2)

    def test_removals_are_logged(self) -> None:
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs("analyzer.extract_resume", level="WARNING") as logs:
                ground_profile(ResumeProfile(skills=["Kubernetes"]), RESUME)
        finally:
            logging.disable(logging.WARNING)
        self.assertIn("Kubernetes", logs.output[0])

    def test_an_invented_project_is_dropped(self) -> None:
        profile = ResumeProfile(projects=[Project(name="Campus Shuttle Tracker"), Project(name="Mars Rover Firmware")])
        self.assertEqual([p.name for p in ground_profile(profile, RESUME).projects], ["Campus Shuttle Tracker"])

    def test_a_stated_duration_is_kept_and_an_invented_one_is_dropped(self) -> None:
        profile = ResumeProfile(
            skill_experience=[
                SkillExperience(skill="Python", duration="24 months"),
                SkillExperience(skill="MySQL", duration="5 years"),
            ]
        )
        kept = ground_profile(profile, RESUME).skill_experience
        self.assertEqual([(s.skill, s.duration) for s in kept], [("Python", "24 months")])


if __name__ == "__main__":
    unittest.main()
