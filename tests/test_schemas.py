"""Tests for the data classes. No model calls, so they run in well under a second.

    uv run python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import unittest

from analyzer.schemas import Education, Experience, Project, ResumeProfile, SkillExperience


class ResumeProfileTests(unittest.TestCase):
    def test_hand_built_profile_only_needs_the_fields_you_care_about(self) -> None:
        profile = ResumeProfile(
            skills=["Python"],
            experience=[Experience(title="Intern")],
            education=[Education(degree="B.S.")],
        )
        self.assertIsNone(profile.experience[0].organization)
        self.assertIsNone(profile.education[0].institution)
        self.assertEqual(profile.projects, [])
        self.assertEqual(profile.certifications, [])
        self.assertEqual(profile.skill_experience, [])
        self.assertTrue(profile.is_fully_populated)

    def test_empty_profile_reports_every_core_section_missing(self) -> None:
        profile = ResumeProfile()
        self.assertEqual(profile.missing_sections(), ["skills", "experience", "education"])
        self.assertFalse(profile.is_fully_populated)

    def test_missing_sections_names_only_the_empty_ones(self) -> None:
        profile = ResumeProfile(skills=["SQL"], education=[Education(degree="B.A.")])
        self.assertEqual(profile.missing_sections(), ["experience"])

    def test_optional_sections_do_not_count_as_missing(self) -> None:
        # Many resumes have no projects or certifications; that is not a failed extraction.
        profile = ResumeProfile(skills=["SQL"], experience=[Experience(title="Analyst")], education=[Education(degree="B.A.")])
        self.assertEqual(profile.missing_sections(), [])

    def test_skills_are_trimmed_and_deduplicated_ignoring_case(self) -> None:
        profile = ResumeProfile(skills=["  Python ", "python", "SQL", "", "   ", "Machine   Learning", "SQL"])
        self.assertEqual(profile.skills, ["Python", "SQL", "Machine Learning"])

    def test_certifications_are_cleaned_like_skills(self) -> None:
        profile = ResumeProfile(certifications=["PMP", " pmp ", "N/A", "CompTIA Security+"])
        self.assertEqual(profile.certifications, ["PMP", "CompTIA Security+"])

    def test_achievements_and_languages_are_cleaned_like_skills(self) -> None:
        profile = ResumeProfile(achievements=["Dean's List", "dean's list", ""], languages=["English", " english ", "Spanish"])
        self.assertEqual(profile.achievements, ["Dean's List"])
        self.assertEqual(profile.languages, ["English", "Spanish"])

    def test_single_facts_default_to_none_and_placeholders_are_cleared(self) -> None:
        self.assertIsNone(ResumeProfile().total_experience)
        self.assertIsNone(ResumeProfile().work_authorization)
        profile = ResumeProfile(total_experience=" 5  years ", work_authorization="N/A")
        self.assertEqual(profile.total_experience, "5 years")
        self.assertIsNone(profile.work_authorization)

    def test_total_experience_keeps_only_the_duration(self) -> None:
        for written, kept in [
            ("2 years of experience", "2 years"),
            ("5+ years of professional experience in testing", "5+ years"),
            ("18 months experience", "18 months"),
            ("3 years", "3 years"),
        ]:
            self.assertEqual(ResumeProfile(total_experience=written).total_experience, kept, written)

    def test_education_can_carry_a_grade(self) -> None:
        self.assertEqual(Education(degree="B.S.", grade=" 3.6 GPA ").grade, "3.6 GPA")
        self.assertIsNone(Education(degree="B.S.").grade)

    def test_a_job_needs_a_title_or_an_employer(self) -> None:
        profile = ResumeProfile(experience=[Experience(), Experience(organization="Acme"), Experience(title="Analyst")])
        self.assertEqual([(e.title, e.organization) for e in profile.experience], [(None, "Acme"), ("Analyst", None)])

    def test_an_education_entry_needs_a_degree_field_or_school(self) -> None:
        profile = ResumeProfile(education=[Education(year="2001"), Education(institution="Harvard", year="2013")])
        self.assertEqual([e.institution for e in profile.education], ["Harvard"])

    def test_unnamed_projects_and_incomplete_durations_are_dropped(self) -> None:
        profile = ResumeProfile(
            projects=[Project(description="no name"), Project(name="Shuttle Tracker")],
            skill_experience=[SkillExperience(skill="Python"), SkillExperience(skill="SQL", duration="24 months")],
        )
        self.assertEqual([p.name for p in profile.projects], ["Shuttle Tracker"])
        self.assertEqual([(s.skill, s.duration) for s in profile.skill_experience], [("SQL", "24 months")])

    def test_blank_and_placeholder_values_become_none(self) -> None:
        job = Experience(title="Analyst", organization="  ", start="", end="N/A", description=" Did \n things ")
        self.assertIsNone(job.organization)
        self.assertIsNone(job.start)
        self.assertIsNone(job.end)
        self.assertEqual(job.description, "Did things")

    def test_a_date_range_in_start_is_split_into_start_and_end(self) -> None:
        for written, start, end in [
            ("June 2015 to February 2018", "June 2015", "February 2018"),
            ("2019 - Present", "2019", "Present"),
            ("Jan 2020 – Mar 2021", "Jan 2020", "Mar 2021"),
        ]:
            job = Experience(title="Developer", start=written)
            self.assertEqual((job.start, job.end), (start, end), written)

    def test_a_single_date_and_an_existing_end_are_left_alone(self) -> None:
        self.assertEqual(Experience(title="Dev", start="Aug-2017").start, "Aug-2017")
        self.assertEqual(Experience(title="Dev", start="Aug 2017 till date").start, "Aug 2017 till date")
        job = Experience(title="Dev", start="2019 - 2020", end="2021")
        self.assertEqual((job.start, job.end), ("2019 - 2020", "2021"))

    def test_round_trips_through_json(self) -> None:
        profile = ResumeProfile(
            skills=["Python", "Flask"],
            skill_experience=[SkillExperience(skill="Python", duration="36 months")],
            experience=[Experience(title="Intern", organization="Acme", start="May 2025", end="Present")],
            projects=[Project(name="Shuttle Tracker", description="Live shuttle map")],
            education=[Education(degree="B.S.", field_of_study="Computer Science", year="2026", grade="3.6 GPA")],
            certifications=["CompTIA Security+"],
            achievements=["Dean's List"],
            languages=["English", "Spanish"],
            total_experience="3 years",
            work_authorization="U.S. Citizen",
        )
        self.assertEqual(ResumeProfile.model_validate_json(profile.model_dump_json()), profile)

    def test_model_sees_every_field_as_required(self) -> None:
        # The model skips fields it is told are optional, which left whole sections empty.
        schema = ResumeProfile.model_json_schema()
        self.assertEqual(
            schema["required"],
            [
                "skills",
                "skill_experience",
                "experience",
                "projects",
                "education",
                "certifications",
                "achievements",
                "languages",
                "total_experience",
                "work_authorization",
            ],
        )
        for name, fields in {
            "Experience": ["title", "organization", "start", "end", "description"],
            "Education": ["degree", "field_of_study", "institution", "year", "grade"],
            "Project": ["name", "description"],
            "SkillExperience": ["skill", "duration"],
        }.items():
            self.assertEqual(schema["$defs"][name]["required"], fields, name)

    def test_model_is_given_a_length_limit_for_every_list(self) -> None:
        # Without a limit the model can repeat entries until it runs out of tokens.
        properties = ResumeProfile.model_json_schema()["properties"]
        lists = {name: field for name, field in properties.items() if field.get("type") == "array"}
        self.assertEqual(len(lists), 8)
        for name, field in lists.items():
            self.assertIn("maxItems", field, name)


if __name__ == "__main__":
    unittest.main()
