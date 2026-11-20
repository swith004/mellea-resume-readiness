# mellea-resume-readiness

FIU Student Mellea Project: Resume Skills-Gap Analyzer

Compare a resume against a real job posting and see exactly which skills are missing — plus suggestions to close the gap. Built with Mellea so every "(missing) skill" is grounded in the actual postings, never made up.

## Setup

### Prerequisites
- **Python 3.11+**. uv can install it for you if you don't have it.
- **[uv](https://docs.astral.sh/uv/)**, the Python package and environment manager:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **An LLM backend**: [Ollama](https://ollama.com) running an IBM Granite model locally, or a hosted Granite endpoint.

### Install
```bash
git clone https://github.com/swith004/mellea-resume-readiness.git   # or your fork
cd mellea-resume-readiness
uv sync          # creates .venv with Python 3.11 and installs Mellea from uv.lock
```

### Verify
```bash
uv run main.py
# -> mellea-resume-readiness is set up. Mellea version: 0.8.0
```

### Backend (Ollama + Granite)
```bash
ollama pull granite4.2:3b     # default model used by mellea.start_session()
ollama serve                  # if Ollama isn't already running
```

### Day-to-day
- Run anything inside the environment with `uv run <command>`, or activate it with `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).
- Add a dependency with `uv add <package>`. This updates `pyproject.toml` and `uv.lock`, so commit both.
- No uv? Plain pip works too: `python3.11 -m venv .venv && source .venv/bin/activate && pip install mellea`.


## Test data: real job postings

`data/` holds real postings pulled from the [USAJOBS Search API](https://developer.usajobs.gov/), saved as JSON. The included file, `data/postings_information-technology_miami-fl.json`, has 20 postings for the keyword "information technology" in Miami, FL. Each file records the query, when it was fetched, and a list of postings with `title`, `organization`, `locations`, `salary`, `summary`, `duties`, `qualifications`, `url`, and so on.

### Re-running the fetch
1. Request a free API key at <https://developer.usajobs.gov/apirequest/>.
2. Add the key and the email you registered with to `.env` (this file is gitignored):
   ```
   USAJOBS_API_KEY=your-key
   USAJOBS_EMAIL=you@example.com
   ```
3. Run the fetch. It asks for a keyword and a location:
   ```
   $ uv run scripts/fetch_postings.py
   Keyword: cybersecurity
   Location: Miami, FL
   Saved 20 postings (20 available) to data/postings_cybersecurity_miami-fl.json
   ```
   You can also pass them as options to skip the prompts. Both options are required when the script isn't run from a terminal (for example, from another script):
   ```bash
   uv run scripts/fetch_postings.py --keyword "cybersecurity" --location "Orlando, FL"
   ```
   The output goes to `data/postings_<keyword>_<location>.json`, overwriting any earlier fetch of the same query. Run `--help` to see all options.

> **Tip:** USAJOBS only lists federal jobs, and federal job titles rarely say "software developer". That keyword returned just 1 posting in Miami, while "information technology" returned 188, and the top 20 are almost all tech roles. That's the search used for the included file.

## Test data: sample resumes

`scripts/fetch_resumes.py` downloads the [Resume Dataset](https://huggingface.co/datasets/Sachinkelenjaguri/Resume_dataset) from Hugging Face and saves it as JSON. The included file, `data/resumes.json`, has 962 resumes across 25 job categories (only 166 are unique, the source repeats the rest). It records the source, when it was fetched, and a list of resumes with `category` and `resume` (the full text).

### Re-running the fetch
No API key is needed:
```bash
uv run scripts/fetch_resumes.py
```
The output goes to `data/resumes.json`, overwriting any earlier fetch.

### Browsing and loading the resumes
`scripts/load_resumes.py` reads `data/resumes.json` and prints the resumes. With no arguments it lists all 962 resumes, duplicates included, one line each: `#<index>  [<category>]  <char count>  <first 60 chars>...`, then a summary line.
```bash
uv run scripts/load_resumes.py # lists every resume
uv run scripts/load_resumes.py --category "Data Science" --unique --limit 5
uv run scripts/load_resumes.py --search "tensorflow" --full --limit 1
uv run scripts/load_resumes.py --index 12 --full
```

The flags are optional and can be combined. A resume must match all of them to be shown.

| Flag | What it does |
| --- | --- |
| `--category NAME` | Only resumes in this category (case-insensitive). An unknown name prints the valid categories with counts and exits with status 2. |
| `--search TEXT` | Only resumes whose text contains `TEXT` (case-insensitive). |
| `--unique` | Drop duplicate resume texts, keeping the first. |
| `--limit N` | Show at most `N` resumes. |
| `--index I` | Show only resume `#I`. Indexes start at 0 and always refer to the position in the file, whatever other filters are used. |
| `--full` | Print the complete resume text (with a header line) instead of the one-line summary. |
| `--max-chars N` | With `--full`, cut each resume off after `N` characters and note how many were left out. Default `0` means no limit. |
| `--list-categories` | Print each category and its resume count, then exit. |


## Resume extraction: `ResumeProfile`

`analyzer/` turns raw resume text into typed data.

### The data classes
`analyzer/schemas.py` holds the data classes and nothing else, so any group can import them and build test objects by hand:
```python
from analyzer import Education, Experience, ResumeProfile

profile = ResumeProfile(
    skills=["Python", "SQL"],
    experience=[Experience(title="Intern", organization="Acme")],
    education=[Education(degree="B.S.", field_of_study="Computer Science")],
)
```

| `ResumeProfile` field | Type | What it holds |
| --- | --- | --- |
| `skills` | `list[str]` | Skills, tools, and technologies, one per item |
| `skill_experience` | `list[SkillExperience]` | How long a skill has been used, where the resume says so: `skill`, `duration` |
| `experience` | `list[Experience]` | Jobs and internships: `title`, `organization`, `start`, `end`, `description` |
| `projects` | `list[Project]` | Named projects, kept apart from jobs: `name`, `description` |
| `education` | `list[Education]` | Degrees and diplomas: `degree`, `field_of_study`, `institution`, `year`, `grade` |
| `certifications` | `list[str]` | Certifications, licenses, and completed training courses |
| `achievements` | `list[str]` | Awards, honors, scholarships, and publications |
| `languages` | `list[str]` | Spoken languages (kept out of `skills`) |
| `total_experience` | `str \| None` | Total work experience, only when the resume states it (`"5 years"`) |
| `work_authorization` | `str \| None` | Citizenship, visa status, or security clearance, only when stated |

- `skills`, `experience`, and `education` are the core sections. `profile.missing_sections()` lists the core sections that came back empty, and `profile.is_fully_populated` is true when none did. Everything else is filled only when the resume has it.
- Every field inside an entry is optional and is `None` when the resume does not state it. Dates, durations, and grades are kept as written (`"May 2021"`, `"Present"`, `"24 months"`, `"3.6 GPA"`).
- Lists of plain strings are trimmed and de-duplicated ignoring case.
- Why these fields: the job postings ask for years of experience (35 of 40), certifications or licenses (25 of 40), citizenship (25 of 40), and a degree (22 of 40), so the profile records what a resume says about each.
- Left out on purpose: name, contact details, date of birth, gender, and marital status. A skills-gap check does not need them, and they should not influence it. Also left out: summary text, links, and hobbies, which no posting requirement is matched against.

### The extractor
`analyzer/extract_resume.py` holds the Mellea `@generative` functions and the `extract_resume(text) -> ResumeProfile` wrapper around them:
```python
from analyzer.extract_resume import extract_resume

profile = extract_resume(resume_text)
```
It reads the resume in focused passes (skills, work history, education, and a few stated facts), because the model mixed sections up when asked for everything at once. Three things are read directly from the text instead of being left to the model:
- Lines such as `Python - Experience - 24 months` (`analyzer/grounding.py`).
- A stated total such as `5 years of experience` (`analyzer/grounding.py`).
- Labelled employers such as `company - Acme Corp` or `Employer: Acme Corp` (`analyzer/layout.py`). Every labelled employer becomes a job, and the model describes each job from that job's own text.

When a resume does not label its employers, a second pass asks the model only for employer names and adds any the work-history pass left out.

### Nothing invented
Before a profile is returned, `ground_profile` checks it against the resume text in plain Python (`analyzer/grounding.py`), so the result does not rely on the model being honest:
- A skill, certification, job title, employer, degree, field, school, or project name must appear in the resume as whole words (ignoring case, spacing, punctuation, and accents). If the model only changed a spelling, the resume's wording is restored; otherwise the value is removed.
- Every word of a date or duration must come from the resume, and a date the resume states once cannot sit on both a school and a job.
- A description or achievement is removed when fewer than 80% of its words come from the resume.
- A value must suit its field. A spoken language must be a known one (so `Java` is not a language), an achievement must mention an award, rank, or publication, a grade must look like a grade, and work authorization must mention citizenship, a visa, or a clearance.

Everything removed is logged as a warning.

### Running it
Run from the project root. It prints the profile as JSON and exits with status 1 if a core section is empty:
```bash
uv run python -m analyzer.extract_resume --index 1                       # resume #1 in data/resumes.json
uv run python -m analyzer.extract_resume --file tests/fixtures/sample_resume.txt
```

### Tests
```bash
uv run python -m unittest discover -s tests -t .                         # no model calls
RUN_LLM_TESTS=1 uv run python -m unittest tests.test_extract_resume      # calls the model
```

### How well it does
Measured on the 166 unique resumes in `data/resumes.json`, using the employers those resumes label (`company - ...`, 372 in all) as the answer key:

| | Employers found | Resumes with every employer found |
| --- | --- | --- |
| Model alone, one work-history pass | 286 of 372 (76.9%) | 113 of 166 |
| With labelled employers read from the text | 370 of 372 (99.5%) | 164 of 166 |
| Same resumes with the labels removed | 280 of 372 (75.3%) | 108 of 166 |

The two not found in the second row are resumes whose label reads "Na" and "None". The third row is the case of a resume that does not label its employers: there the second, employers-only pass is what lifts recall from 54.8% to 75.3%. No run produced an unreadable answer.

### Known limits
The default model is small (Granite 4.2, 3B).
- On a resume that does not label its employers, about one employer in four is still missed.
- The employers-only pass adds names with no title or description. Most are real employers, but some are clients, courts, or product names that the resume mentions. A job with only an `organization` came from that pass.
- The model sometimes files a value under the wrong field (a job title as a field of study, for example).
- Stated total experience and skill durations are only read in the common layouts (`5 years of experience`, `Python - Experience - 24 months`).

The checks above stop the extractor from adding things the resume does not say; they cannot make it find everything.

## Posting extraction: `JobRequirements`

`analyzer/` also turns a job posting into typed data, built the same way as the resume extractor so the two can be compared field by field.

### The data class
`JobRequirements` lives in `analyzer/schemas.py` next to `ResumeProfile`, so any group can build one by hand:
```python
from analyzer import JobRequirements

requirements = JobRequirements(
    required_skills=["Customer Service", "Active Directory"],
    qualifications=["installing, configuring, and troubleshooting IT equipment"],
    work_authorization="U.S. Citizen",
)
```

| `JobRequirements` field | Type | What it holds | Compared with `ResumeProfile` |
| --- | --- | --- | --- |
| `required_skills` | `list[str]` | Skills, tools, and competencies the posting requires | `skills` |
| `preferred_skills` | `list[str]` | Skills the posting calls preferred, desired, or a plus | `skills` |
| `qualifications` | `list[str]` | Required experience or qualification statements, in the posting's words | `experience`, `projects` |
| `education` | `list[str]` | Required degrees or schooling | `education` |
| `certifications` | `list[str]` | Required certifications and licenses | `certifications` |
| `languages` | `list[str]` | Required spoken languages | `languages` |
| `min_experience` | `str \| None` | Minimum experience, as written (`"one year"`) | `total_experience` |
| `work_authorization` | `str \| None` | Required citizenship, visa, or clearance (`"U.S. Citizen"`) | `work_authorization` |

- `required_skills` and `qualifications` are the core sections that `missing_sections()` and `is_fully_populated` check. `preferred_skills` is not core: most USAJOBS postings mark nothing as preferred.
- A skill listed as both required and preferred is kept once, as required, so the match step does not count it twice.
- Left out on purpose: title, agency, salary, and locations. The posting dict already has them, and none of them is a requirement a resume can meet.

### The extractor
`analyzer/extract_posting.py` holds the Mellea `@generative` functions and the `extract_requirements(posting) -> JobRequirements` wrapper. It takes a posting dict from `data/postings_*.json` or plain posting text:
```python
from analyzer.extract_posting import extract_requirements

requirements = extract_requirements(posting)
```
`posting_text()` joins the posting's `summary`, `duties`, `qualifications`, `education`, and `requirements` fields, each with a label. That text is what the model reads and what every value is checked against. The model gets two focused passes (skills, then qualifications, education, certifications, and languages). Three things are read straight from the text in `analyzer/posting_layout.py`, because USAJOBS writes them the same way every time:
- Citizenship (`You must be a U.S. Citizen ...`), or a security clearance.
- Minimum experience (`one year of specialized experience ...`). Definitions such as "One year of experience refers to full-time work" are skipped.
- OPM competencies listed as `Name - definition` in the Qualifications section (`Attention to Detail - Is thorough ...`).

On the 40 saved postings, these readers find citizenship in 22, minimum experience in 29, and a competency list in 12, with no model calls.

### Nothing invented
`ground_requirements` checks the result against the posting in plain Python, reusing `analyzer/grounding.py`:
- A skill must appear in the posting as whole words; a label in front of it is trimmed, and a "skill" longer than six words is dropped.
- Certifications and languages must appear word for word (a changed spelling is restored), and a language must be a known spoken language.
- A qualification or education entry is removed when fewer than 80% of its words come from the posting. An education entry must name a degree or schooling, so the boilerplate "There is no educational substitution at this grade level" is not kept.
- Minimum experience and work authorization must come word for word from the posting.

Everything removed is logged as a warning. Pass a list to collect it as well, which is how the baseline comparison (#16) counts invented values. An entry with a reason in parentheses, such as `(a sentence, not a skill)`, was in the posting but in the wrong field; an entry with no reason was not in the posting at all:
```python
removed: list[str] = []
requirements = ground_requirements(requirements, text, removed)
```

### Running it
Run from the project root. It prints the requirements as JSON and exits with status 1 if a core section is empty:
```bash
uv run python -m analyzer.extract_posting --index 0        # posting #0 in the IT postings file
uv run python -m analyzer.extract_posting --postings data/postings_construction_miami-fl.json --index 3
uv run python -m analyzer.extract_posting --file tests/fixtures/sample_posting.json
```

### Tests
```bash
uv run python -m unittest discover -s tests -t .                  # no model calls
RUN_LLM_TESTS=1 uv run python -m unittest tests.test_extract_posting  # calls the model
```

### Known limits
- Three of the 40 saved postings (direct-hire notices) are under 1,000 characters and only link elsewhere, so they cannot come back fully populated.
- Federal postings describe skills as competencies and duties more than as tool names, so `required_skills` holds phrases such as "Customer Service" that a resume rarely repeats word for word. Matching them is left to the match step.
- The fixed-wording readers only cover USAJOBS phrasing; postings from other sources rely on the model alone.
