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
