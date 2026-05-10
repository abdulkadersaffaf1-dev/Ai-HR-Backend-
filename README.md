# HR AI Agent — FastAPI Backend

## Project structure

```
hr_ai_backend/
├── app/
│   ├── __init__.py
│   ├── main.py          ← FastAPI routes (4 endpoints)
│   └── ml_engine.py     ← All scoring logic ported from the notebook
├── models/              ← ⚠ Place your 3 pkl files here
│   ├── xgboost_resume_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── data/                ← ⚠ Place your 5 CSV files here
│   ├── Resume.csv
│   ├── jobs_combined_extended.csv
│   ├── courses_combined.csv
│   ├── faq_clean.csv
│   └── interview_clean.csv
├── requirements.txt
└── README.md
```

---

## Step 1 — Export your artifacts from Colab

Run this at the end of your Colab notebook (Cell 18 already has this):

```python
import joblib
from google.colab import files

joblib.dump(best_model, 'xgboost_resume_model.pkl')
joblib.dump(tfidf,      'tfidf_vectorizer.pkl')
joblib.dump(le,         'label_encoder.pkl')

files.download('xgboost_resume_model.pkl')
files.download('tfidf_vectorizer.pkl')
files.download('label_encoder.pkl')
```

Move the 3 downloaded `.pkl` files into the `models/` folder.

---

## Step 2 — Copy your CSV datasets

Copy these 5 files from your Colab / Google Drive into the `data/` folder:

- `Resume.csv`
- `jobs_combined_extended.csv`
- `courses_combined.csv`
- `faq_clean.csv`
- `interview_clean.csv`

---

## Step 3 — Create a virtual environment and install dependencies

```bash
cd hr_ai_backend

# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac / Linux)
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Download spaCy model (needed by the cleaner)
python -m spacy download en_core_web_sm
```

---

## Step 4 — Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First startup takes ~30–60 seconds because the Sentence-Transformer model
is downloaded once and the resume matrix is pre-computed.

---

## Step 5 — Test the API

Open your browser at:  http://localhost:8000/docs

This gives you Swagger UI where you can test every endpoint interactively.

---

## API Reference

### GET /
Health check.

### GET /stats
Dashboard counts — total resumes, categories, jobs, courses.

### GET /jobs
Full list of job titles available for matching.

### GET /categories
All 24 job categories the classifier knows.

---

### POST /evaluate  (Engine 1)
Evaluate a candidate that already exists in the dataset.

**Form fields:**
| Field       | Type   | Example         |
|-------------|--------|-----------------|
| resume_id   | int    | 21830565        |
| job_title   | string | "HR Consultant" |

**Response:**
```json
{
  "resume_id": 21830565,
  "predicted_category": "HR",
  "target_job": "HR Consultant",
  "target_category": "HR",
  "score": 72.4,
  "verdict": {"label": "SHORTLIST / INTERVIEW", "code": "shortlist"},
  "score_breakdown": {
    "semantic_similarity": 68.1,
    "percentile_rank": 81.3,
    "skill_coverage": 60.0,
    "category_fit": 94.5
  },
  "skills_matched": ["recruitment", "onboarding"],
  "skills_missing": ["hris", "payroll", "labour law"],
  "recommended_courses": ["HR Analytics Fundamentals", ...],
  "faqs": [{"question": "...", "answer": "..."}],
  "interview_questions": [{"question": "...", "guideline": "..."}]
}
```

---

### GET /leaderboard  (Engine 2)
Rank every resume in the dataset against a job.

**Query params:**
| Param     | Type   | Default | Example            |
|-----------|--------|---------|--------------------|
| job_title | string | —       | "Data Scientist"   |
| top_n     | int    | 10      | 20                 |

**Response:**
```json
{
  "job_title": "Data Scientist",
  "job_category": "DATA SCIENCE",
  "total_ranked": 2484,
  "top_n": 10,
  "candidates": [
    {
      "rank": 1,
      "resume_id": 1042,
      "category": "DATA SCIENCE",
      "score": 87.2,
      "semantic_sim": 91.0,
      "skill_coverage": 80.0,
      "category_fit": 95.1,
      "percentile_rank": 84.3,
      "matched_skills": ["python", "machine learning"],
      "verdict": {"label": "WORTH HIRING", "code": "hire"}
    },
    ...
  ]
}
```

---

### POST /upload  (Engine 3)
Evaluate a brand-new resume file in real time.

**Form fields:**
| Field     | Type   | Notes                    |
|-----------|--------|--------------------------|
| file      | File   | PDF or TXT               |
| job_title | string | "Software Engineer"      |

**Response:** Same shape as `/evaluate` with `filename` instead of `resume_id`.

---

## How the scoring works

The score formula is **identical** across all three engines, matching the notebook exactly:

```
score = (0.45 × semantic_similarity)
      + (0.35 × percentile_rank)
      + (0.15 × skill_coverage)
      + (0.05 × category_fit)
```

Penalty rule: if the job requires specific skills and the candidate
has **none** of them, the score is hard-capped at 40%.

Verdict thresholds:
- ≥ 75% → WORTH HIRING
- 55–74% → SHORTLIST / INTERVIEW
- 40–54% → CONSIDER WITH TRAINING
- < 40%  → REJECT

---

## VS Code tips

1. Install the **Python** and **REST Client** extensions.
2. Select your `venv` as the Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter").
3. Open the **terminal** (Ctrl+`) and run the uvicorn command from Step 4.
4. Use http://localhost:8000/docs to test endpoints without writing any extra code.
