"""
HR AI Agent — FastAPI Backend
Mirrors the exact logic of the Colab notebook:
  Engine 1 → POST /evaluate
  Engine 2 → GET  /leaderboard
  Engine 3 → POST /upload
  Stats    → GET  /stats
  Helpers  → GET  /jobs  /categories
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.ml_engine import engine   # singleton loaded at startup

app = FastAPI(
    title="HR AI Agent API",
    version="1.0.0",
    description="Resume screening, candidate scoring, and leaderboard powered by XGBoost + Sentence-Transformers",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "HR AI Agent API is running"}


# ── Helpers ────────────────────────────────────────────────────────────────
@app.get("/jobs", tags=["helpers"])
def list_jobs():
    """Return every job title available for matching."""
    titles = sorted(engine.jobs_df["title"].dropna().unique().tolist())
    return {"count": len(titles), "titles": titles}


@app.get("/categories", tags=["helpers"])
def list_categories():
    """Return the 24 job categories the classifier knows."""
    cats = sorted(engine.le.classes_.tolist())
    return {"count": len(cats), "categories": cats}


@app.get("/stats", tags=["dashboard"])
def get_stats():
    """Aggregate counts for the dashboard home page."""
    return engine.get_stats()


# ── Engine 1 ───────────────────────────────────────────────────────────────
@app.post("/evaluate", tags=["Engine 1 — Evaluate candidate"])
def evaluate_candidate(resume_id: int = Form(...), job_title: str = Form(...)):
    """
    Evaluate a candidate from the existing resume pool.

    Body (form):
      resume_id  — integer ID from Resume.csv
      job_title  — job title string (fuzzy matched)

    Returns the full hiring report including score breakdown,
    skill gaps, recommended courses, FAQs, and interview questions.
    """
    result = engine.evaluate_by_id(resume_id, job_title)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Engine 2 ───────────────────────────────────────────────────────────────
@app.get("/leaderboard", tags=["Engine 2 — Job leaderboard"])
def get_leaderboard(job_title: str, top_n: int = 10):
    """
    Rank every resume in the pool against a job title.

    Query params:
      job_title  — job title to rank candidates for
      top_n      — how many top candidates to return (default 10)
    """
    if top_n < 1 or top_n > 100:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 100")
    result = engine.get_leaderboard(job_title, top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Engine 3 ───────────────────────────────────────────────────────────────
@app.post("/upload", tags=["Engine 3 — Upload resume"])
async def upload_resume(
    file: UploadFile = File(...),
    job_title: str = Form(...),
):
    """
    Evaluate a brand-new resume (PDF or TXT) in real time.

    Form:
      file       — PDF or plain-text resume file
      job_title  — job title string (fuzzy matched)

    Extracts text, cleans it, classifies it, scores it against the
    job using the same formula as Engines 1 and 2, and returns a
    full hiring report.
    """
    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    content = await file.read()
    result = engine.evaluate_uploaded_resume(content, file.filename, job_title)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
