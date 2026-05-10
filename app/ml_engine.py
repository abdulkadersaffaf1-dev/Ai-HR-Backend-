"""
ml_engine.py
============
Singleton that loads the trained artifacts once at startup and exposes
three scoring methods that mirror the three Colab engines exactly.

Scoring formula (identical across all three engines):
  score = (0.45 × semantic_sim)
        + (0.35 × percentile_rank)
        + (0.15 × skill_coverage)
        + (0.05 × category_fit)

Penalty: if job requires skills and candidate has none → cap score at 40.
"""

import ast
import io
import os
import re
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import contractions
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent   # hr_ai_backend/
MODELS_DIR  = BASE_DIR / "models"
DATA_DIR    = BASE_DIR / "data"

COURSE_TITLE_COLUMN = "title"


class HREngine:
    """
    Loads all artifacts and datasets once. Exposes three public methods:
      evaluate_by_id()          → Engine 1
      get_leaderboard()         → Engine 2
      evaluate_uploaded_resume()→ Engine 3
    """

    def __init__(self):
        print("🔄 Loading ML artifacts...")
        self.model    = joblib.load(MODELS_DIR / "xgboost_resume_model.pkl")
        self.tfidf    = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        self.le       = joblib.load(MODELS_DIR / "label_encoder.pkl")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._stop    = set(stopwords.words("english"))
        self._lemma   = WordNetLemmatizer()
        self.known_categories = set(self.le.classes_)
        print("✅ Artifacts loaded.")

        print("🔄 Loading datasets...")
        self._load_datasets()
        print("✅ Datasets ready.")

        print("🔄 Pre-computing resume TF-IDF matrix...")
        self.X_resume = self.tfidf.transform(
            self.resume_df["clean_resume"].fillna("")
        )
        print(f"✅ Matrix ready: {self.X_resume.shape}")

    # ── Data loading ───────────────────────────────────────────────────────
    def _load_datasets(self):
        self.resume_df    = pd.read_csv(DATA_DIR / "Resume.csv")
        self.jobs_df      = pd.read_csv(DATA_DIR / "jobs_combined_extended.csv")
        self.courses_df   = pd.read_csv(DATA_DIR / "courses_combined.csv")
        self.faq_df       = pd.read_csv(DATA_DIR / "faq_clean.csv")
        self.interview_df = pd.read_csv(DATA_DIR / "interview_clean.csv")

        # Normalise category columns (same as notebook)
        self.resume_df["Category"]            = self.resume_df["Category"].str.upper().str.strip()
        self.jobs_df["category"]              = self.jobs_df["category"].str.upper().str.strip()
        self.courses_df["category"]           = self.courses_df["category"].str.upper().str.strip()
        self.interview_df["category_clean"]   = (
            self.interview_df["category"].str.upper().str.strip()
        )

        # Pre-clean resume text if column missing
        if "clean_resume" not in self.resume_df.columns:
            print("  ↳ Pre-cleaning resume text (first run only, may take a minute)...")
            self.resume_df["clean_resume"] = self.resume_df["Resume_str"].apply(self.industrial_clean)

        # Build job match text if missing
        if "job_match_text" not in self.jobs_df.columns:
            self.jobs_df["job_match_text"] = self.jobs_df.apply(self._build_job_profile, axis=1)

        self.jobs_df["_title_key"] = (
            self.jobs_df["title"].fillna("").astype(str).map(self._norm_space)
        )

    # ── Text cleaning (industrial_clean from notebook) ─────────────────────
    def industrial_clean(self, text: str) -> str:
        if pd.isnull(text):
            return ""
        text = contractions.fix(str(text).lower())
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(
            r"http\S+|www\S+|\S+@\S+|(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
            " ", text,
        )
        text = re.sub(r"[^a-z0-9+#\s]", " ", text)
        tokens = [
            self._lemma.lemmatize(w)
            for w in text.split()
            if w not in self._stop and len(w) > 1
        ]
        return " ".join(tokens)

    # ── Text helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _norm_space(text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    @staticmethod
    def _norm_skill(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9+#/ ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    # ── Skill helpers ──────────────────────────────────────────────────────
    def _parse_job_skills(self, job_row: pd.Series) -> list:
        skills = []
        raw_list = job_row.get("skills_list", None)
        raw_list = "" if raw_list is None else str(raw_list).strip()
        if raw_list and raw_list.lower() != "nan":
            try:
                parsed = ast.literal_eval(raw_list)
                if isinstance(parsed, (list, tuple)):
                    skills = [self._norm_skill(x) for x in parsed]
            except Exception:
                pass
        if not skills:
            raw = str(job_row.get("skills_raw", "") or "").strip()
            if raw and raw.lower() != "nan":
                skills = [self._norm_skill(x) for x in re.split(r"[,;|]+", raw)]
        if not skills:
            raw = str(job_row.get("skills_clean", "") or "").strip()
            skills = [self._norm_skill(x) for x in raw.split()]
        skills = [s for s in skills if len(s) > 1]
        return list(dict.fromkeys(skills))

    @staticmethod
    def _skill_in_resume(skill: str, resume_text: str) -> bool:
        resume_tokens = set(re.sub(r"[^a-z0-9+#/ ]", " ", resume_text.lower()).split())
        skill_tokens  = [t for t in skill.split() if len(t) > 1]
        return bool(skill_tokens) and all(t in resume_tokens for t in skill_tokens)

    # ── Job profile / finder ───────────────────────────────────────────────
    def _build_job_profile(self, job_row: pd.Series) -> str:
        title    = str(job_row.get("title", ""))
        category = str(job_row.get("category", "")).replace("-", " ")
        skills   = str(job_row.get("skills_raw", "") or job_row.get("skills_clean", ""))
        combined = f"{title} {title} {category} {category} {skills} {skills} {job_row.get('job_text', '')}"
        return self.industrial_clean(combined)

    def _find_job(self, title_query: str):
        q = self._norm_space(title_query)
        if not q:
            return None
        exact = self.jobs_df[self.jobs_df["_title_key"] == q]
        if not exact.empty:
            return exact.iloc[0]
        contains = self.jobs_df[
            self.jobs_df["_title_key"].str.contains(q, regex=False, na=False)
        ].copy()
        if not contains.empty:
            contains["_len"] = contains["_title_key"].str.len()
            return contains.sort_values(["_len", "title"]).iloc[0]
        tokens = [t for t in q.split() if len(t) > 1]
        if not tokens:
            return None
        scored = self.jobs_df.copy()
        scored["_hits"] = scored["_title_key"].apply(
            lambda t: sum(tok in t.split() for tok in tokens)
        )
        scored = scored[scored["_hits"] > 0]
        if scored.empty:
            return None
        scored["_len"] = scored["_title_key"].str.len()
        return scored.sort_values(["_hits", "_len", "title"], ascending=[False, True, True]).iloc[0]

    # ── Similarity helpers ─────────────────────────────────────────────────
    def _semantic_sim(self, text_a: str, text_b: str) -> float:
        embs    = self.embedder.encode([text_a, text_b], convert_to_tensor=True)
        sim     = util.cos_sim(embs[0], embs[1]).item()
        raw     = max(0.0, sim) * 100
        scaled  = (raw - 20.0) / (85.0 - 20.0) * 100.0
        return float(np.clip(scaled, 0.0, 100.0))

    def _percentile_sim(self, resume_vec, job_vec):
        raw_sim    = float(cosine_similarity(resume_vec, job_vec)[0][0])
        all_sims   = cosine_similarity(job_vec, self.X_resume).flatten()
        sorted_s   = np.sort(all_sims)
        percentile = 100.0 * np.searchsorted(sorted_s, raw_sim, side="right") / len(sorted_s)
        return raw_sim, float(percentile)

    def _category_fit(self, resume_vec, target_category: str) -> float:
        cat = str(target_category).upper().strip()
        if cat not in self.known_categories:
            return 0.0
        try:
            if hasattr(self.model, "predict_proba"):
                idx = self.le.transform([cat])[0]
                return float(self.model.predict_proba(resume_vec)[0][idx]) * 100
        except Exception:
            pass
        predicted = self.le.inverse_transform(self.model.predict(resume_vec))[0]
        return 100.0 if predicted == cat else 0.0

    # ── MASTER scoring (shared across all 3 engines) ───────────────────────
    def _score(self, resume_vec, resume_text: str, job_row: pd.Series) -> dict:
        job_raw   = str(job_row.get("job_text", "") or job_row.get("job_match_text", ""))
        sem_sim   = self._semantic_sim(resume_text, job_raw)

        job_vec   = self.tfidf.transform([job_row["job_match_text"]])
        _, pctile = self._percentile_sim(resume_vec, job_vec)

        job_skills = self._parse_job_skills(job_row)
        matched    = [s for s in job_skills if self._skill_in_resume(s, resume_text)]
        missing    = [s for s in job_skills if s not in matched]
        skill_pct  = (len(matched) / len(job_skills) * 100.0) if job_skills else 0.0

        cat_fit = self._category_fit(resume_vec, job_row["category"])

        score = (
            (0.45 * sem_sim)
            + (0.35 * pctile)
            + (0.15 * skill_pct)
            + (0.05 * cat_fit)
        )
        if job_skills and not matched:
            score = min(score, 40.0)
        score = float(np.clip(score, 0, 95))

        return {
            "score"       : round(score, 1),
            "semantic_sim": round(sem_sim, 1),
            "percentile"  : round(pctile, 1),
            "skill_pct"   : round(skill_pct, 1),
            "category_fit": round(cat_fit, 1),
            "matched"     : matched,
            "missing"     : missing,
        }

    # ── Verdict helper ─────────────────────────────────────────────────────
    @staticmethod
    def _verdict(score: float) -> dict:
        if score >= 75:
            return {"label": "WORTH HIRING",        "code": "hire"}
        if score >= 55:
            return {"label": "SHORTLIST / INTERVIEW","code": "shortlist"}
        if score >= 40:
            return {"label": "CONSIDER WITH TRAINING","code": "training"}
        return {"label": "REJECT",                  "code": "reject"}

    # ── Course recommender ─────────────────────────────────────────────────
    def _recommend_courses(self, category: str, missing_skills: list) -> list:
        cat_courses = self.courses_df[
            self.courses_df["category"].astype(str).str.upper().str.strip() == category
        ]
        recs = []
        for skill in missing_skills[:5]:
            mask = (
                cat_courses[COURSE_TITLE_COLUMN].astype(str).str.contains(skill, case=False, na=False, regex=False)
                | cat_courses["skills"].astype(str).str.contains(skill, case=False, na=False, regex=False)
            )
            recs.extend(cat_courses.loc[mask, COURSE_TITLE_COLUMN].head(1).tolist())
            if len(recs) >= 3:
                break
        if len(recs) < 3:
            recs.extend(cat_courses[COURSE_TITLE_COLUMN].head(5).tolist())
        return list(dict.fromkeys(recs))[:3]

    # ── Interview questions ────────────────────────────────────────────────
    def _interview_questions(self, category: str, n: int = 5) -> list:
        pool = self.interview_df[self.interview_df["category_clean"] == category]
        generic = ["tell me about", "weakness", "hire you", "strengths", "overtime", "deadline"]
        smart   = pool[~pool["question"].str.contains("|".join(generic), case=False, na=False)]
        if smart.empty:
            smart = pool
        if smart.empty:
            smart = self.interview_df.sample(n, random_state=42)
        selected = smart.sample(n=min(n, len(smart)), random_state=42)
        return [
            {"question": row["question"], "guideline": row["answer_guideline"]}
            for _, row in selected.iterrows()
        ]

    # ── FAQs ───────────────────────────────────────────────────────────────
    def _faqs(self, category: str) -> list:
        if "category" in self.faq_df.columns:
            rows = self.faq_df[
                self.faq_df["category"].astype(str).str.upper().str.strip() == category
            ].head(2)
        else:
            rows = pd.DataFrame()
        if rows.empty:
            rows = self.faq_df.head(2)
        return [{"question": r["Question"], "answer": r["Answer"]} for _, r in rows.iterrows()]

    # ── Full report builder (shared shape for all engines) ─────────────────
    def _build_report(
        self,
        resume_id,
        resume_text: str,
        clean_text: str,
        resume_vec,
        job_row: pd.Series,
        filename: str = None,
    ) -> dict:
        pred_idx  = self.model.predict(resume_vec)[0]
        pred_cat  = self.le.inverse_transform([pred_idx])[0]
        target    = str(job_row["category"]).upper().strip()
        s         = self._score(resume_vec, resume_text, job_row)
        verdict   = self._verdict(s["score"])
        courses   = self._recommend_courses(target, s["missing"])
        qs        = self._interview_questions(target)
        faqs      = self._faqs(target)

        return {
            "resume_id"         : resume_id,
            "filename"          : filename,
            "predicted_category": pred_cat,
            "target_job"        : str(job_row["title"]),
            "target_category"   : target,
            "score"             : s["score"],
            "verdict"           : verdict,
            "score_breakdown"   : {
                "semantic_similarity": s["semantic_sim"],
                "percentile_rank"    : s["percentile"],
                "skill_coverage"     : s["skill_pct"],
                "category_fit"       : s["category_fit"],
            },
            "skills_matched"    : s["matched"][:6],
            "skills_missing"    : s["missing"][:6],
            "recommended_courses": courses,
            "faqs"              : faqs,
            "interview_questions": qs,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ENGINE 1 — Evaluate by Resume ID
    # ═══════════════════════════════════════════════════════════════════════
    def evaluate_by_id(self, resume_id: int, job_title: str) -> dict:
        row = self.resume_df[self.resume_df["ID"] == resume_id]
        if row.empty:
            return {"error": f"Resume ID {resume_id} not found in the dataset."}

        job = self._find_job(job_title)
        if job is None:
            return {"error": f"No job matching '{job_title}'. Call GET /jobs for the full list."}

        r   = row.iloc[0]
        vec = self.tfidf.transform([r["clean_resume"]])
        return self._build_report(
            resume_id=resume_id,
            resume_text=str(r["Resume_str"]),
            clean_text=str(r["clean_resume"]),
            resume_vec=vec,
            job_row=job,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # ENGINE 2 — Leaderboard (batch scoring)
    # ═══════════════════════════════════════════════════════════════════════
    def get_leaderboard(self, job_title: str, top_n: int = 10) -> dict:
        job = self._find_job(job_title)
        if job is None:
            return {"error": f"No job matching '{job_title}'. Call GET /jobs for the full list."}

        job_category = str(job["category"]).upper().strip()
        job_raw_text = str(job.get("job_text", "") or job.get("job_match_text", ""))
        job_vec      = self.tfidf.transform([job["job_match_text"]])

        # Batch semantic encoding (fast, runs once per request)
        resume_texts      = self.resume_df["Resume_str"].fillna("").tolist()
        job_emb           = self.embedder.encode(job_raw_text, convert_to_tensor=True)
        resume_embs       = self.embedder.encode(
            resume_texts, convert_to_tensor=True, batch_size=64, show_progress_bar=False
        )
        all_sem_raw  = util.cos_sim(job_emb, resume_embs)[0].cpu().numpy()
        all_sem_sims = np.clip(
            (np.clip(all_sem_raw, 0, 1) * 100 - 20.0) / (85.0 - 20.0) * 100.0,
            0.0, 100.0,
        )

        # Pool percentile baseline
        all_sims   = cosine_similarity(job_vec, self.X_resume).flatten()
        sorted_s   = np.sort(all_sims)

        # Category fit (batch)
        if job_category in self.known_categories and hasattr(self.model, "predict_proba"):
            prob_idx   = self.le.transform([job_category])[0]
            cat_probs  = self.model.predict_proba(self.X_resume)[:, prob_idx] * 100
        else:
            cat_probs  = np.zeros(len(self.resume_df))

        job_skills = self._parse_job_skills(job)

        rows = []
        for idx, r in self.resume_df.iterrows():
            sem_sim    = float(all_sem_sims[idx])
            raw_sim_i  = float(all_sims[idx])
            percentile = 100.0 * np.searchsorted(sorted_s, raw_sim_i, side="right") / len(sorted_s)
            cat_fit    = float(cat_probs[idx])

            matched   = [s for s in job_skills if self._skill_in_resume(s, str(r["clean_resume"]))]
            skill_pct = (len(matched) / len(job_skills) * 100.0) if job_skills else 0.0

            score = (0.45 * sem_sim) + (0.35 * percentile) + (0.15 * skill_pct) + (0.05 * cat_fit)
            if job_skills and not matched:
                score = min(score, 40.0)
            score = float(np.clip(score, 0, 95))

            rows.append({
                "rank"              : 0,
                "resume_id"         : int(r["ID"]),
                "category"          : str(r["Category"]),
                "score"             : round(score, 1),
                "semantic_sim"      : round(sem_sim, 1),
                "skill_coverage"    : round(skill_pct, 1),
                "category_fit"      : round(cat_fit, 1),
                "percentile_rank"   : round(percentile, 1),
                "matched_skills"    : matched[:4],
                "verdict"           : self._verdict(score),
            })

        ranked = sorted(rows, key=lambda x: (x["score"], x["skill_coverage"], x["category_fit"]), reverse=True)
        for i, r in enumerate(ranked, 1):
            r["rank"] = i

        return {
            "job_title"   : str(job["title"]),
            "job_category": job_category,
            "total_ranked": len(ranked),
            "top_n"       : top_n,
            "candidates"  : ranked[:top_n],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ENGINE 3 — Upload resume (new candidate)
    # ═══════════════════════════════════════════════════════════════════════
    def evaluate_uploaded_resume(self, content: bytes, filename: str, job_title: str) -> dict:
        # Extract raw text
        try:
            if filename.lower().endswith(".pdf"):
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    raw_text = " ".join(
                        p.extract_text() for p in pdf.pages if p.extract_text()
                    )
            else:
                raw_text = content.decode("utf-8", errors="ignore")
        except Exception as e:
            return {"error": f"Could not read file: {e}"}

        if not raw_text.strip():
            return {"error": "No readable text found in the uploaded file."}

        clean_text = self.industrial_clean(raw_text)
        if not clean_text.strip():
            return {"error": "Text extraction succeeded but produced empty content after cleaning."}

        job = self._find_job(job_title)
        if job is None:
            return {"error": f"No job matching '{job_title}'. Call GET /jobs for the full list."}

        vec = self.tfidf.transform([clean_text])
        return self._build_report(
            resume_id=None,
            resume_text=raw_text,
            clean_text=clean_text,
            resume_vec=vec,
            job_row=job,
            filename=filename,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Dashboard stats
    # ═══════════════════════════════════════════════════════════════════════
    def get_stats(self) -> dict:
        return {
            "total_resumes"    : int(len(self.resume_df)),
            "total_categories" : int(self.resume_df["Category"].nunique()),
            "total_jobs"       : int(len(self.jobs_df)),
            "total_courses"    : int(len(self.courses_df)),
            "categories"       : sorted(self.resume_df["Category"].unique().tolist()),
        }


# ── Singleton — loaded once when the module is first imported ──────────────
engine = HREngine()
