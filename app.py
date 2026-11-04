import os, sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np

from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file if present

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-key")
DB_PATH = os.getenv("SQLITE_PATH", "students.db")
MODEL_PATH = os.path.join("model", "performance_model.joblib")

# Optional cloud database: set SUPABASE_URL and SUPABASE_KEY in deployment environment.
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
CLOUD_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

def get_supabase():
    if not CLOUD_ENABLED:
        return None
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                attendance REAL NOT NULL,
                study_hours REAL NOT NULL,
                previous_score REAL NOT NULL,
                assignment_score REAL NOT NULL,
                predicted_score REAL NOT NULL,
                performance TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def all_students():
    if CLOUD_ENABLED:
        try:
            rows = get_supabase().table("students").select("*").order("created_at", desc=True).execute().data
            return rows or []
        except Exception as exc:
            app.logger.exception("Supabase read failed")
            flash("Cloud database connection failed. Check your Supabase settings/table.", "error")
            return []
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute("SELECT * FROM students ORDER BY id DESC").fetchall()]

def save_student(record):
    if CLOUD_ENABLED:
        get_supabase().table("students").insert(record).execute()
    else:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""
                INSERT INTO students
                (name, attendance, study_hours, previous_score, assignment_score,
                 predicted_score, performance, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (record["name"], record["attendance"], record["study_hours"],
                  record["previous_score"], record["assignment_score"],
                  record["predicted_score"], record["performance"], record["created_at"]))

def delete_student(student_id):
    if CLOUD_ENABLED:
        get_supabase().table("students").delete().eq("id", student_id).execute()
    else:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("DELETE FROM students WHERE id = ?", (student_id,))

def predict(attendance, study_hours, previous_score, assignment_score):
    model = joblib.load(MODEL_PATH)
    value = float(model.predict([[attendance, study_hours, previous_score, assignment_score]])[0])
    value = max(0.0, min(100.0, value))
    label = "Excellent" if value >= 80 else "Good" if value >= 65 else "Needs Support"
    return round(value, 1), label

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name = request.form["name"].strip()
            attendance = float(request.form["attendance"])
            study_hours = float(request.form["study_hours"])
            previous_score = float(request.form["previous_score"])
            assignment_score = float(request.form["assignment_score"])
            if not name:
                raise ValueError("Student name is required.")
            if not (0 <= attendance <= 100 and 0 <= study_hours <= 16 and
                    0 <= previous_score <= 100 and 0 <= assignment_score <= 100):
                raise ValueError("Check the input ranges and try again.")
            score, label = predict(attendance, study_hours, previous_score, assignment_score)
            record = {
                "name": name, "attendance": attendance, "study_hours": study_hours,
                "previous_score": previous_score, "assignment_score": assignment_score,
                "predicted_score": score, "performance": label,
                "created_at": datetime.utcnow().isoformat()
            }
            save_student(record)
            flash(f"Prediction saved for {name}.", "success")
            return redirect(url_for("index"))
        except Exception as exc:
            flash(str(exc), "error")

    students = all_students()
    scores = [float(s.get("predicted_score", 0)) for s in students]
    average = round(sum(scores) / len(scores), 1) if scores else 0
    counts = {
        "Excellent": sum(1 for s in students if s.get("performance") == "Excellent"),
        "Good": sum(1 for s in students if s.get("performance") == "Good"),
        "Needs Support": sum(1 for s in students if s.get("performance") == "Needs Support")
    }
    return render_template("index.html", students=students, total=len(students),
                           average=average, counts=counts, cloud=CLOUD_ENABLED)

@app.post("/delete/<student_id>")
def delete(student_id):
    try:
        delete_student(student_id)
        flash("Student record deleted.", "success")
    except Exception:
        flash("Could not delete record. Check database configuration.", "error")
    return redirect(url_for("index"))

@app.get("/health")
def health():
    return {"status": "ok", "database": "Supabase" if CLOUD_ENABLED else "SQLite (local fallback)"}

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
