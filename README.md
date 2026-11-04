# Cloud-Based Student Performance Prediction System

A Flask + scikit-learn web app that predicts a student's estimated score from attendance, study hours, previous score and assignment score. It stores records in Supabase when cloud credentials are configured; otherwise it uses SQLite locally for an offline demonstration.

## 1. Run locally (fastest)
Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

## 2. Connect the cloud database (Supabase)
1. Create a project at https://supabase.com/.
2. Open SQL Editor and run the contents of `supabase_schema.sql`.
3. In Project Settings -> API, copy the Project URL and anon/public key.
4. Set environment variables before starting the app:

Windows PowerShell:
```powershell
$env:SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
$env:SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
python app.py
```
macOS/Linux:
```bash
export SUPABASE_URL="https://YOUR-PROJECT.supabase.co"
export SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
python app.py
```

The app uses Supabase when both variables are present. Otherwise it uses local SQLite. The included SQL policies are deliberately permissive for a classroom demo; do not use them for real student data or production.

## 3. Deploy to Render
1. Push this folder to a GitHub repository.
2. In Render, choose **New -> Web Service** and connect the repository.
3. Render detects `render.yaml`. Confirm the build command is `pip install -r requirements.txt && python train_model.py` and start command is `gunicorn app:app`.
4. Add `SUPABASE_URL` and `SUPABASE_KEY` in Render Environment settings.
5. Deploy. Visit the Render URL.

## Inputs and output
- Attendance: 0-100%
- Study hours per day: 0-16
- Previous score: 0-100
- Assignment score: 0-100
- Output: estimated score and one of Excellent / Good / Needs Support

## Important academic limitation
The included model is trained on **synthetic demonstration data**, not real student records. It proves the end-to-end software workflow but does not establish real-world predictive accuracy. Do not present its sample MAE/R² as evidence about actual students. For a research-grade project, train and evaluate on an appropriate, consented dataset and document its limitations.

## Project structure
- `app.py` - Flask web application, prediction endpoint, database operations
- `train_model.py` - synthetic data generation and model training
- `model/performance_model.joblib` - trained model artifact
- `templates/index.html` - dashboard UI
- `static/style.css` - styling
- `supabase_schema.sql` - cloud database schema
- `render.yaml` - Render deployment configuration
