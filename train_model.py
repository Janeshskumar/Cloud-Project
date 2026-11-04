"""Train a reproducible demonstration model on synthetic sample data.
Replace this with an ethically sourced, consented dataset for real research use."""
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

rng = np.random.default_rng(42)
n = 2500
attendance = rng.uniform(35, 100, n)
study_hours = rng.uniform(0, 10, n)
previous_score = rng.uniform(20, 100, n)
assignment_score = rng.uniform(20, 100, n)
noise = rng.normal(0, 6, n)
target = (0.25 * attendance + 1.8 * study_hours +
          0.35 * previous_score + 0.25 * assignment_score + noise)
target = np.clip(target, 0, 100)
X = np.column_stack([attendance, study_hours, previous_score, assignment_score])
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=180, max_depth=12, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
Path("model").mkdir(exist_ok=True)
joblib.dump(model, "model/performance_model.joblib")
print(f"Model saved. Demo MAE: {mean_absolute_error(y_test, pred):.2f}; R²: {r2_score(y_test, pred):.3f}")
