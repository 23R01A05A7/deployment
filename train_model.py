import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Generate/Load Data
np.random.seed(42)
n_samples = 1000

X_raw = np.zeros((n_samples, 10))
X_raw[:, 0] = np.random.randint(0, 4, n_samples)
X_raw[:, 1] = np.random.randint(0, 4, n_samples)
X_raw[:, 2] = np.random.randint(0, 3, n_samples)
X_raw[:, 3] = np.random.randint(0, 3, n_samples)
X_raw[:, 4] = np.random.uniform(10, 150, n_samples)
X_raw[:, 5] = np.random.uniform(10, 100, n_samples)
X_raw[:, 6] = np.random.uniform(10, 100, n_samples)
X_raw[:, 7] = np.random.uniform(4.5, 8.5, n_samples)
X_raw[:, 8] = np.random.uniform(500, 2500, n_samples)
X_raw[:, 9] = np.random.randint(2010, 2025, n_samples)

y = (X_raw[:, 4] * 50 + X_raw[:, 5] * 30 + X_raw[:, 8] * 5 + (X_raw[:, 7] - 7)**2 * -100 + np.random.normal(0, 1000, n_samples))
y = np.clip(y, 5000, 60000)

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)

# 3. Build Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

# 4. Fit
print("Training model...")
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Evaluation ---")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 6. Save
joblib.dump(pipeline, 'model.pkl')
print("\nImproved model saved as model.pkl")
