import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Create a dummy model for testing
# Features: [State, Crop, Soil_Type, Fertilizer, N, P, K, Soil_pH, Rainfall_mm, Year]
# 10 features
X = np.random.rand(100, 10)
y = np.random.rand(100) * 60000  # Yield up to 60000

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
print("Model created successfully as model.pkl")
