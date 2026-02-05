
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Create directory for model if it doesn't exist
os.makedirs("surge-pricing-dashboard/backend/model", exist_ok=True)

# --- 1. Generate Synthetic Data ---
num_samples = 1000
np.random.seed(42)

data = {
    'time_of_day': np.random.randint(0, 24, size=num_samples),
    'day_of_week': np.random.randint(0, 7, size=num_samples),
    'location': np.random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone D'], size=num_samples),
    'demand_level': np.random.uniform(1, 10, size=num_samples),
    'supply_level': np.random.uniform(1, 10, size=num_samples),
    'weather': np.random.choice(['Clear', 'Rainy', 'Cloudy', 'Snowy'], size=num_samples),
    'is_event': np.random.choice([True, False], size=num_samples, p=[0.1, 0.9])
}
df = pd.DataFrame(data)

# --- 2. Define Target Variable (Surge Multiplier) ---
# Create a simple formula to generate a plausible surge multiplier
base_surge = 1.0
demand_factor = df['demand_level'] / df['supply_level']
time_factor = np.sin(df['time_of_day'] * (2 * np.pi / 24)) * 0.5 + 1 # Peak hours bonus
weather_factor = df['weather'].apply(lambda x: 1.5 if x == 'Rainy' else (1.8 if x == 'Snowy' else 1.0))
event_factor = df['is_event'].apply(lambda x: 1.5 if x else 1.0)

# Combine factors and add some noise
surge_multiplier = base_surge * demand_factor * time_factor * weather_factor * event_factor
surge_multiplier += np.random.normal(0, 0.2, size=num_samples)
df['surge_multiplier'] = np.clip(surge_multiplier, 1.0, 5.0).round(2)

# --- 3. Preprocessing and Model Training ---
X = df.drop('surge_multiplier', axis=1)
y = df['surge_multiplier']

# Identify categorical and numerical features
categorical_features = ['location', 'weather']
numerical_features = ['time_of_day', 'day_of_week', 'demand_level', 'supply_level', 'is_event']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'
)

# Create the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

print(f"Model trained. Score on test set: {model_pipeline.score(X_test, y_test):.2f}")

# --- 4. Save the Model and Columns ---
model_path = 'surge-pricing-dashboard/backend/model/surge_model.pkl'
joblib.dump(model_pipeline, model_path)

# Save the columns used for training
columns_path = 'surge-pricing-dashboard/backend/model/model_columns.pkl'
training_columns = list(X.columns)
joblib.dump(training_columns, columns_path)

print(f"Model and training columns saved to: {os.path.dirname(model_path)}")
