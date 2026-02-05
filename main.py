
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import os

# --- 1. Application Setup ---
app = FastAPI(
    title="Surge Pricing Prediction API",
    description="An API to predict surge pricing multipliers using a trained ML model.",
    version="1.0.0"
)

# --- 2. CORS Configuration ---
# Allow requests from the frontend (which will be running on a different port)
origins = [
    "http://localhost:3000",
    "http://localhost:5174", # Default Vite port
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Model and Columns Loading ---
# Load the trained model pipeline and the list of training columns
# This happens once when the application starts
model_path = os.path.join(os.path.dirname(__file__), "model", "surge_model.pkl")
columns_path = os.path.join(os.path.dirname(__file__), "model", "model_columns.pkl")

try:
    model = joblib.load(model_path)
    training_columns = joblib.load(columns_path)
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("Error: Model or columns file not found. Make sure 'train.py' has been run.")
    model = None
    training_columns = []


# --- 4. Pydantic Models for API I/O ---
class SurgeFeatures(BaseModel):
    time_of_day: int
    day_of_week: int
    location: str
    demand_level: float
    supply_level: float
    weather: str
    is_event: bool

    class Config:
        json_schema_extra = {
            "example": {
                "time_of_day": 20,
                "day_of_week": 3,
                "location": "Zone A",
                "demand_level": 8.5,
                "supply_level": 4.2,
                "weather": "Rainy",
                "is_event": True
            }
        }

class PredictionResponse(BaseModel):
    surge_multiplier: float
    confidence: float


# --- 5. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Surge Pricing Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_surge(features: SurgeFeatures):
    if not model or not training_columns:
        return {"error": "Model not loaded. Cannot make predictions."}

    # Convert input data to a pandas DataFrame
    # The model expects a DataFrame with specific column order
    input_df = pd.DataFrame([features.dict()], columns=training_columns)

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # --- Calculate Confidence Score ---
    # For a RandomForestRegressor, we can use the std dev of predictions from individual trees
    # as a measure of uncertainty (inverse of confidence).
    estimators = model.named_steps['regressor'].estimators_
    individual_tree_predictions = [tree.predict(model.named_steps['preprocessor'].transform(input_df)) for tree in estimators]
    
    # The predictions are inside a nested list, so we extract them
    individual_tree_predictions = [pred[0] for pred in individual_tree_predictions]

    # Calculate standard deviation
    std_dev = np.std(individual_tree_predictions)
    
    # Normalize the confidence score to be between 0 and 1
    # A lower std dev means higher confidence.
    # We use an exponential decay function to map std_dev to a 0-1 range.
    # The scaling factor 'k' can be tuned. A smaller 'k' makes the confidence drop faster.
    k = 0.1 
    confidence = np.exp(-k * std_dev)

    # Clean up surge multiplier to be a standard float
    surge_multiplier = float(prediction)

    return PredictionResponse(
        surge_multiplier=round(surge_multiplier, 2),
        confidence=round(confidence, 2)
    )
