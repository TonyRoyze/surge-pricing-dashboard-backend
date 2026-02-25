
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

# --- 1. Application Setup ---
app = FastAPI(
    title="Surge Pricing Prediction API",
    description="An API to predict trip prices using a trained SHAP-selected feature model.",
    version="1.0.0"
)

# --- 2. CORS Configuration ---
# Allow requests from the frontend (which will be running on a different port)
# Supports comma-separated overrides via CORS_ORIGINS env var.
default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",  # Default Vite port
    "http://127.0.0.1:5173",
    "https://surge-pricing-dashboard-frontend.vercel.app",
]

origins_env = os.getenv("CORS_ORIGINS")
origins = [o.strip() for o in origins_env.split(",")] if origins_env else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHAP_FEATURE_COLUMNS = [
    "Number_of_Riders",
    "Number_of_Drivers",
    "Expected_Ride_Duration",
]

# --- 3. Model Loading ---
# Load the selected SHAP-based model once at startup
model_path = os.path.join(os.path.dirname(__file__), "model", "exp6_shap_features.pkl")

try:
    model = joblib.load(model_path)
    print("SHAP feature model loaded successfully.")
except FileNotFoundError:
    print("Error: SHAP model file not found.")
    model = None
except Exception as exc:
    print(f"Error loading SHAP model: {exc}")
    model = None


# --- 4. Pydantic Models for API I/O ---
class SurgeFeatures(BaseModel):
    Number_of_Riders: float
    Number_of_Drivers: float
    Expected_Ride_Duration: float

    class Config:
        json_schema_extra = {
            "example": {
                "Number_of_Riders": 120,
                "Number_of_Drivers": 45,
                "Expected_Ride_Duration": 22
            }
        }

class PredictionResponse(BaseModel):
    predicted_price: float


# --- 5. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Surge Pricing Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(features: SurgeFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Cannot make predictions.")

    # Keep feature order consistent with training.
    input_df = pd.DataFrame([features.dict()], columns=SHAP_FEATURE_COLUMNS)

    prediction = float(model.predict(input_df)[0])

    return PredictionResponse(
        predicted_price=round(prediction, 2)
    )
