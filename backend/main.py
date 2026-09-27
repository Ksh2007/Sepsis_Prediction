from pathlib import Path
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Sepsis Prediction API",
    description="This FastAPI application provides sepsis predictions using a machine learning model.",
    version="1.0"
)

# FIX: Get the folder where main.py lives, then look for the pkl file there
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_and_key_components.pkl"

with open(MODEL_PATH, 'rb') as file:
    loaded_components = pickle.load(file)
    
model = loaded_components['model']
encoder = loaded_components['encoder']
scaler = loaded_components['scaler']

class InputData(BaseModel):
    PRG: int
    PL: float
    PR: float
    SK: float
    TS: int
    M11: float
    BD2: float
    Age: int

class OutputData(BaseModel):
    Sepsis: str

def preprocess_data(input_data: InputData):
    numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
    # Dict syntax updated to .model_dump() for modern Pydantic v2 compatibility
    input_data_scaled = scaler.transform([list(input_data.model_dump().values())])
    return pd.DataFrame(input_data_scaled, columns=numerical_cols)

def prediction_sepsis(input_data_scaled_df: pd.DataFrame):
    prediction_numeric = model.predict(input_data_scaled_df)
    predicted_label = encoder.inverse_transform(prediction_numeric)[0]
    return {"prediction": str(predicted_label)}

@app.get("/")
async def root():
    return "Sepsis Classification Project"

# FIX: Renamed function to avoid infinite loop recursion crash
@app.post("/predict/", response_model=OutputData)
async def predict_sepsis_endpoint(input_data: InputData):
    try:
        input_data_scaled_df = preprocess_data(input_data)
        # FIX: Calls prediction_sepsis instead of calling itself
        result_dict = prediction_sepsis(input_data_scaled_df)
        return {"Sepsis": result_dict["prediction"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    # Run the FastAPI application on the local host and port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)