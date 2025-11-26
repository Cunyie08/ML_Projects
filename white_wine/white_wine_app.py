from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

# Let's load our saved model and scaler
model = joblib.load('best_model.pkl')

scaler = joblib.load('scaler.pkl')

# Let's run our application 
app = FastAPI()

# Lets class validation using pydantic
class white_wine_features(BaseModel):
    fixed_acidity: float
    volatile_acidity: float 
    citric_acid : float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# Let's create our end_point
@app.get("/")
def home():
    return {"Message": "Welcome to Tee's wine quality predictor"}

# Let's create a prediction endpoint
@app.post("/predict")
def wine_predict(wine:white_wine_features):
    # Convert it to a 2D numpy array
    features = np.array([[
    wine.fixed_acidity,
    wine.volatile_acidity,
    wine.citric_acid,
    wine.residual_sugar,
    wine.chlorides,
    wine.free_sulfur_dioxide,
    wine.total_sulfur_dioxide,
    wine.density,
    wine.pH,
    wine.sulphates,
    wine.alcohol
    ]])

    # Scale the feature
    scaled_features = scaler.transform(features)

    # Predict using our model
    prediction = model.predict(scaled_features)

    return {"Predicted quality": str(prediction[0])}

# Let's run our application
# uvicorn white_wine_app:app --reload

