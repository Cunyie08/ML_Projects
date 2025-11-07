from sklearn.preprocessing import MinMaxScaler
from pydantic import BaseModel
from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np


# lets load our saved models
model = joblib.load('Best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Let's initialize our application

app = FastAPI()

# let's create our pydantic model
class winefeatures(BaseModel):
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

# Create the end point
@app.get("/")
def home():
    return {"Message": "Welcome to Cunyie wine quality predictor"}

# Prediction endpoint
@app.post("/predict")
def predict(wine:winefeatures):
# Convert the features into a 2D numpy array
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
    wine.alcohol,
    ]])

    # let scale our input features using the loaded scaler to normalize our input
    scaled_features = scaler.transform(features)

    # Lets make prediction
    prediction = model.predict(scaled_features)

# return the prediction and the prediction converted to string for serialization
    return {"Predicted_quality": str(prediction[0])}

# lets run our prediction app
# uvicorn wine_app:app --reload