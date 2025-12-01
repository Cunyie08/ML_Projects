from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from fastapi import FastAPI
import typing
import joblib
import numpy as np


# Let's load our model and scaler
scaler = joblib.load('scaler.pkl')
model = joblib.load('best_model.pkl')


# Let's run the app
app = FastAPI()

# Let's check the class validation using pydantic
class patient_data_features(BaseModel):
    radius_mean : float
    texture_mean: float
    smoothness_mean: float
    compactness_mean: float 
    concavity_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


# Let's create an endpoint
@app.get("/")
def home():
    return {"Message": "Welcome to Kay's Breast Cancer Predictor"}

# Let's create a prediction endpoit
@app.post("/predict")
def breast_cancer_predict(predictor:patient_data_features):
    # Convert it to a 2D numpy array
    features = np.array([[
    predictor.radius_mean,
    predictor.texture_mean,
    predictor.smoothness_mean,
    predictor.compactness_mean,
    predictor.concavity_mean,
    predictor.symmetry_mean,
    predictor.fractal_dimension_mean,
    predictor.radius_se,
    predictor.texture_se,
    predictor.smoothness_se,
    predictor.compactness_se,
    predictor.concavity_se,
    predictor.concave_points_se,
    predictor.symmetry_se,
    predictor.fractal_dimension_se,
    predictor.smoothness_worst,
    predictor.compactness_worst,
    predictor.concavity_worst,
    predictor.symmetry_worst,
    predictor.fractal_dimension_worst
    
    ]])

    # Scale the features
    scaled_features = scaler.transform(features)
    

    # predict using our model
    prediction = model.predict(scaled_features)

    return {"Prediction": prediction[0]}

# Let's run our application

#uvicorn breast_cancer_app:app --reload




