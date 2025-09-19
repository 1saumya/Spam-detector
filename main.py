from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('bow.pkl', 'rb') as f:
    bow = joblib.load(f)

class spam_predictor(BaseModel):
    text : str

# functions for prediction
def text_to_vec(text):
    text = bow.transform([text])
    return text

def predict(text):

    text = text_to_vec(text)
    pred = model.predict(text)
    return "Not Spam" if pred == 0 else "Spam"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def home():
    return {"Welcome to Spam Predictor"}

@app.post('/predict')
def prediction(data : spam_predictor):

    res = predict(data.text)
    return {"prediction" : res}