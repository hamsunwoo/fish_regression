from fastapi import FastAPI, HTTPException
import os
from fishregression.model.manager import get_model_path
import pickle

app = FastAPI()

# 모델 로드 함수
def load_model():
    try:
        model_path = get_model_path()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/regression")
def predict_weight(length: float):
    regression_model = load_model()

    try:
        prediction = regression_model.predict([[length**2, length]])
        return {
                "length": length,
                "weight": prediction[0]
                }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

