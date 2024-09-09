from fastapi import FastAPI, HTTPException
import os
import pickle

app = FastAPI()

# 모델 로드 함수
def load_model():
    try:
        # 현재 작업 디렉토리에서 모델 파일 경로 설정
        model_path = os.path.join(os.getcwd(), "regression.pkl")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.get("/regression")
def predict_weight(l: float):
    regression_model = load_model()

    try:
        prediction = regression_model.predict([[l**2, l]])
    
        return {"weight": prediction[0]}  # 배열에서 첫 번째 값 반환
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
