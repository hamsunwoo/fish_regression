import requests
import json

def lr_api(length):
    headers = {
        'accept': 'application/json',
    }

    params = {
        'length': length,
    }

    response = requests.get('http://127.0.0.1:8000/regression', params=params, headers=headers)
    data = json.loads(response.text)
    r = data['weight']
        
    return r

    

def kn_api(length, weight):
    headers = {
        'accept': 'application/json',
    }

    params = {
        'l': length,
        'w': weight,
    }

    response = requests.get('http://127.0.0.1:8002/fish', params=params, headers=headers)
    data = json.loads(response.text)
    r = data['prediction']

    return r

def predict():
    length = float(input("물고기의 길이를 입력하세요: "))
    
    ## weight 예측 선형회귀 API 호출 
    weight =lr_api(length)
    
    ## 물고기 분류 API 호출 
    fish_class = kn_api(length, weight)
    
    print(f"length:{length} 물고기는 weight:{weight} 으로 예측 되며 종류는 {fish_class} 입니다.")
