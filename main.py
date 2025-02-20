from typing import Union
from fastapi import FastAPI

# model.py를 가져온다
import model

# 그 안에 있는 AndModel 클래스의 인스턴스를 생성
model = model.AndModel()

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# items/{item_id} -> 경로
# item_id : 경로 매개변수(파라미터)
@app.get("/items/{item_id}") # endpoint
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/predict/left/{left}/right/{right}") 
def predict(left: int, right: int):
    result = model.predict([left,right])
    return {"result": result}

@app.get("/train")
def train():
    model.train()
    return {"result": "OK"}