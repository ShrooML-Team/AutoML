# api/main.py

from fastapi import FastAPI, HTTPException
from api.schemas import *
from api import service
from api.db import init_db
from fastapi import Depends, Body
from api.auth import create_user, verify_password, create_access_token
from api.db import SessionLocal, User
from sqlalchemy.orm import Session
from api.auth import get_current_user
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm


app = FastAPI(
    title="AutoML API",
    version="0.1.0",
    description="API exposing AutoML functionality with user authentication",
)

@app.on_event("startup")
def startup_event():
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    user = create_user(req.username, req.password)
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    db = SessionLocal()
    user = db.query(User).filter(User.username == form_data.username).first()
    db.close()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}



@app.post("/fit", response_model=StatusResponse)
def fit(req: FitRequest, current_user: User = Depends(get_current_user)):
    try:
        service.fit_model(req.X, req.y, req.automl_params)
        return {"status": "model trained"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, current_user: User = Depends(get_current_user)):
    try:
        preds = service.predict_model(req.X)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/eval", response_model=EvalResponse)
def evaluate(req: EvalRequest, current_user: User = Depends(get_current_user)):
    try:
        scores = service.eval_model()
        return {"scores": scores}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/me")
def read_users_me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}