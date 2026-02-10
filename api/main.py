# api/main.py

from fastapi import FastAPI, HTTPException
from api.schemas import (
    FitRequest,
    PredictRequest,
    EvalRequest,
    StatusResponse,
    PredictResponse,
    EvalResponse,
)
from api import service
from api.db import init_db
from fastapi import Depends, Body
from api.auth import create_user, verify_password, create_access_token
from api.db import SessionLocal, User
from sqlalchemy.orm import Session
from api.auth import get_current_user
from fastapi import Depends


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

""" #TODO: limiter la création de comptes pour éviter les abus, ou ajouter une route d'admin pour créer les comptes manuellement
@app.post("/register")
def register(username: str = Body(...), password: str = Body(...)):
    user = create_user(username, password)
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
"""

@app.post("/login")
def login(username: str = Body(...), password: str = Body(...)):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if not user or not verify_password(password, user.hashed_password):
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
