# api/schemas.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class FitRequest(BaseModel):
    X: List[Dict[str, Any]]
    y: List[Any]
    automl_params: Optional[Dict[str, Any]] = {}


class PredictRequest(BaseModel):
    X: List[Dict[str, Any]]


class EvalRequest(BaseModel):
    X: List[Dict[str, Any]]
    y: List[Any]


class StatusResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    predictions: List[Any]


class EvalResponse(BaseModel):
    scores: Dict[str, Any]

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

