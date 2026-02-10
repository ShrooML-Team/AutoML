from passlib.context import CryptContext
from api.db import SessionLocal, User
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status


SECRET_KEY = "iloveshrooms"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Retrieve the current user based on JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    db.close()
    if user is None:
        raise credentials_exception
    return user

def create_access_token(data: dict, expires_delta: int | None = None):
    """Create a JWT token with expiration."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_password_hash(password: str) -> str:
    """Return hashed version of the password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that the plain password matches the hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def create_user(username: str, password: str):
    """Create a new user in the database."""
    db = SessionLocal()
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        db.close()
        return existing_user
    hashed = get_password_hash(password)
    user = User(username=username, hashed_password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    return user
