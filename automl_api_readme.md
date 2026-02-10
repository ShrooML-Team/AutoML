# AutoML API

API FastAPI pour entraîner, prédire et évaluer un modèle AutoML avec authentification JWT.

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer le serveur :
```bash
uvicorn api.main:app --reload
```

L'API sera accessible à `http://127.0.0.1:8000`.

## Endpoints

### 1. Register

Créer un utilisateur et récupérer un token JWT.

```bash
curl -X POST "http://127.0.0.1:8000/register" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass123"}'
```

Réponse :
```json
{
  "access_token": "<JWT_TOKEN>",
  "token_type": "bearer"
}
```

### 2. Login

Se connecter et récupérer un token JWT.

```bash
curl -X POST "http://127.0.0.1:8000/login" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass123"}'
```

Réponse :
```json
{
  "access_token": "<JWT_TOKEN>",
  "token_type": "bearer"
}
```

### 3. Fit (entraîner le modèle)

```bash
curl -X POST "http://127.0.0.1:8000/fit" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -d '{
     "X": [               
      {"x1": 1, "x2": 2},
      {"x1": 2, "x2": 3},
      {"x1": 3, "x2": 4},
      {"x1": 4, "x2": 5},
      {"x1": 5, "x2": 6},
      {"x1": 6, "x2": 7}, 
      {"x1": 7, "x2": 8},  
      {"x1": 8, "x2": 9},  
      {"x1": 9, "x2": 10},
      {"x1": 10, "x2": 11},
      {"x1": 11, "x2": 12},        
      {"x1": 12, "x2": 13}
    ],
    "y": [0,0,0,0,0,0,1,1,1,1,1,1], 
    "automl_params": {}
  }'
```

Réponse :
```json
{"status":"model trained"}
```

### 4. Predict (prédire avec le modèle)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -d '{
    "X": [
      {"x1": 1, "x2": 2},
      {"x1": 3, "x2": 4}
    ]
  }'
```

Réponse :
```json
{"predictions": [0,1]}
```

### 5. Eval (évaluer le modèle)

```bash
curl -X POST "http://127.0.0.1:8000/eval" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <JWT_TOKEN>" \
  -d '{
    "X": [
      {"x1": 1, "x2": 2},
      {"x1": 3, "x2": 4}
    ],
    "y": [0,1]
  }'
```

Réponse :
```json
{"scores": {"accuracy": 1.0}}
```

## Notes

- Toujours utiliser le token JWT retourné par `/login` ou `/register` pour accéder aux routes protégées (`/fit`, `/predict`, `/eval`).
- Le header doit être exactement : `Authorization: Bearer <JWT_TOKEN>`.
- Les modèles doivent être entraînés via `/fit` avant de pouvoir utiliser `/predict` ou `/eval`.
- Les données doivent correspondre au format JSON attendu (X = liste de dictionnaires, y = liste de labels).

---


