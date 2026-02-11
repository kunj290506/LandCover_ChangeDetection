from datetime import datetime, timedelta
from typing import Dict, List

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .settings import Settings

security = HTTPBearer()


def create_access_token(subject: str, roles: List[str], settings: Settings) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": subject,
        "roles": roles,
        "iat": now,
        "exp": now + timedelta(seconds=settings.access_token_expire_seconds),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(credentials: HTTPAuthorizationCredentials = Security(security), settings: Settings = Depends(Settings)) -> Dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    return payload


def require_role(required: str):
    def _checker(payload: Dict = Depends(decode_token)) -> Dict:
        roles = payload.get("roles", [])
        if required not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return payload

    return _checker
