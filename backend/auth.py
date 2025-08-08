import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from .database import get_db
import sqlite3

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def create_api_key() -> str:
    """Generate a secure API key"""
    return "sk-" + secrets.token_urlsafe(32)

def get_current_user(api_key: str = Depends(api_key_header)):
    """Verify API key and return user info"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing"
        )
    
    # In a real app, this would query your database
    # For now, we'll use a simple check
    if api_key != "student-api-key-123":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    return {"user_id": 1, "email": "student@example.com"}

# For database implementation (advanced)
def get_user_from_db(api_key: str):
    """Look up user by API key in database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email FROM users WHERE api_key = ?", 
            (api_key,)
        )
        user = cursor.fetchone()
        
        if not user:
            return None
            
        return {"id": user[0], "email": user[1]}