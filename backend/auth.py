import secrets
import hashlib
import time
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from .database import get_db
import sqlite3
from typing import Dict, Optional

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def create_api_key() -> str:
    """Generate a secure API key"""
    # Create a secure random key
    raw_key = secrets.token_urlsafe(32)
    # Hash it for storage (never store raw keys)
    hashed_key = hash_api_key(raw_key)
    # Return the raw key to the user, store the hash
    return raw_key, hashed_key

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage"""
    # Add a pepper (secret value) for extra security
    pepper = "your-secret-pepper-value"  # In real app, store in env variable
    combined = api_key + pepper
    return hashlib.sha256(combined.encode()).hexdigest()

def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash"""
    pepper = "your-secret-pepper-value"
    combined = api_key + pepper
    return secrets.compare_digest(
        hashlib.sha256(combined.encode()).hexdigest(),
        stored_hash
    )

def get_user_from_db(api_key: str):
    """Look up user by API key in database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, api_key_hash FROM users WHERE api_key_hash IS NOT NULL", 
            ()
        )
        users = cursor.fetchall()
        
        # Verify the key against all users (inefficient but secure)
        for user in users:
            user_id, email, api_key_hash = user
            if api_key_hash and verify_api_key(api_key, api_key_hash):
                return {"id": user_id, "email": email}
        
        return None

def get_current_user(api_key: str = Depends(api_key_header)):
    """Verify API key and return user info"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing"
        )
    
    user = get_user_from_db(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    return user

def create_user(email: str) -> Dict:
    """Create a new user with API key"""
    raw_api_key, hashed_api_key = create_api_key()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (email, api_key_hash, created_at) VALUES (?, ?, datetime('now'))",
            (email, hashed_api_key)
        )
        user_id = cursor.lastrowid
        conn.commit()
    
    return {
        "id": user_id,
        "email": email,
        "api_key": raw_api_key  # Return raw key only on creation
    }