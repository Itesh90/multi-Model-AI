import time
from fastapi import Request, HTTPException, status
from collections import defaultdict
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class RateLimiter:
    """Simple in-memory rate limiter for student projects"""
    
    def __init__(self, requests_per_minute: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
            window_seconds: Time window for rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(list)
        logger.info(f"Intialized rate limiter: {requests_per_minute} requests per {window_seconds} seconds")
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed based on rate limits
        
        Args:
            client_id: Identifier for the client (API key or IP)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        client_requests = self.request_counts[client_id]
        
        # Clean up old timestamps
        client_requests = [
            timestamp for timestamp in client_requests
            if current_time - timestamp < self.window_seconds
        ]
        self.request_counts[client_id] = client_requests
        
        # Check if limit exceeded
        if len(client_requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return False
        
        # Add current request
        self.request_counts[client_id].append(current_time)
        return True

# Create a rate limiter instance (60 requests per minute for students)
rate_limiter = RateLimiter(requests_per_minute=60)

async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting"""
    # Identify client (API key is best, fallback to IP)
    api_key = request.headers.get("X-API-Key", "")
    client_id = api_key if api_key else request.client.host
    
    if not rate_limiter.is_allowed(client_id):
        logger.warning(f"Request blocked for {client_id} - rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response