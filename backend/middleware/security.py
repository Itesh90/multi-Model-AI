from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware

def add_security_headers(response: Response):
    """Add security headers to response"""
    # Basic security headers
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Remove server information
    response.headers["Server"] = "Multi-Modal AI Platform"
    
    return response

async def security_middleware(request: Request, call_next):
    """Security middleware to add headers and handle security concerns"""
    response = await call_next(request)
    
    # Add security headers
    add_security_headers(response)
    
    return response