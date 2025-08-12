import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_api_key_required(client: AsyncClient):
    """Test that API key is required for protected endpoints"""
    endpoints = [
        "/text/sentiment",
        "/text/embedding",
        "/process-image",
        "/search/multi-modal",
        "/rag/generate"
    ]
    
    for endpoint in endpoints:
        response = await client.post(
            endpoint,
            json={"text": "Test message"}
        )
        assert response.status_code == 401, f"Endpoint {endpoint} should require API key"

@pytest.mark.asyncio
async def test_invalid_api_key(client: AsyncClient):
    """Test that invalid API key is rejected"""
    endpoints = [
        "/text/sentiment",
        "/text/embedding",
        "/process-image",
        "/search/multi-modal",
        "/rag/generate"
    ]
    
    for endpoint in endpoints:
        response = await client.post(
            endpoint,
            json={"text": "Test message"},
            headers={"X-API-Key": "invalid-key"}
        )
        assert response.status_code == 401, f"Endpoint {endpoint} should reject invalid API key"

@pytest.mark.asyncio
async def test_rate_limiting(client: AsyncClient, api_key: str):
    """Test rate limiting functionality"""
    endpoint = "/text/sentiment"
    payload = {"text": "This is a test message"}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Send requests until rate limit is hit
    responses = []
    for _ in range(70):  # Exceed the 60 requests/minute limit
        response = await client.post(endpoint, json=payload, headers=headers)
        responses.append(response)
    
    # Count successful vs rate limited responses
    successful = sum(1 for r in responses if r.status_code == 200)
    rate_limited = sum(1 for r in responses if r.status_code == 429)
    
    print(f"\nRate Limiting Test Results:")
    print(f"Successful requests: {successful}")
    print(f"Rate limited requests: {rate_limited}")
    
    # Verify rate limiting works
    assert rate_limited > 0, "Rate limiting did not trigger"
    assert rate_limited >= 5, "Not enough rate limited requests (expected at least 5)"

@pytest.mark.asyncio
async def test_input_validation(client: AsyncClient, api_key: str):
    """Test input validation and sanitization"""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Test XSS injection attempt
    xss_payload = {
        "text": "<script>alert('xss')</script>This is a safe message"
    }
    response = await client.post(
        "/text/sentiment",
        json=xss_payload,
        headers=headers
    )
    assert response.status_code == 200
    
    # Test very long input
    long_payload = {
        "text": "A" * 6000  # Exceeding our 5000 character limit
    }
    response = await client.post(
        "/text/sentiment",
        json=long_payload,
        headers=headers
    )
    assert response.status_code == 422  # Should fail validation
    
    # Test SQL injection attempt
    sql_payload = {
        "text": "'; DROP TABLE users; --"
    }
    response = await client.post(
        "/text/sentiment",
        json=sql_payload,
        headers=headers
    )
    assert response.status_code == 200  # Should not crash

@pytest.mark.asyncio
async def test_security_headers(client: AsyncClient):
    """Test security headers are present"""
    response = await client.get("/health")
    
    # Check for security headers
    assert "Content-Security-Policy" in response.headers
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert "X-XSS-Protection" in response.headers
    assert "Strict-Transport-Security" in response.headers
    
    # Verify header values
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"

@pytest.mark.asyncio
async def test_content_moderation(client: AsyncClient, api_key: str, tmp_path):
    """Test content moderation functionality"""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Test safe content
    safe_response = await client.post(
        "/moderation/check",
        json={"text": "This is a normal, safe message."},
        headers=headers
    )
    assert safe_response.status_code == 200
    safe_data = safe_response.json()
    assert safe_data["is_safe"] is True
    
    # Test unsafe content
    unsafe_response = await client.post(
        "/moderation/check",
        json={"text": "This message contains hate speech and explicit content."},
        headers=headers
    )
    assert unsafe_response.status_code == 200
    unsafe_data = unsafe_response.json()
    assert unsafe_data["is_safe"] is False
    assert "hate" in unsafe_data["categories"] or "sexual" in unsafe_data["categories"]
    
    # Create a test image
    from PIL import Image
    img_path = tmp_path / "unsafe.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path, format='JPEG')
    
    # Test image moderation (our implementation just checks description)
    with open(img_path, "rb") as f:
        image_response = await client.post(
            "/process-image",
            headers={"X-API-Key": api_key},
            files={"file": ("unsafe.jpg", f, "image/jpeg")}
        )
    
    assert image_response.status_code == 200
    # In our implementation, unsafe images are flagged but still processed