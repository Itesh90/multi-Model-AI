import pytest
from httpx import AsyncClient
from backend.main import app
from backend.database import init_db
import os

@pytest.fixture(autouse=True)
def setup_database():
    """Initialize database before tests"""
    init_db()
    yield
    # Cleanup can go here if needed

@pytest.fixture
async def client():
    """Create test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def api_key():
    """Provide API key for tests"""
    return "student-api-key-123"