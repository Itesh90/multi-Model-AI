import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

app = FastAPI(
    title="Multi-Modal AI Platform Performance Dashboard",
    description="Real-time monitoring and performance visualization",
    version="1.0.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "student-api-key-123"
REFRESH_INTERVAL = 5  # seconds

# In-memory storage for historical data
historical_data = {
    "requests": [],
    "response_times": [],
    "error_rates": [],
    "memory_usage": []
}

@app.on_event("startup")
async def startup_event():
    """Start background monitoring task"""
    asyncio.create_task(monitor_system())

async def monitor_system():
    """Background task to monitor system performance"""
    global historical_data
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Get performance metrics
                headers = {"X-API-Key": API_KEY}
                response = await client.get(
                    f"{API_BASE_URL}/performance/dashboard",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Store historical data
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Request metrics
                    historical_data["requests"].append({
                        "timestamp": timestamp,
                        "value": data["request_metrics"]["total_requests"]
                    })
                    
                    # Response time metrics (take average of all endpoints)
                    avg_response_time = 0
                    count = 0
                    for endpoint, metrics in data["endpoint_performance"].items():
                        avg_response_time += metrics["average_response_time"]
                        count += 1
                    avg_response_time = avg_response_time / count if count > 0 else 0
                    
                    historical_data["response_times"].append({
                        "timestamp": timestamp,
                        "value": avg_response_time
                    })
                    
                    # Error rate
                    error_rate = float(data["request_metrics"]["error_rate"].rstrip('%'))
                    historical_data["error_rates"].append({
                        "timestamp": timestamp,
                        "value": error_rate
                    })
                    
                    # Memory usage
                    if "memory" in data:
                        memory_mb = data["memory"]["memory_mb"]
                        historical_data["memory_usage"].append({
                            "timestamp": timestamp,
                            "value": memory_mb
                        })
                
                # Keep only last 60 data points (5 minutes at 5s interval)
                for key in historical_data:
                    if len(historical_data[key]) > 60:
                        historical_data[key] = historical_data[key][-60:]
                
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
            
            await asyncio.sleep(REFRESH_INTERVAL)

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics for dashboard"""
    async with httpx.AsyncClient() as client:
        try:
            headers = {"X-API-Key": API_KEY}
            response = await client.get(
                f"{API_BASE_URL}/performance/dashboard",
                headers=headers
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/historical")
async def get_historical():
    """Get historical metrics for charts"""
    return historical_data

@app.get("/api/system-resources")
async def get_system_resources():
    """Get current system resource usage"""
    async with httpx.AsyncClient() as client:
        try:
            headers = {"X-API-Key": API_KEY}
            response = await client.get(
                f"{API_BASE_URL}/system/resources",
                headers=headers
            )
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "api_base_url": API_BASE_URL
    }

@app.get("/")
async def dashboard():
    """Serve the dashboard UI"""
    dashboard_path = Path(__file__).parent / "dashboard.html"
    return {"content": dashboard_path.read_text()}

if __name__ == "__main__":
    print("Starting Performance Dashboard on http://localhost:8001")
    print("Make sure your main application is running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8001)