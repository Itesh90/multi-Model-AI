import pytest
import asyncio
import time
import statistics
from httpx import AsyncClient
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

@pytest.mark.asyncio
async def test_api_response_time(client: AsyncClient, api_key: str):
    """Test API response time under normal load"""
    endpoint = "/text/sentiment"
    payload = {"text": "This is a test message"}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Warm-up requests
    for _ in range(5):
        await client.post(endpoint, json=payload, headers=headers)
    
    # Measure response times
    response_times = []
    for _ in range(50):  # 50 requests
        start = time.time()
        response = await client.post(endpoint, json=payload, headers=headers)
        end = time.time()
        
        assert response.status_code == 200
        response_times.append(end - start)
    
    # Calculate statistics
    avg_time = statistics.mean(response_times)
    p95 = np.percentile(response_times, 95)
    max_time = max(response_times)
    
    print(f"\nAPI Performance Metrics for {endpoint}:")
    print(f"Average response time: {avg_time:.4f}s")
    print(f"95th percentile: {p95:.4f}s")
    print(f"Max response time: {max_time:.4f}s")
    
    # Create visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(10, 6))
    plt.plot(response_times, 'b-')
    plt.axhline(y=p95, color='r', linestyle='--', label=f'95th percentile: {p95:.4f}s')
    plt.title(f'API Response Time - {endpoint}')
    plt.xlabel('Request Number')
    plt.ylabel('Response Time (s)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'api_response_time_{timestamp}.png')
    
    # Assert performance requirements
    assert avg_time < 1.0, f"Average response time {avg_time:.4f}s exceeds 1.0s threshold"
    assert p95 < 2.0, f"95th percentile {p95:.4f}s exceeds 2.0s threshold"

@pytest.mark.asyncio
async def test_concurrent_requests(client: AsyncClient, api_key: str):
    """Test API under concurrent load"""
    endpoint = "/text/sentiment"
    payload = {"text": "This is a test message"}
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Number of concurrent requests
    num_requests = 20
    
    # Create async tasks
    start_time = time.time()
    tasks = [
        client.post(endpoint, json=payload, headers=headers)
        for _ in range(num_requests)
    ]
    
    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks)
    end_time = time.time()
    
    # Process results
    response_times = [(end_time - start_time) / num_requests] * num_requests
    successful = sum(1 for r in responses if r.status_code == 200)
    failure_rate = (num_requests - successful) / num_requests * 100
    
    print(f"\nConcurrent Load Test Results ({num_requests} requests):")
    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print(f"Total time: {end_time - start_time:.4f}s")
    print(f"Requests per second: {num_requests / (end_time - start_time):.2f}")
    
    # Assert reliability requirements
    assert failure_rate < 5.0, f"Failure rate {failure_rate:.2f}% exceeds 5% threshold"
    assert successful == num_requests, "Not all requests were successful"

@pytest.mark.asyncio
async def test_memory_usage(client: AsyncClient, api_key: str, tmp_path):
    """Test memory usage during image processing"""
    # Create a larger test image
    from PIL import Image
    img_path = tmp_path / "large_test.jpg"
    img = Image.new('RGB', (1000, 1000), color='red')
    img.save(img_path, format='JPEG', quality=95)
    
    # Get initial memory usage
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Process the image
    with open(img_path, "rb") as f:
        response = await client.post(
            "/process-image",
            headers={"X-API-Key": api_key},
            files={"file": ("large_test.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    
    # Get final memory usage
    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"\nMemory Usage Test:")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    # Assert memory constraints
    assert memory_increase < 200, f"Memory increase {memory_increase:.2f} MB exceeds 200 MB threshold"