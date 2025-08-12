import asyncio
import time
import random
import logging
import aiohttp
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LoadTester:
    """Simple load testing tool for student projects"""
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        num_users: int = 10,
        ramp_up_time: int = 30,
        test_duration: int = 60
    ):
        """
        Initialize load tester
        
        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            num_users: Number of virtual users
            ramp_up_time: Time to ramp up to full user count (seconds)
            test_duration: Duration of the main test (seconds)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.num_users = num_users
        self.ramp_up_time = ramp_up_time
        self.test_duration = test_duration
        self.results = []
        logger.info(
            f"Intialized load tester: {num_users} users over {ramp_up_time}s, "
            f"testing for {test_duration}s"
        )
    
    async def send_request(self, session: aiohttp.ClientSession, endpoint: str, payload: Dict[str, Any]):
        """Send a single request and record performance"""
        start_time = time.time()
        try:
            headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
            async with session.post(f"{self.base_url}{endpoint}", json=payload, headers=headers) as response:
                response_time = time.time() - start_time
                status = response.status
                success = 200 <= status < 300
                
                # Record result
                self.results.append({
                    "endpoint": endpoint,
                    "response_time": response_time,
                    "status": status,
                    "success": success,
                    "timestamp": time.time()
                })
                
                return response_time, status, success
        except Exception as e:
            response_time = time.time() - start_time
            self.results.append({
                "endpoint": endpoint,
                "response_time": response_time,
                "status": 500,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            })
            return response_time, 500, False
    
    async def simulate_user(self, user_id: int, session: aiohttp.ClientSession):
        """Simulate a user making requests"""
        # Randomize user behavior
        endpoints = [
            ("/text/sentiment", {"text": "This is a test message"}),
            ("/text/embedding", {"text": "Another test message"}),
            ("/search/multi-modal", {"query": "test query"})
        ]
        
        start_time = time.time()
        while time.time() - start_time < self.test_duration:
            # Choose random endpoint
            endpoint, payload = random.choice(endpoints)
            
            # Add some random delay between requests
            await asyncio.sleep(random.uniform(0.1, 1.0))
            
            # Send request
            await self.send_request(session, endpoint, payload)
    
    async def run_test(self):
        """Run the load test"""
        logger.info("Starting load test...")
        
        # Create session
        connector = aiohttp.TCPConnector(limit_per_host=self.num_users)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Ramp up users gradually
            tasks = []
            for i in range(self.num_users):
                # Calculate delay to spread users over ramp-up period
                delay = (i / self.num_users) * self.ramp_up_time
                await asyncio.sleep(delay / self.num_users)
                
                task = asyncio.create_task(self.simulate_user(i, session))
                tasks.append(task)
                logger.debug(f"Started user {i+1}/{self.num_users}")
            
            # Wait for test duration
            await asyncio.sleep(self.test_duration)
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for cancellation to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Load test completed!")
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze and visualize test results"""
        if not self.results:
            logger.warning("No results to analyze")
            return {}
        
        # Calculate metrics
        total_requests = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failure_rate = (1 - successful / total_requests) * 100 if total_requests > 0 else 0
        
        response_times = [r["response_time"] for r in self.results]
        avg_response_time = sum(response_times) / len(response_times)
        p95 = np.percentile(response_times, 95) if response_times else 0
        
        status_codes = {}
        for r in self.results:
            status = r["status"]
            status_codes[status] = status_codes.get(status, 0) + 1
        
        # Create visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_visualization(timestamp)
        
        # Return analysis
        analysis = {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failure_rate": f"{failure_rate:.2f}%",
            "average_response_time": f"{avg_response_time:.3f}s",
            "p95_response_time": f"{p95:.3f}s",
            "status_codes": status_codes,
            "timestamp": timestamp
        }
        
        logger.info(f"Test results: {analysis}")
        return analysis
    
    def _create_visualization(self, timestamp: str):
        """Create visualizations of the test results"""
        if not self.results:
            return
        
        # Create response time over time plot
        plt.figure(figsize=(12, 8))
        
        # Response time over time
        plt.subplot(2, 1, 1)
        times = [r["timestamp"] - self.results[0]["timestamp"] for r in self.results]
        response_times = [r["response_time"] for r in self.results]
        statuses = [r["status"] for r in self.results]
        
        # Color by status
        colors = ['g' if 200 <= status < 300 else 'r' for status in statuses]
        
        plt.scatter(times, response_times, c=colors, alpha=0.5)
        plt.axhline(y=np.percentile(response_times, 95), color='r', linestyle='-', alpha=0.3, label='95th percentile')
        plt.title('Response Time Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Response Time (seconds)')
        plt.grid(True)
        plt.legend()
        
        # Status code distribution
        plt.subplot(2, 1, 2)
        status_counts = {}
        for r in self.results:
            status = r["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        plt.bar(status_counts.keys(), status_counts.values(), color=['g' if 200 <= s < 300 else 'r' for s in status_counts.keys()])
        plt.title('Status Code Distribution')
        plt.xlabel('Status Code')
        plt.ylabel('Count')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'load_test_results_{timestamp}.png')
        logger.info(f"Saved visualization to load_test_results_{timestamp}.png")

# Test the load tester
if __name__ == "__main__":
    # Configure for local testing
    BASE_URL = "http://localhost:8000"
    API_KEY = "student-api-key-123"
    
    # Create and run load test
    tester = LoadTester(
        base_url=BASE_URL,
        api_key=API_KEY,
        num_users=5,           # 5 virtual users for students
        ramp_up_time=10,       # Ramp up over 10 seconds
        test_duration=30       # Test for 30 seconds
    )
    
    # Run the test
    results = asyncio.run(tester.run_test())
    
    # Print summary
    print("\nLoad Test Summary:")
    print(f"Total requests: {results['total_requests']}")
    print(f"Successful: {results['successful_requests']}")
    print(f"Failure rate: {results['failure_rate']}")
    print(f"Average response time: {results['average_response_time']}")
    print(f"95th percentile response time: {results['p95_response_time']}")
    print(f"Status codes: {results['status_codes']}")