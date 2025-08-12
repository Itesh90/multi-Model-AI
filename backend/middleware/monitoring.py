import time
import logging
from typing import Callable
from fastapi import Request, Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RequestLogger:
    """Logs request metrics for monitoring and exposes aggregated metrics."""

    def __init__(self):
        self.total_requests = 0
        self.error_count = 0
        self.start_time = time.time()
        self.endpoint_metrics = {}
        logger.info("Initialized request monitoring")

    def log_request(self, request: Request, response: Response, process_time: float) -> None:
        """Log request details and maintain endpoint-level metrics."""
        self.total_requests += 1

        endpoint = request.url.path
        if endpoint not in self.endpoint_metrics:
            self.endpoint_metrics[endpoint] = {
                "count": 0,
                "total_time": 0.0,
                "error_count": 0,
            }

        metrics = self.endpoint_metrics[endpoint]
        metrics["count"] += 1
        metrics["total_time"] += process_time

        if response.status_code >= 400:
            metrics["error_count"] += 1
            self.error_count += 1
            log_level = logging.WARNING if response.status_code < 500 else logging.ERROR
        else:
            log_level = logging.INFO

        uptime = time.time() - self.start_time

        logger.log(
            log_level,
            f"{request.method} {endpoint} | Status: {response.status_code} | "
            f"Time: {process_time:.2f}s | Uptime: {uptime:.0f}s",
        )

    def get_metrics(self) -> dict:
        """Return aggregated monitoring metrics including per-endpoint averages."""
        uptime = time.time() - self.start_time
        overall_error_rate = (self.error_count / self.total_requests * 100) if self.total_requests > 0 else 0.0

        endpoint_averages = {}
        for endpoint, metrics in self.endpoint_metrics.items():
            avg_time = metrics["total_time"] / metrics["count"] if metrics["count"] > 0 else 0.0
            endpoint_error_rate = (metrics["error_count"] / metrics["count"] * 100) if metrics["count"] > 0 else 0.0
            endpoint_averages[endpoint] = {
                "request_count": metrics["count"],
                "average_response_time": round(avg_time, 3),
                "error_rate": round(endpoint_error_rate, 2),
                "total_time": round(metrics["total_time"], 3),
            }

        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": f"{overall_error_rate:.2f}%",
            "uptime_seconds": int(uptime),
            "status": "healthy" if overall_error_rate < 5.0 else "degraded",
            "endpoint_performance": endpoint_averages,
        }


# Create a monitoring instance
monitor = RequestLogger()


async def monitoring_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware that measures request time and records metrics."""
    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as exc:
        # If an exception occurs while processing the request, create a 500 response for logging
        process_time = time.time() - start_time
        # Create a minimal Response for logging purposes
        response = Response(status_code=500, content=b"Internal Server Error")
        monitor.log_request(request, response, process_time)
        logger.exception("Unhandled exception in request processing: %s", exc)
        raise

    process_time = time.time() - start_time
    monitor.log_request(request, response, process_time)

    # Optionally add monitoring headers to the response
    response.headers["X-Process-Time"] = f"{process_time:.2f}"

    return response