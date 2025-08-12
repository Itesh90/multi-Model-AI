import time
import logging
from typing import Any, Dict, Optional, Callable, TypeVar, cast
from functools import wraps
import threading

T = TypeVar('T')

logger = logging.getLogger(__name__)

class InMemoryCache:
    """Simple in-memory cache for student projects"""
    
    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
        logger.info(f"Intialized in-memory cache (TTL: {default_ttl}s)")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Value if found and not expired, None otherwise
        """
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                return None
            
            # Check if expired
            if time.time() > self.expiry[key]:
                # Remove expired item
                del self.cache[key]
                del self.expiry[key]
                return None
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (optional, overrides default)
        """
        with self.lock:
            self.cache[key] = value
            self.expiry[key] = time.time() + (ttl or self.default_ttl)
            logger.debug(f"Cached item: {key} (TTL: {ttl or self.default_ttl}s)")
    
    def delete(self, key: str):
        """
        Delete item from cache
        
        Args:
            key: Cache key to delete
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.expiry[key]
                logger.debug(f"Removed from cache: {key}")
    
    def clear(self):
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            self.expiry.clear()
            logger.info("Cache cleared")

# Create a cache instance (5-minute TTL for students)
cache = InMemoryCache(default_ttl=300)

def cache_result(key_prefix: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results
    
    Args:
        key_prefix: Prefix for cache key
        ttl: Time-to-live in seconds (optional)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Create cache key from function name and arguments
            cache_key = f"{key_prefix}:{func.__name__}"
            if args:
                cache_key += f":{str(args)}"
            if kwargs:
                cache_key += f":{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cast(T, result)
            
            # If not in cache, call function and store result
            logger.debug(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return cast(T, result)
        
        return wrapper
    return decorator

# Test the cache
if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test basic caching
    print("\nTesting basic cache operations...")
    cache.set("test_key", "test_value", ttl=10)
    print(f"Get test_key: {cache.get('test_key')}")
    
    # Test decorator
    print("\nTesting cache decorator...")
    
    @cache_result("test", ttl=5)
    def expensive_operation(x: int) -> int:
        print(f"Performing expensive operation for {x}...")
        time.sleep(1)  # Simulate expensive operation
        return x * 2
    
    print("First call:")
    print(expensive_operation(5))
    
    print("\nSecond call (should be cached):")
    print(expensive_operation(5))
    
    print("\nWaiting for cache to expire...")
    time.sleep(6)
    
    print("\nThird call (cache expired):")
    print(expensive_operation(5))