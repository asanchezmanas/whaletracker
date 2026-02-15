import time
import threading
from typing import Dict

class RateLimiter:
    """
    Thread-safe global rate limiter for API calls.
    Ensures that multiple connectors sharing the same resource (e.g., SEC)
    don't exceed the combined request limit.
    """
    _instances: Dict[str, 'RateLimiter'] = {}
    _lock = threading.Lock()

    def __init__(self, name: str, requests_per_second: float):
        self.name = name
        self.delay = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    @classmethod
    def get_limiter(cls, name: str, requests_per_second: float = 10.0) -> 'RateLimiter':
        """Get or create a named rate limiter instance."""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, requests_per_second)
            return cls._instances[name]

    def wait(self):
        """Wait for the next available request slot."""
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_request_time = time.time()
