"""Locust load test for the V1.5 endpoint.

Manual run only (not exercised in CI):

    pip install locust
    locust -f eval/locust_v15.py --host http://localhost:8080
"""

from __future__ import annotations

try:
    from locust import HttpUser, between, task
except ImportError:  # locust is an optional, manual-only dep
    HttpUser = object  # type: ignore[assignment,misc]

    def task(fn):  # type: ignore[no-redef]
        return fn

    def between(_a, _b):  # type: ignore[no-redef]
        return None


class V15User(HttpUser):  # type: ignore[misc]
    wait_time = between(1, 3)

    @task
    def route_v15(self):
        self.client.post("/route/v15", json={"prompt": "What is 2 + 2?"})
