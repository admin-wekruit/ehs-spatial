"""Read the selected Modal or HTTP provider from the process environment."""
import os


def service_backend(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default).strip().lower() or default
