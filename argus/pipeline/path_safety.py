from pathlib import Path


def validate_safe_path_segment(value: str, field_name: str) -> str:
    if value in {"", ".", ".."} or Path(value).name != value or "\\" in value:
        raise ValueError(f"{field_name} must be a single path-safe segment")
    return value
