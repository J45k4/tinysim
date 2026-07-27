"""Deterministic source identities for generated validation evidence."""

from hashlib import sha256
from pathlib import Path


def source_tree_sha256(path: str | Path) -> str:
    """Hash Python source under ``path`` with stable relative names."""
    root = Path(path)
    sources = [root] if root.is_file() else sorted(root.rglob("*.py"))
    digest = sha256()
    for source in sources:
        relative = source.name if root.is_file() else str(source.relative_to(root))
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()


def project_source_sha256(path: str | Path) -> str:
    """Hash release-owned source/config/docs, excluding generated artifacts."""
    root = Path(path)
    included_suffixes = {".py", ".md", ".toml", ".json", ".xml"}
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        "__pycache__",
        "artifacts",
        "tinygrad",
    }
    sources = sorted(
        source
        for source in root.rglob("*")
        if source.is_file()
        and source.suffix in included_suffixes
        and not excluded_parts.intersection(source.relative_to(root).parts)
    )
    digest = sha256()
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return digest.hexdigest()
