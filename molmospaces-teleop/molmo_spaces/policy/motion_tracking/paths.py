"""Repository-local paths used by the bundled tracking runtime."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "assets"
REAL_G1_ROOT = REPO_ROOT


def resolve_asset_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path
