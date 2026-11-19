"""Entry point for the `ai-music-detector` console command."""

import sys
from pathlib import Path


def main() -> None:
    """Launch the GUI (app.py in the repository root)."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import app
    except ImportError as exc:
        print(
            f"Could not load the GUI ({exc}). Run: uv run python app.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise SystemExit(app.main())
