"""Точка входа консольной команды `ai-music-detector`."""

import sys
from pathlib import Path


def main() -> None:
    """Запустить графический интерфейс (app.py в корне проекта)."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import app
    except ImportError as exc:
        print(
            f"Не удалось загрузить GUI ({exc}). Запустите: uv run python app.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise SystemExit(app.main())
