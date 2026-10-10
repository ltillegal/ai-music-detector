import os
from pathlib import Path

# Тесты используют относительные пути (модель, fixtures) — фиксируем cwd на корень проекта,
# чтобы `pytest` работал из любого каталога.
os.chdir(Path(__file__).resolve().parent)
