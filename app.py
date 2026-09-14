#!/usr/bin/env python3
"""QML-интерфейс для детекции AI-музыки через check_ai_music."""

import argparse
import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from PySide6.QtWidgets import QApplication, QFileDialog

from check_ai_music import detect
from record_and_check import record_and_detect

MIC_DURATION = 10

ROOT = Path(__file__).resolve().parent


class Worker(QObject):
    """Вся работа (детекция, запись) исполняется в отдельном потоке."""

    resultReady = Signal(str)

    @Slot(str)
    def process_file(self, path):
        try:
            result = detect(path)
        except Exception as exc:
            result = {"file": path, "p_ai": None, "verdict": "Uncertain", "error": str(exc)}
        self.resultReady.emit(json.dumps(result, ensure_ascii=False, indent=2))

    @Slot()
    def process_mic(self):
        try:
            result = record_and_detect(MIC_DURATION)
        except Exception as exc:
            result = {"file": "(mic)", "p_ai": None, "verdict": "Uncertain", "error": str(exc)}
        self.resultReady.emit(json.dumps(result, ensure_ascii=False, indent=2))


class DetectorBridge(QObject):
    """Мост QML <-> Python. Сигналы в QML идут только через resultChanged."""

    resultChanged = Signal(str)

    _run_file = Signal(str)
    _run_mic = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = Worker()
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._worker.resultReady.connect(self.resultChanged)
        self._run_file.connect(self._worker.process_file)
        self._run_mic.connect(self._worker.process_mic)
        self._thread.start()

    def shutdown(self):
        self._thread.quit()
        self._thread.wait()

    @Slot(str)
    def detectFile(self, path):
        if path:
            self._run_file.emit(path)

    @Slot()
    def detectMic(self):
        self._run_mic.emit()

    @Slot()
    def chooseFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Выберите аудиофайл",
            str(Path.home()),
            "Аудио (*.wav *.mp3 *.flac);;Все файлы (*)",
        )
        if path:
            self.detectFile(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="QML-интерфейс для детекции AI-музыки",
    )
    parser.add_argument("--smoke-test", action="store_true", help="инициализировать QApplication и выйти с кодом 0")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.smoke_test and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication(sys.argv)

    if args.smoke_test:
        return 0

    QQuickStyle.setStyle("Material")
    bridge = DetectorBridge(app)
    app.aboutToQuit.connect(bridge.shutdown)
    engine = QQmlApplicationEngine(app)
    engine.rootContext().setContextProperty("bridge", bridge)
    engine.load(QUrl.fromLocalFile(str(ROOT / "ui" / "main.qml")))
    if not engine.rootObjects():
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())