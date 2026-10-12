import check_ai_music


def test_detect_reports_progress_and_fields():
    events = []
    result = check_ai_music.detect(
        "fixtures/known_suno.wav",
        progress=lambda frac, stage: events.append((frac, stage)),
    )
    assert result["verdict"] == "AI"
    assert events[0][1] == "reading"
    assert events[-1] == (1.0, "done")
    assert all(0.0 <= f <= 1.0 for f, _ in events)
    assert result["reasons"]
    assert isinstance(result["warnings"], list)
    assert "artifactnet" in result["windows"]


def test_detect_degraded_without_sonics(monkeypatch):
    monkeypatch.setattr(check_ai_music.sonics_detector, "available", lambda: False)
    result = check_ai_music.detect("fixtures/known_human.wav")
    assert result["verdict"] == "Uncertain"
    assert "sonics_missing" in result["reasons"]
    assert "sonics_unavailable" in result["warnings"]


def test_warning_localized():
    ru = check_ai_music.i18n.tr("ru", "warn_short_audio")
    en = check_ai_music.i18n.tr("en", "warn_short_audio")
    assert ru and en and ru != en
