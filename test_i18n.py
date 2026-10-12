import i18n


def test_languages_have_same_keys():
    ru = set(i18n.STRINGS["ru"])
    en = set(i18n.STRINGS["en"])
    assert ru == en


def test_tr_returns_language_specific_string():
    assert i18n.tr("ru", "choose_file") == "Выбрать файл"
    assert i18n.tr("en", "choose_file") == "Choose file"


def test_tr_falls_back_to_default_for_unknown_language():
    assert i18n.tr("de", "choose_file") == i18n.STRINGS[i18n.DEFAULT_LANG]["choose_file"]


def test_tr_returns_key_when_missing():
    assert i18n.tr("en", "no_such_key") == "no_such_key"
