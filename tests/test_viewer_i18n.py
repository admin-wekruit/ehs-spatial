"""Viewer catalogs and display strings share one complete three-language contract."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web" / "src"


def test_catalogs_have_the_same_keys_and_placeholders():
    catalogs = {language: json.loads((SOURCE / "locales" / f"{language}.json").read_text()) for language in ("en", "zh", "nl")}
    for language, catalog in catalogs.items():
        assert catalog.keys() == catalogs["en"].keys(), f"{language}: missing {sorted(catalogs['en'].keys() - catalog.keys())}"
        for key, value in catalog.items():
            assert isinstance(value, str) and value.strip(), (language, key)
            assert sorted(re.findall(r"\{(\w+)\}", value)) == sorted(re.findall(r"\{(\w+)\}", catalogs["en"][key])), (language, key)
            if language != "zh" and key != "lang.zh":
                assert not re.search(r"[\u3400-\u9fff]", value), (language, key, value)


def test_source_display_strings_do_not_bypass_catalogs():
    for directory in (SOURCE, ROOT / "web" / "tests", ROOT / "web" / "checks"):
        for path in directory.rglob("*"):
            if path.suffix not in (".ts", ".tsx", ".mjs", ".css", ".html"):
                continue
            for line in path.read_text().splitlines():
                assert not re.search(r"[\u3400-\u9fff\uff00-\uffef]", line), (path.relative_to(ROOT), line)


def test_mounted_viewers_receive_language_changes():
    for filename, receiver in (("App.tsx", "runtime"), ("LiveReport.tsx", "viewer")):
        source = (SOURCE / filename).read_text()
        assert re.search(rf"useEffect\(\(\) => \{{\s*{receiver}\.current\?\.setLocale\(language\);?\s*\}}, \[language\]\)", source), filename


if __name__ == "__main__":
    test_catalogs_have_the_same_keys_and_placeholders()
    test_source_display_strings_do_not_bypass_catalogs()
    test_mounted_viewers_receive_language_changes()
    print("Viewer catalog, placeholder, display-copy and language-update checks passed")
