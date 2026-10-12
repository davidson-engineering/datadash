"""The fonts assets/fonts.css declares ship with the package, and nothing else does."""

import re

import yaml

from datadash.dashboard.base import assets_path

FONTS_CSS = (assets_path / "fonts.css").read_text()
FONT_SUFFIXES = {".woff2", ".woff", ".ttf", ".otf"}
GENERIC_FAMILIES = {"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"}


def _sources() -> list[tuple[str, str]]:
    """Each @font-face src in fonts.css, as (url, format)."""
    return re.findall(r"url\('\./([^']+)'\)\s*format\('([^']+)'\)", FONTS_CSS)


def test_every_declared_font_file_ships():
    sources = _sources()
    assert sources
    for url, _ in sources:
        assert (assets_path / url).is_file(), url


def test_every_bundled_font_file_is_declared():
    declared = {url for url, _ in _sources()}
    bundled = {
        str(path.relative_to(assets_path))
        for path in (assets_path / "fonts").rglob("*")
        if path.suffix in FONT_SUFFIXES
    }
    assert bundled == declared


def test_each_font_file_is_declared_in_its_own_format():
    formats = {".woff2": "woff2", ".woff": "woff", ".ttf": "truetype", ".otf": "opentype"}
    for url, declared_format in _sources():
        assert declared_format == formats[url[url.rindex(".") :]], url


def test_the_default_themes_fonts_ship():
    config = yaml.safe_load((assets_path.parent / "themes" / "config" / "default.yaml").read_text())
    declared = set(re.findall(r"font-family:\s*'([^']+)'", FONTS_CSS))
    for stack in config["fonts"].values():
        for family in (name.strip().strip("'\"") for name in stack.split(",")):
            assert family in declared or family in GENERIC_FAMILIES, family
