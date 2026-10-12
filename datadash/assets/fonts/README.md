# Bundled fonts

`../fonts.css` declares these for the bundled themes. They are cut down from
the Nerd Fonts v3.4.0 release files (each directory's `README.md` and licence
are the upstream ones), so a page fetches tens of kilobytes rather than
megabytes:

| File | Made from |
| --- | --- |
| `DroidSansMono/DroidSansMNerdFontPropo-*.woff2` | `DroidSansMNerdFontPropo-Regular.otf` |
| `DroidSansMono/DroidSansMNerdFontMono-*.woff2` | `DroidSansMNerdFontMono-Regular.otf` |
| `ProFont/ProFontIIxNerdFont-*.woff2` | `ProFontIIxNerdFont-Regular.ttf` |

Each font is split in two with fonttools' `pyftsubset`, keeping every glyph
outside the Nerd Font icon ranges in `-Regular.woff2` and only the Material
Design icons (U+F0001-F1AF0, the set the themes' tab icons come from) in
`-Icons.woff2`. The other Nerd Font icon sets are dropped. ProFont's text file
also leaves out the characters its `unicode-range` in `fonts.css` excludes:

```bash
COMMON="--layout-features=* --name-IDs=* --name-languages=* --name-legacy --notdef-outline --legacy-kern --flavor=woff2"
uvx --from 'fonttools[woff]' pyftsubset DroidSansMNerdFontPropo-Regular.otf --unicodes="U+0000-DFFF,U+F900-FFFF" $COMMON --output-file=DroidSansMNerdFontPropo-Regular.woff2
uvx --from 'fonttools[woff]' pyftsubset DroidSansMNerdFontPropo-Regular.otf --unicodes="U+F0001-F1AF0" $COMMON --output-file=DroidSansMNerdFontPropo-Icons.woff2
```

and likewise for the Mono variant and for ProFont, whose text file takes the
`unicode-range` of its first face in `fonts.css` as `--unicodes`.
