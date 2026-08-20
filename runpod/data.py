"""Load the Nietzsche corpus from the original Gutenberg files, one document per work.

We read the 7 sources directly rather than nietzsche_clean.txt: that combined file was
written by fix.py joining on ("=" * 80), but the separators are not present in it, so it
offers no reliable way to recover book boundaries. Blocks must never straddle two works.

Line wrapping is left exactly as-is, by request.
"""

import re
from pathlib import Path

FILES = [
    "also_sprach.txt",
    "EccoHomo.txt",
    "geburt_der_tragödie.txt",
    "götzen.txt",
    "jenseits_von_gut_und_böse.txt",
    "Menschliches.txt",
    "wille_zur_macht.txt",
]

_START = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_END = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

# Transcriber front-matter: English derivation notice, or the German transcription note on
# wille_zur_macht. Both sit in the first paragraphs and are not Nietzsche.
_NOTICE = re.compile(
    r"\A(?:This text has been derived from HTML files.*?\n\s*\n"
    r"|Anmerkungen zur Transkription:.*?\n\s*\n(?:.*?\n\s*\n)??(?=\s*Der Wille zur Macht))",
    re.IGNORECASE | re.DOTALL,
)

_JUNK_LINES = {"cover", "inhaltsverzeichnis", "contents", "inhalt"}


def load_works(data_dir: Path) -> list[tuple[str, str]]:
    """Return [(name, text)] for each work, Gutenberg wrapper and notices removed."""
    works = []
    for fn in FILES:
        raw = Path(data_dir, fn).read_text(encoding="utf-8-sig")
        text = raw.replace("\r\n", "\n").replace("\r", "\n")

        if m := _START.search(text):
            text = text[m.end():]
        if m := _END.search(text):
            text = text[: m.start()]

        text = text.lstrip("\n")
        text = _NOTICE.sub("", text, count=1)

        lines = text.split("\n")
        while lines and lines[0].strip().lower() in _JUNK_LINES:
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        text = "\n".join(lines)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        works.append((fn.removesuffix(".txt"), text))
    return works


if __name__ == "__main__":
    for name, text in load_works(Path(__file__).resolve().parent.parent):
        print(f"{name:32} {len(text):>8} chars  | {text[:60]!r}")
