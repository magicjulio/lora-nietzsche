import re
from pathlib import Path

files = [
    "also_sprach.txt",
    "EccoHomo.txt",
    "geburt_der_tragödie.txt",
    "götzen.txt",
    "jenseits_von_gut_und_böse.txt",
    "Menschliches.txt",
    "wille_zur_macht.txt",
]

start_re = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
end_re   = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

def strip_pg(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    m = start_re.search(text)
    if m:
        text = text[m.end():]

    m = end_re.search(text)
    if m:
        text = text[:m.start()]

    text = text.lstrip("\n")

    lines = text.split("\n")

    # optional: offensichtliche Frontmatter direkt am Anfang weg
    junk_prefixes = {
        "cover",
        "inhaltsverzeichnis",
        "contents",
    }

    while lines and lines[0].strip().lower() in junk_prefixes:
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    text = "\n".join(lines)

    # 3+ Leerzeilen auf max 2 reduzieren
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

parts = []
for fn in files:
    raw = Path(fn).read_text(encoding="utf-8-sig")
    cleaned = strip_pg(raw)
    parts.append(cleaned)

out = "\n\n" + ("=" * 80) + "\n\n"
Path("nietzsche_clean.txt").write_text(out.join(parts), encoding="utf-8")

print("Wrote nietzsche_clean.txt")

