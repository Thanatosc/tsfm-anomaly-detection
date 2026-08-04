"""Split the persisted score archive into upload-sized parts for Zenodo.

The browser upload of a single 234 MB file is unreliable over a long-haul link,
and `.npz` is already compressed, so a single zip buys nothing. Each part is a
self-contained zip holding `scores/*.npz`, so extracting every part into
`results/` reconstitutes `results/scores/` regardless of the order they arrive in.

Files are assigned to parts in sorted order, greedily, so the split is
deterministic: the same input always produces the same parts.

Run: .venv/Scripts/python.exe make_score_parts.py
Out: artifacts/zenodo/scores_partNN.zip, artifacts/zenodo/SHA256SUMS.txt
"""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "scores"
OUT = ROOT / "artifacts" / "zenodo"
TARGET = 25 * 1024 * 1024  # bytes per part, before zip overhead


def parts(files):
    """Greedily pack sorted files into groups of at most TARGET bytes."""
    group, size = [], 0
    for f in files:
        n = f.stat().st_size
        if group and size + n > TARGET:
            yield group
            group, size = [], 0
        group.append(f)
        size += n
    if group:
        yield group


def main():
    files = sorted(SRC.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz under {SRC}")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("scores_part*.zip"):
        old.unlink()

    groups = list(parts(files))
    width = max(2, len(str(len(groups))))
    lines, total = [], 0

    for i, group in enumerate(groups, start=1):
        path = OUT / f"scores_part{i:0{width}d}.zip"
        # ZIP_STORED: the .npz members are already deflated, so re-deflating
        # costs time and returns nothing.
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
            for f in group:
                z.write(f, f"scores/{f.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        total += size
        lines.append(f"{digest}  {path.name}")
        print(f"{path.name}  {len(group):4d} files  {size / 1048576:6.1f} MB")

    (OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(groups)} parts, {len(files)} files, {total / 1048576:.1f} MB total")
    print(f"checksums -> {OUT / 'SHA256SUMS.txt'}")


if __name__ == "__main__":
    main()
