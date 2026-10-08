#!/usr/bin/env python3
"""Verify distributed bytes, phase coverage and relative Markdown file links."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "PUBLICATION_MANIFEST.json").read_text())
    failures = []
    links = 0
    if (ROOT / ".git").exists() and shutil.which("git"):
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        tracked = set(filter(None, tracked))
        if "PUBLICATION_MANIFEST.json" in tracked:
            expected = {entry["path"] for entry in manifest["files"]} | {"PUBLICATION_MANIFEST.json"}
            for path in sorted(tracked - expected):
                failures.append(f"Tracked file missing from manifest: {path}")
            for path in sorted(expected - tracked):
                failures.append(f"Manifest file missing from checkout index: {path}")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f'Missing file: {entry["path"]}')
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            failures.append(f'Changed bytes: {entry["path"]}')
        if len(data) >= 50 * 1024 * 1024:
            failures.append(f'Unexpected large file: {entry["path"]}')
        if path.suffix.lower() == ".md":
            # Code examples and inline code are not rendered Markdown links.
            content = re.sub(r"```.*?```", "", data.decode("utf-8"), flags=re.S)
            content = re.sub(r"`[^`]*`", "", content)
            for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", content):
                value = match.group(1).strip()
                value = value[1:value.index(">")] if value.startswith("<") and ">" in value else value.split(' "', 1)[0]
                if value.startswith("#") or urlsplit(value).scheme:
                    continue
                target = unquote(value.split("#", 1)[0].split("?", 1)[0])
                links += 1
                if not (path.parent / target).exists():
                    failures.append(f'Broken link: {entry["path"]} -> {target}')
    for i in range(1, 11):
        for name in ("SCIENTIFIC_STATUS.md", "REPRODUCE.md"):
            if not (ROOT / "phases" / f"phase{i}" / name).is_file():
                failures.append(f"Missing phase {i} {name}")
    print(json.dumps({"passed": not failures, "files": len(manifest["files"]), "relative_links": links, "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
