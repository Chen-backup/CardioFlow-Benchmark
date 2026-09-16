"""Download public sources with per-file checksums and pinned split manifests.

Sources are downloaded as individual files, so archive extraction is not needed.
Existing files are reused only after their expected checksum is verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

from .constants import HEARTLANG_COMMIT, HEARTLANG_FILES, LUDB_COMMIT, LUDB_FILES


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(root: Path, relative: str) -> Path:
    """Reject traversal, Windows drive paths, and links escaping the root."""
    path = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or path.is_absolute():
        raise ValueError(f"Unsafe relative path: {relative!r}")
    if any(part in {".", ".."} for part in relative.split("/")):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    target = root.joinpath(*path.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Destination escapes download root: {relative!r}")
    return target


def fetch(url: str, destination: Path, expected_sha256: str | None = None) -> str:
    """Write an HTTPS response atomically after checksum verification."""
    if not url.startswith("https://"):
        raise ValueError("Public downloads require HTTPS")
    if destination.is_file() and expected_sha256 and sha256(destination) == expected_sha256:
        return expected_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as out:
            temporary = Path(out.name)
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if not response.url.startswith("https://"):
                    raise ValueError("Download redirected to a non-HTTPS URL")
                for block in response.iter_content(1024 * 1024):
                    if block:
                        digest.update(block)
                        out.write(block)
            result = digest.hexdigest()
        if expected_sha256 and result != expected_sha256:
            raise ValueError(f"Checksum mismatch for {destination.name}")
        os.replace(temporary, destination)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_checksums(text: str) -> dict[str, str]:
    entries = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Malformed SHA256 index")
        safe_destination(Path.cwd(), name)
        if name in entries:
            raise ValueError(f"Duplicate checksum entry: {name}")
        entries[name] = digest.lower()
    return entries


def download(dataset: str, root: str | Path, workers: int = 4) -> Path:
    """Fetch one dataset into ``root/dataset`` and record all file identities."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if dataset not in {"ptbxl", "cpsc2018", "ludb"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    target = Path(root) / dataset
    target.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path, str]] = []
    provenance: dict = {"dataset": dataset, "files": {}}
    if dataset in {"ptbxl", "cpsc2018", "ludb"}:
        version = {"ptbxl": "ptb-xl/1.0.3", "cpsc2018": "challenge-2020/1.0.2", "ludb": "ludb/1.0.1"}[dataset]
        base = f"https://physionet.org/files/{version}/"
        index = target / "SHA256SUMS.txt"
        provenance["checksum_index_sha256"] = fetch(base + "SHA256SUMS.txt", index)
        entries = parse_checksums(index.read_text(encoding="utf-8"))
        for name, digest in entries.items():
            if dataset == "ptbxl":
                selected = name in {"ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt", "RECORDS"} or (
                    name.startswith("records100/") and name.endswith((".hea", ".dat"))
                )
            elif dataset == "cpsc2018":
                selected = bool(re.search(r"(?:^|/)A\d+\.(?:hea|mat)$", name)) or name == "LICENSE.txt"
            else:
                selected = name in {"RECORDS", "LICENSE.txt"} or bool(re.fullmatch(r"data/\d+\.(?:hea|dat)", name))
            if selected:
                jobs.append((base + quote(name, safe="/"), safe_destination(target, name), digest))
        if len(jobs) < 100:
            raise ValueError("Official checksum index does not contain the expected records")
        provenance["source"] = base
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "cpsc2018":
        base = f"https://raw.githubusercontent.com/PKUDigitalHealth/HeartLang/{HEARTLANG_COMMIT}/datasets/dataset_preprocess/CPSC2018/"
        manifests = HEARTLANG_FILES
    elif dataset == "ludb":
        base = f"https://huggingface.co/spaces/MedicalAI-DP/ECG_Delineation/resolve/{LUDB_COMMIT}/res/ludb/dataset/"
        manifests = LUDB_FILES
    else:
        manifests = {}
    for name, digest in manifests.items():
        jobs.append((base + name, safe_destination(target, "protocol/" + name), digest))

    def run(job: tuple[str, Path, str]) -> tuple[str, dict]:
        url, path, expected = job
        digest = fetch(url, path, expected)
        return path.relative_to(target).as_posix(), {"sha256": digest, "url": url}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (name, identity) in enumerate(pool.map(run, jobs), 1):
            provenance["files"][name] = identity
            if number % 100 == 0 or number == len(jobs):
                print(f"{dataset}: verified {number}/{len(jobs)} files", flush=True)
    (target / "download_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return target
