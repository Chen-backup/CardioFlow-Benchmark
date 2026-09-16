"""Deterministic, task-specific transforms with fixed record/subject splits."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import irfft, rfft
from scipy.signal import resample_poly

from .constants import (
    CACHE_NAMES, CLASSES, CPSC_COLUMNS, CPSC_REMOVED, HEARTLANG_FILES,
    LEADS, LUDB_FILES, LUDB_TEST, LUDB_VAL, PROTOCOLS, SPLITS,
)
from .download import safe_destination, sha256


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_manifests(root: Path, expected: dict[str, str]) -> dict[str, str]:
    result = {}
    for name, digest in expected.items():
        actual = sha256(root / "protocol" / name)
        if actual != digest:
            raise ValueError(f"Published split manifest checksum mismatch: {name}")
        result[name] = actual
    return result


def verify_download(root: Path) -> str | None:
    manifest_path = root / "download_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        if sha256(safe_destination(root, name)) != identity["sha256"]:
            raise ValueError(f"Downloaded source checksum mismatch: {name}")
    return sha256(manifest_path)


def read_record(record: Path) -> tuple[np.ndarray, int]:
    import wfdb
    signal, fields = wfdb.rdsamp(str(record))
    names = [name.lower() for name in fields["sig_name"]]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate lead names in {record.name}")
    order = [names.index(lead.lower()) for lead in LEADS]
    signal = signal[:, order].T.astype(np.float32)
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite signal in {record.name}")
    return signal, int(fields["fs"])


def center_length(signal: np.ndarray, length: int, *, reject_short: bool = False) -> np.ndarray:
    current = signal.shape[-1]
    if current >= length:
        start = (current - length) // 2
        return signal[..., start:start + length]
    if reject_short:
        raise ValueError(f"Record has {current} samples; at least {length} are required")
    left = (length - current) // 2
    return np.pad(signal, [(0, 0)] * (signal.ndim - 1) + [(left, length - current - left)])


def fixed_fft_downsample_5000_to1000(signal: np.ndarray) -> np.ndarray:
    """Reference float32 FFT resampling with normalization before the inverse.

    The spectral scaling order reproduces the reference SciPy 1.18.1 cache.
    Keeping this order explicit avoids version-dependent float16 rounding.
    """
    signal = np.asarray(signal)
    if signal.dtype != np.float32 or signal.ndim < 1 or signal.shape[-1] != 5000:
        raise ValueError("Fixed FFT downsampling requires float32 input of length 5000")
    spectrum = rfft(signal, axis=-1)[..., :501]
    spectrum[..., 500] *= 2.0
    return irfft(spectrum / 5.0, n=1000, axis=-1, overwrite_x=True).astype(np.float32)


def cpsc_transform(signal: np.ndarray, sample_rate: int = 500) -> np.ndarray:
    """Center 5 s, append 5 s, global [-3,3] scaling, FFT resample, float16."""
    if sample_rate != 500 or signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError("CPSC preprocessing requires a 12-lead 500-Hz signal")
    cropped = center_length(signal, 5 * sample_rate, reject_short=True)
    padded = np.pad(cropped, ((0, 0), (0, 5 * sample_rate)))
    low, high = float(padded.min()), float(padded.max())
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Cannot normalize a constant or nonfinite CPSC signal")
    normalized = 6.0 * (padded - low) / (high - low) - 3.0
    return fixed_fft_downsample_5000_to1000(normalized).astype(np.float16)


def exact_deduplicate(signals: np.ndarray, targets: np.ndarray, split_codes: np.ndarray) -> np.ndarray:
    """Keep all rows in the highest-priority split: test > val > train."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, signal in enumerate(signals):
        digest = hashlib.sha256()
        digest.update(signal.dtype.str.encode("ascii"))
        digest.update(np.asarray(signal.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(signal).tobytes())
        groups[digest.hexdigest()].append(index)
    dropped = []
    for indices in groups.values():
        if len(set(int(split_codes[index]) for index in indices)) < 2:
            continue
        if any(not np.array_equal(targets[index], targets[indices[0]]) for index in indices[1:]):
            raise ValueError("Cross-split duplicate waveform has conflicting labels")
        winner = max(int(split_codes[index]) for index in indices)
        dropped.extend(index for index in indices if split_codes[index] != winner)
    return np.asarray(sorted(dropped), dtype=np.int64)


def interval_target(row: dict[str, str], length: int = 5000) -> np.ndarray:
    """Convert inclusive published P/QRS/T intervals to background/P/QRS/T IDs."""
    target = np.zeros(length, dtype=np.uint8)
    for label, column in enumerate(("p_onoffs", "qrs_onoffs", "t_onoffs"), 1):
        intervals = ast.literal_eval(row[column])
        if not isinstance(intervals, (list, tuple)):
            raise ValueError(f"Invalid interval list in {column}")
        for onset, offset in intervals:
            onset, offset = int(onset), int(offset)
            if not 0 <= onset <= offset < length:
                raise ValueError(f"Invalid {column} interval {(onset, offset)}")
            if np.any(target[onset:offset + 1]):
                raise ValueError("Overlapping waveform annotations")
            target[onset:offset + 1] = label
    return target


def normalize_ludb_lead(lead: np.ndarray) -> np.ndarray:
    """Per-lead float32 statistics, followed by the native float16 cache step."""
    lead = np.asarray(lead, dtype=np.float32)
    return ((lead - lead.mean()) / max(float(lead.std()), 1e-6) * 0.1).astype(np.float16)


def ludb_transform(tensor: np.ndarray, lead_index: int) -> np.ndarray:
    """Standardize a raw 500-Hz lead and resample it into its anatomical slot."""
    tensor = np.asarray(tensor, dtype=np.float32)
    if tensor.shape == (5000, 12):
        tensor = tensor.T
    if tensor.shape != (12, 5000) or not np.isfinite(tensor).all():
        raise ValueError(f"Invalid native LUDB signal: {tensor.shape}")
    if not 0 <= lead_index < 12:
        raise ValueError("Invalid lead index")
    lead = normalize_ludb_lead(tensor[lead_index]).astype(np.float32)
    x = np.zeros((12, 1000), dtype=np.float32)
    x[lead_index] = resample_poly(lead, 1, 5, window=("kaiser", 5.0)).astype(np.float32)
    return x


def train_statistics(signals: np.ndarray, indices: np.ndarray) -> dict[str, float | str]:
    total = total_sq = 0.0
    count = 0
    for start in range(0, len(indices), 64):
        block = np.asarray(signals[indices[start:start + 64]], dtype=np.float64)
        total += float(block.sum())
        total_sq += float(np.square(block).sum())
        count += block.size
    if not count:
        raise ValueError("Empty training split")
    mean = total / count
    std = float(np.sqrt(max(total_sq / count - mean ** 2, 0.0)))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Degenerate training normalization")
    return {"name": "train_global", "mean": mean, "std": std}


def check_groups(records: list[dict], split_codes: np.ndarray) -> None:
    assigned: dict[str, int] = {}
    for row, split in zip(records, split_codes, strict=True):
        group = str(row["group_id"])
        if group in assigned and assigned[group] != int(split):
            raise ValueError(f"Group occurs in multiple splits: {group}")
        assigned[group] = int(split)


def validate_protocol_rows(dataset: str, records: list[dict], splits: np.ndarray) -> None:
    """Check fixed split membership before assigning a benchmark protocol ID."""
    check_groups(records, splits)
    if dataset == "ptbxl":
        if len(records) != 21388:
            raise ValueError("PTB-XL requires 21388 diagnostic records")
        if [int((splits == code).sum()) for code in range(3)] != [17084, 2146, 2158]:
            raise ValueError("PTB-XL split sizes differ from the official diagnostic folds")
        for row, code in zip(records, splits, strict=True):
            fold = int(row["strat_fold"])
            if not 1 <= fold <= 10 or int(code) != (0 if fold <= 8 else fold - 8):
                raise ValueError("PTB-XL split disagrees with official strat_fold")
    elif dataset == "cpsc2018":
        expected = set(range(6877)) - set(CPSC_REMOVED)
        if len(records) != len(expected) or {row["source_index"] for row in records} != expected:
            raise ValueError("CPSC source indices do not match clean-v2 membership")
        for row, code in zip(records, splits, strict=True):
            index = row["source_index"]
            expected_code = 0 if index < 4950 else (1 if index < 5501 else 2)
            if int(code) != expected_code:
                raise ValueError("CPSC split disagrees with the pinned source ordering")
    else:
        expected_keys = {(subject, lead) for subject in range(1, 201) for lead in range(12)}
        if len(records) != 2400 or {(int(row["subject_id"]), int(row["lead_index"])) for row in records} != expected_keys:
            raise ValueError("LUDB requires every subject/lead pair exactly once")
        for row, code in zip(records, splits, strict=True):
            subject = int(row["subject_id"])
            expected_code = 1 if subject in LUDB_VAL else (2 if subject in LUDB_TEST else 0)
            if int(code) != expected_code:
                raise ValueError("LUDB split disagrees with the published subject partition")


def export_cache(
    stage: Path, dataset: str, signals: np.ndarray, targets: np.ndarray,
    records: list[dict], split_codes: np.ndarray, source: dict,
) -> dict:
    """Write immutable role directories and a hash-addressed dataset manifest."""
    if signals.shape != (len(records), 12, 1000) or len(targets) != len(records):
        raise ValueError("Input array/metadata shape mismatch")
    expected = (len(records), 5000) if dataset == "ludb" else (len(records), len(CLASSES[dataset]))
    if targets.shape != expected:
        raise ValueError(f"Unexpected target shape: {targets.shape}, expected {expected}")
    check_groups(records, split_codes)
    normalization = train_statistics(signals, np.flatnonzero(split_codes == 0)) if dataset == "ptbxl" else {"name": "none"}
    meta = {
        "format_version": 1, "dataset": dataset, "protocol": PROTOCOLS[dataset],
        "task": "delineation" if dataset == "ludb" else "classification",
        "class_names": list(CLASSES[dataset]), "num_classes": len(CLASSES[dataset]),
        "sample_rate": 100, "input_shape": [12, 1000],
        "target_sample_rate": 500 if dataset == "ludb" else 100,
        "valid_length": 500 if dataset == "cpsc2018" else 1000,
        "normalization": normalization, "source": source,
        "software": {"numpy": np.__version__, "scipy": scipy.__version__},
        "splits": {},
    }
    for code, name in enumerate(SPLITS):
        indices = np.flatnonzero(split_codes == code)
        if not len(indices):
            raise ValueError(f"Missing {name} split")
        folder = stage / name
        folder.mkdir()
        output = np.lib.format.open_memmap(folder / "signals.npy", mode="w+", dtype=signals.dtype, shape=(len(indices), 12, 1000))
        for start in range(0, len(indices), 64):
            output[start:start + 64] = signals[indices[start:start + 64]]
        output.flush()
        del output
        np.save(folder / "targets.npy", np.asarray(targets[indices], dtype=np.uint8), allow_pickle=False)
        write_json(folder / "records.json", [records[index] for index in indices])
        meta["splits"][name] = {
            "records": len(indices),
            "groups": len({str(records[index]["group_id"]) for index in indices}),
            "files": {file: sha256(folder / file) for file in ("signals.npy", "targets.npy", "records.json")},
        }
    write_json(stage / "meta.json", meta)
    return meta


def _find_records(root: Path) -> dict[str, Path]:
    records = {}
    for header in root.rglob("*.hea"):
        if header.stem in records:
            raise ValueError(f"Ambiguous record ID: {header.stem}")
        records[header.stem] = header.with_suffix("")
    return records


def _prepare_ptbxl(root: Path, work: Path):
    statements = read_csv(root / "scp_statements.csv")
    diagnostic = {
        next(iter(row.values())): row["diagnostic_class"]
        for row in statements
        if row.get("diagnostic") in {"1", "1.0"} and row.get("diagnostic_class")
    }
    selected = []
    for row in read_csv(root / "ptbxl_database.csv"):
        labels = {diagnostic[key] for key in ast.literal_eval(row["scp_codes"]) if key in diagnostic}
        if labels:
            selected.append((row, [int(label in labels) for label in CLASSES["ptbxl"]]))
    if len(selected) != 21388:
        raise ValueError(f"Expected 21388 PTB-XL diagnostic records, found {len(selected)}")
    signals = np.lib.format.open_memmap(work / "_signals.npy", mode="w+", dtype=np.float16, shape=(len(selected), 12, 1000))
    targets, records, splits = [], [], []
    for index, (row, target) in enumerate(selected):
        signal, rate = read_record(root / row["filename_lr"])
        if rate != 100:
            raise ValueError("PTB-XL must use the native records100 release")
        signals[index] = center_length(signal, 1000)
        targets.append(target)
        fold = int(row["strat_fold"])
        if not 1 <= fold <= 10:
            raise ValueError("PTB-XL strat_fold must be in [1,10]")
        splits.append(0 if fold <= 8 else fold - 8)
        records.append({"record_id": row["ecg_id"], "group_id": "patient:" + str(int(float(row["patient_id"]))), "source_index": index, "strat_fold": fold})
    source = {"kind": "official_wfdb", "version": "1.0.3", "metadata_sha256": sha256(root / "ptbxl_database.csv"), "statements_sha256": sha256(root / "scp_statements.csv")}
    return signals, np.asarray(targets, dtype=np.uint8), records, np.asarray(splits, dtype=np.int8), source


def _prepare_cpsc(root: Path, work: Path):
    identities = verify_manifests(root, HEARTLANG_FILES)
    rows, splits = [], []
    for code, filename in enumerate(HEARTLANG_FILES):
        subset = read_csv(root / "protocol" / filename)
        rows.extend(subset)
        splits.extend([code] * len(subset))
    if [splits.count(code) for code in range(3)] != [4950, 551, 1376]:
        raise ValueError("Unexpected HeartLang split counts")
    available = _find_records(root)
    signals = np.lib.format.open_memmap(work / "_signals.npy", mode="w+", dtype=np.float16, shape=(len(rows), 12, 1000))
    targets, records = [], []
    for index, row in enumerate(rows):
        record_id = Path(row["filename"]).stem
        if record_id not in available:
            raise FileNotFoundError(f"Missing CPSC WFDB record: {record_id}")
        signal, rate = read_record(available[record_id])
        signals[index] = cpsc_transform(signal, rate)
        targets.append([int(float(row[column])) for column in CPSC_COLUMNS])
        records.append({"record_id": record_id, "group_id": "record:" + record_id, "source_index": index})
    targets = np.asarray(targets, dtype=np.uint8)
    splits = np.asarray(splits, dtype=np.int8)
    removed = exact_deduplicate(signals, targets, splits)
    if tuple(removed.tolist()) != CPSC_REMOVED:
        raise ValueError("Waveform duplicates do not reproduce the clean-v2 removal list; check source files and preprocessing")
    signals.flush()
    source = {"kind": "official_wfdb", "version": "challenge-2020/1.0.2", "fft_resampling": "real_fft_5000_to1000_spectral_scaling_before_inverse", "split_files": identities, "removed_source_indices": removed.tolist(), "pre_dedup_signals_sha256": sha256(work / "_signals.npy")}
    # A view of the source is exported through a compact temporary memmap.
    keep = np.setdiff1d(np.arange(len(rows)), removed)
    compact = np.lib.format.open_memmap(work / "_compact.npy", mode="w+", dtype=np.float16, shape=(len(keep), 12, 1000))
    for start in range(0, len(keep), 64):
        compact[start:start + 64] = signals[keep[start:start + 64]]
    del signals
    return compact, targets[keep], [records[index] for index in keep], splits[keep], source


def _prepare_ludb(root: Path, work: Path):
    identities = verify_manifests(root, LUDB_FILES)
    by_key, subject_split = {}, {}
    events = [0, 0, 0]
    for code, filename in enumerate(LUDB_FILES):
        for row in read_csv(root / "protocol" / filename):
            subject = int(row["subject_id"])
            lead_index = [name.lower() for name in LEADS].index(row["lead_type"].lower())
            key = (subject, lead_index)
            if key in by_key:
                raise ValueError(f"Duplicate LUDB subject/lead: {key}")
            if subject in subject_split and subject_split[subject] != code:
                raise ValueError("LUDB subject crosses data splits")
            by_key[key] = row
            subject_split[subject] = code
            for index, column in enumerate(("p_onoffs", "qrs_onoffs", "t_onoffs")):
                events[index] += len(ast.literal_eval(row[column]))
    if len(by_key) != 2400 or set(subject_split) != set(range(1, 201)):
        raise ValueError("Expected all 200 LUDB subjects and 12 leads")
    if tuple(sorted(subject for subject, code in subject_split.items() if code == 1)) != LUDB_VAL or tuple(sorted(subject for subject, code in subject_split.items() if code == 2)) != LUDB_TEST:
        raise ValueError("LUDB subject split does not match the released protocol")
    if events != [16796, 21946, 19655]:
        raise ValueError(f"Unexpected LUDB annotation counts: {events}")
    signals = np.lib.format.open_memmap(work / "_signals.npy", mode="w+", dtype=np.float32, shape=(2400, 12, 1000))
    targets = np.lib.format.open_memmap(work / "_targets.npy", mode="w+", dtype=np.uint8, shape=(2400, 5000))
    records, splits = [], []
    for subject in range(1, 201):
        tensor, rate = read_record(root / "data" / str(subject))
        if rate != 500:
            raise ValueError("LUDB must use native 500-Hz WFDB records")
        tensor = center_length(tensor, 5000)
        if subject == 1:
            native = np.stack([normalize_ludb_lead(lead) for lead in tensor])
            fixture_sha256 = hashlib.sha256(native.tobytes()).hexdigest()
            if fixture_sha256 != "6ce1a896ee65997562685da3880047518bda317b92db6ac165ba2d4b754c6d3d":
                raise ValueError("LUDB subject-1 native normalization does not match the verified float16 fixture")
        for lead_index in range(12):
            index = len(records)
            signals[index] = ludb_transform(tensor, lead_index)
            targets[index] = interval_target(by_key[(subject, lead_index)])
            records.append({"record_id": f"{subject}/{LEADS[lead_index]}", "group_id": f"subject:{subject}", "subject_id": subject, "lead_index": lead_index, "lead": LEADS[lead_index], "source_index": index})
            splits.append(subject_split[subject])
    source = {"kind": "official_wfdb", "version": "1.0.1", "native_normalization": "(lead-mean)/max(std,1e-6)*0.1", "subject_1_native_float16_sha256": fixture_sha256, "split_files": identities, "waveform_counts": dict(zip(CLASSES["ludb"][1:], events))}
    return signals, targets, records, np.asarray(splits, dtype=np.int8), source


def prepare(dataset: str, raw_root: str | Path, processed_root: str | Path) -> Path:
    """Prepare all three fixed splits; refuse to overwrite an existing dataset."""
    builders = {"ptbxl": _prepare_ptbxl, "cpsc2018": _prepare_cpsc, "ludb": _prepare_ludb}
    if dataset not in builders:
        raise ValueError(f"Unknown dataset: {dataset}")
    destination = Path(processed_root) / dataset
    if destination.exists():
        raise FileExistsError(f"Prepared dataset already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_root) / dataset
    download_identity = verify_download(raw)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=destination.parent) as temporary:
        work = Path(temporary)
        stage = work / "export"
        stage.mkdir()
        signals, targets, records, splits, source = builders[dataset](raw, work)
        validate_protocol_rows(dataset, records, splits)
        if download_identity is not None:
            source["download_manifest_sha256"] = download_identity
        export_cache(stage, dataset, signals, targets, records, splits, source)
        del signals, targets
        os.replace(stage, destination)
    return destination
