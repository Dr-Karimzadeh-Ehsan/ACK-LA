"""Official-source acquisition, integrity audits, alignment and train-only preprocessing."""

from __future__ import annotations

import html
import json
import re
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from uclada.common import digest, file_hash, write_json
from uclada.env import TraceEpisode

NIST_BASE = "https://data.nist.gov/od/ds/ark:/88434/mds2-2553/"
CATALOG = (
    "https://catalog.data.gov/dataset/distributed-sensing-of-radio-spectrum-occupancy-amid-covid-19-18abf"
)
ZENODO = "https://zenodo.org/api/records/7521246"
CANONICAL_COLUMNS = [
    "trace_id",
    "dataset_id",
    "site_id",
    "session_id",
    "sensor_id",
    "timestamp_utc",
    "timestamp_quality",
    "freq_center_hz",
    "band_width_hz",
    "raw_power_value",
    "raw_power_unit",
    "calibrated_power_dbm",
    "noise_floor_dbm",
    "occupancy_reference",
    "occupancy_confidence",
    "occupancy_threshold_dbm",
    "ground_truth_kind",
    "observed",
    "timestamp_skew_ms",
    "source_file",
    "source_sha256",
    "preprocess_config_sha256",
    "split_id",
    "license_id",
]


def fetch(url: str, target: Path, timeout: int = 25) -> dict[str, Any]:
    """Save raw response and request evidence, including failure evidence."""
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    record: dict[str, Any] = {"url": url, "target": str(target)}
    try:
        response = requests.get(url, timeout=(10, timeout))
        record.update(status=response.status_code, headers=dict(response.headers), final_url=response.url)
        target.write_bytes(response.content)
        record.update(bytes=len(response.content), sha256=file_hash(target))
        response.raise_for_status()
        record["success"] = True
    except requests.RequestException as exc:
        record.update(success=False, error=str(exc))
    record["seconds"] = time.perf_counter() - started
    write_json(target.with_suffix(target.suffix + ".request.json"), record)
    return record


def audit_nist(root: Path) -> dict[str, Any]:
    raw = root / "data/raw/audit"
    urls = [
        (CATALOG, raw / "nist_catalog.html"),
        (NIST_BASE + "README.txt", raw / "nist_readme.txt"),
        (NIST_BASE + "README.txt.sha256", raw / "nist_readme.sha256"),
    ]
    with ThreadPoolExecutor(3) as pool:
        requests_log = list(pool.map(lambda x: fetch(*x), urls))
    catalog = raw / "nist_catalog.html"
    resources = []
    if catalog.exists():
        text = html.unescape(catalog.read_text(errors="replace"))
        urls_found = sorted(set(re.findall(r'https://data\.nist\.gov/od/ds/[^\s"<>\\]+', text)))
        for url in urls_found:
            resources.append(
                {
                    "dataset": "nist",
                    "source_url": url,
                    "resource_type": "time_series" if "time%20series" in url else "other",
                    "declared_size": None,
                    "downloaded": False,
                }
            )
    manifest = pd.DataFrame(
        resources, columns=["dataset", "source_url", "resource_type", "declared_size", "downloaded"]
    )
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(root / "data/manifests/nist_resources.parquet", index=False)
    manifest.to_csv(root / "data/manifests/nist_resources.csv", index=False)
    readme_ok = requests_log[1]["success"]
    report = {
        "dataset": "nist",
        "requests": requests_log,
        "catalog_resources": len(resources),
        "source_verified": bool(resources),
        "raw_time_series_received": False,
        "license_checked": False,
        "number_synchronous_channels": 0,
        "decision": "BLOCKED_RAW_ACCESS" if not readme_ok else "NEEDS_RAW_ALIGNMENT_AUDIT",
        "limitations": "Catalog is not proof of downloaded time-series content, units or simultaneity.",
    }
    write_json(root / "reports/dataset/nist_audit.json", report)
    update_eligibility(root)
    return report


def download_nist(root: Path, site: str = "00") -> dict[str, Any]:
    raw = root / "data/raw/nist"
    url = NIST_BASE + f"time%20series%20denoised/{site}.tar.gz"
    # Bounded prototype request: probe first, never create an invented substitute.
    try:
        response = requests.head(url, timeout=(10, 20), allow_redirects=True)
        response.raise_for_status()
        size = int(response.headers.get("Content-Length", 0))
    except requests.RequestException as exc:
        result = {"success": False, "url": url, "error": str(exc), "decision": "BLOCKED_RAW_ACCESS"}
        write_json(root / "reports/dataset/nist_download.json", result)
        return result
    if size > shutil.disk_usage(root).free // 2:
        raise RuntimeError("NIST archive exceeds half available storage; choose a smaller verified source")
    path = raw / f"denoised-{site}.tar.gz"
    result = stream_download(url, path, expected_md5=None)
    if result["success"]:
        checksum_path = raw / f"denoised-{site}.sha256"
        check = fetch(url + ".sha256", checksum_path)
        if check["success"]:
            expected = checksum_path.read_text().split()[0]
            result["official_sha256_matches"] = expected == result["sha256"]
            if not result["official_sha256_matches"]:
                raise RuntimeError("NIST official checksum mismatch")
    write_json(root / "reports/dataset/nist_download.json", result)
    return result


def stream_download(url: str, path: Path, expected_md5: str | None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if path.exists() and (not expected_md5 or file_hash(path, "md5") == expected_md5):
        return {
            "success": True,
            "cached": True,
            "url": url,
            "path": str(path),
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
    start = time.perf_counter()
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        with partial.open("ab" if append else "wb") as target:
            for chunk in response.iter_content(2**20):
                target.write(chunk)
    md5 = file_hash(partial, "md5")
    if expected_md5 and md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {partial}; retained for inspection")
    partial.rename(path)
    return {
        "success": True,
        "cached": False,
        "url": url,
        "path": str(path),
        "md5": md5,
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
        "seconds": time.perf_counter() - start,
    }


def audit_electrosense(root: Path, download: bool = False) -> dict[str, Any]:
    raw = root / "data/raw/audit"
    requests_log = []
    for name, url in [
        ("zenodo_metadata.txt", ZENODO),
        ("electrosense_readme.txt", "https://zenodo.org/records/7521246/files/README.md?download=1"),
        ("electrosense_license.txt", "https://zenodo.org/records/7521246/files/DATASET%20LICENCE?download=1"),
    ]:
        requests_log.append(fetch(url, raw / name))
    metadata = json.loads((raw / "zenodo_metadata.txt").read_text())
    archive = next(f for f in metadata["files"] if f["key"].endswith("tar.gz"))
    path = root / "data/raw/electrosense/spectrum_bands.tar.gz"
    if download:
        if shutil.disk_usage(root).free < archive["size"] * 3:
            raise RuntimeError("Insufficient storage for archive integrity audit")
        receipt = stream_download(archive["links"]["self"], path, archive["checksum"].split(":")[1])
        write_json(root / "reports/dataset/electrosense_download.json", receipt)
    report: dict[str, Any] = {
        "dataset": "electrosense",
        "source_verified": requests_log[0]["success"],
        "license_checked": requests_log[2]["success"],
        "requests": requests_log,
        "archive_expected_bytes": archive["size"],
        "archive_official_md5": archive["checksum"],
        "download_success": path.exists(),
        "sampling": "asynchronous_sweep",
        "number_synchronous_channels": 0,
        "timestamps_exist": None,
        "decision": "NO_GO_TIME_AWARE_MULTICHANNEL_UNTIL_METADATA_PROVES_ALIGNMENT",
        "limitations": "Sequential 2-MHz sweeps; published arrays target technology classification.",
    }
    if path.exists():
        report.update(inspect_electrosense(path, root))
    write_json(root / "reports/dataset/electrosense_audit.json", report)
    update_eligibility(root)
    return report


def inspect_electrosense(path: Path, root: Path) -> dict[str, Any]:
    """Read archive member metadata and NumPy headers without bulk extraction or pickle."""
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            row: dict[str, Any] = {"name": member.name, "bytes": member.size, "shape": None, "dtype": None}
            if member.name.endswith(".npy"):
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                magic = np.lib.format.read_magic(stream)
                if magic == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                row.update(
                    shape=str(shape),
                    dtype=str(dtype),
                    fortran=fortran,
                    has_named_timestamp_field=bool(
                        dtype.names and any("time" in n.lower() for n in dtype.names)
                    ),
                )
                if len(summaries) < 12 and not dtype.hasobject and member.size < 32 * 2**20:
                    data = stream.read()
                    values = np.frombuffer(data, dtype=dtype)
                    if np.issubdtype(dtype, np.number):
                        finite = values[np.isfinite(values)]
                        summaries.append(
                            {
                                "name": member.name,
                                "shape": list(shape),
                                "dtype": str(dtype),
                                "finite_fraction": float(len(finite) / max(1, len(values))),
                                "min": float(finite.min()) if len(finite) else None,
                                "median": float(np.median(finite)) if len(finite) else None,
                                "max": float(finite.max()) if len(finite) else None,
                            }
                        )
            rows.append(row)
    inventory = pd.DataFrame(rows)
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(root / "data/manifests/electrosense_members.parquet", index=False)
    inventory.to_csv(root / "data/manifests/electrosense_members.csv", index=False)
    return {
        "archive_sha256": file_hash(path),
        "archive_md5": file_hash(path, "md5"),
        "archive_members_checked": len(rows),
        "sample_array_statistics": summaries,
        "timestamp_fields_found": bool(
            inventory.get("has_named_timestamp_field", pd.Series(dtype=bool)).fillna(False).any()
        ),
        "timestamp_note": "Unnamed numeric matrix axes alone do not establish temporal cadence or chronology.",
    }


def update_eligibility(root: Path) -> None:
    rows = []
    for name in ["nist", "electrosense", "optional"]:
        path = root / f"reports/dataset/{name}_audit.json"
        x = json.loads(path.read_text()) if path.exists() else {}
        rows.append(
            {
                "dataset": name,
                "source_verified": x.get("source_verified", False),
                "download_success": x.get("download_success", x.get("raw_time_series_received", False)),
                "license_checked": x.get("license_checked", False),
                "files_checked": x.get("archive_members_checked", 0),
                "units_known": False,
                "timestamps_exist": x.get("timestamps_exist"),
                "time_monotonic": None,
                "sample_cadence_known": False,
                "number_synchronous_channels": x.get("number_synchronous_channels", 0),
                "per_site_overlap_seconds": None,
                "gap_fraction": None,
                "nontrivial_state_changes": None,
                "noise_properties": "NOT ESTABLISHED",
                "band_license_context": "See source; no per-observation occupant identity established",
                "trace_replay_eligible": False,
                "pu_label_supported": False,
                "limitations": x.get("limitations", "Raw accessibility and alignment not established"),
                "decision": x.get("decision", "NOT_ELIGIBLE_WITHOUT_RAW_AUDIT"),
            }
        )
    (root / "data/manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "data/manifests/data_eligibility_matrix.csv", index=False)


def chronological_split(length: int, embargo: int = 20) -> dict[str, np.ndarray]:
    a, b = int(length * 0.6), int(length * 0.8)
    if a <= embargo or b - a <= 2 * embargo or length - b <= embargo:
        raise ValueError("Insufficient observations for embargoed chronological splits")
    return {
        "train": np.arange(0, a - embargo),
        "validation": np.arange(a + embargo, b - embargo),
        "test": np.arange(b + embargo, length),
    }


def fit_threshold(train_power_dbm: np.ndarray, margin_db: float = 6) -> dict[str, float]:
    values = train_power_dbm[np.isfinite(train_power_dbm)]
    if not len(values):
        raise ValueError("No finite training calibration measurements")
    floor = float(np.quantile(values, 0.1))
    return {"noise_floor_dbm": floor, "threshold_dbm": floor + margin_db, "margin_db": margin_db}


def align_channels(
    frame: pd.DataFrame, slot_seconds: float, max_skew_fraction: float = 0.25, max_gap_fraction: float = 0.05
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"timestamp_utc", "freq_center_hz", "occupancy_reference", "site_id", "session_id"}
    if not required.issubset(frame):
        raise ValueError(f"Missing canonical columns: {required - set(frame)}")
    if frame[["site_id", "session_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Align one site/session at a time")
    f = frame.copy()
    times = (
        pd.to_datetime(f["timestamp_utc"], utc=True, format="ISO8601")
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
        / 1e9
    )
    anchor = float(np.min(times))
    bins = np.round((times - anchor) / slot_seconds).astype(np.int64)
    skew = np.abs(times - (anchor + bins * slot_seconds))
    if np.any(skew > slot_seconds * max_skew_fraction + 1e-6):
        raise ValueError("Asynchronous channels exceed prespecified timestamp skew")
    f["slot"] = bins
    if f.duplicated(["slot", "freq_center_hz"]).any():
        raise ValueError("Duplicate channel-slot samples require declared aggregation before alignment")
    pivot = f.pivot(index="slot", columns="freq_center_hz", values="occupancy_reference")
    pivot = pivot.reindex(np.arange(pivot.index.min(), pivot.index.max() + 1))
    missing = float(pivot.isna().to_numpy().mean())
    if missing > max_gap_fraction:
        raise ValueError("Missingness exceeds eligibility threshold; no forward filling permitted")
    return pivot, {
        "max_skew_seconds": float(skew.max()),
        "gap_fraction": missing,
        "channels": pivot.shape[1],
        "complete_slots": int(pivot.notna().all(axis=1).sum()),
        "overlap_seconds": int(pivot.notna().all(axis=1).sum()) * slot_seconds,
    }


def load_audited_trace(path: str | Path, split: str) -> TraceEpisode:
    path = Path(path)
    sidecar = path.with_suffix(".metadata.json")
    if not path.exists() or not sidecar.exists():
        raise FileNotFoundError("A real suite requires canonical Parquet plus an audited metadata sidecar")
    metadata = json.loads(sidecar.read_text())
    if not metadata.get("trace_replay_eligible") or metadata.get("synthetic", True):
        raise ValueError("Trace is not approved for real-data replay")
    frame = pd.read_parquet(path)
    frame = frame[frame["split_id"] == split]
    pivot, alignment = align_channels(frame, metadata["slot_seconds"])
    if pivot.isna().any().any():
        raise ValueError("Choose contiguous complete episodes; do not concatenate across missing slots")
    metadata["alignment"] = alignment
    return TraceEpisode(pivot.to_numpy(), f"{path.stem}-{split}", "threshold_measurement_proxy", metadata)


def prepare_canonical(source: Path, root: Path, slot_seconds: float = 0.01, margin_db: float = 6) -> Path:
    """Prepare explicit timestamped dBm CSV; refuses unknown-unit/untimestamped sources."""
    f = pd.read_csv(source)
    required = {
        "site_id",
        "session_id",
        "timestamp_utc",
        "freq_center_hz",
        "calibrated_power_dbm",
        "raw_power_unit",
        "band_width_hz",
        "license_id",
    }
    if not required.issubset(f) or set(f["raw_power_unit"].dropna()) != {"dBm"}:
        raise ValueError("Verified timestamped dBm source schema and license_id required")
    if f[["site_id", "session_id"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Prepare independent site/sessions separately")
    f["timestamp_utc"] = pd.to_datetime(f["timestamp_utc"], utc=True, format="ISO8601")
    f = f.sort_values(["timestamp_utc", "freq_center_hz"])
    timestamps = sorted(f["timestamp_utc"].unique())
    splits = chronological_split(len(timestamps), embargo=20)
    split_by_time = {timestamps[int(i)]: name for name, indices in splits.items() for i in indices}
    f["split_id"] = f["timestamp_utc"].map(split_by_time)
    preprocessing = {"margin_db": margin_db, "fit_on": "train", "embargo_slots": 20}
    for _, group in f.groupby("freq_center_hz"):
        threshold = fit_threshold(
            group.loc[group["split_id"] == "train", "calibrated_power_dbm"].to_numpy(), margin_db
        )
        f.loc[group.index, "noise_floor_dbm"] = threshold["noise_floor_dbm"]
        f.loc[group.index, "occupancy_threshold_dbm"] = threshold["threshold_dbm"]
    f["occupancy_reference"] = (f["calibrated_power_dbm"] >= f["occupancy_threshold_dbm"]).astype(int)
    f["raw_power_value"] = f["calibrated_power_dbm"]
    f["dataset_id"], f["ground_truth_kind"] = "nist", "threshold_measurement_proxy"
    f["source_file"], f["source_sha256"] = str(source), file_hash(source)
    f["timestamp_quality"], f["observed"] = "source_utc", True
    f["preprocess_config_sha256"] = digest(preprocessing)
    f["trace_id"], f["sensor_id"] = source.stem, f["site_id"]
    f["timestamp_skew_ms"], f["occupancy_confidence"] = None, None
    pivot, audit = align_channels(f, slot_seconds)
    target = root / "data/processed" / f"{source.stem}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    f[CANONICAL_COLUMNS].to_parquet(target, index=False)
    write_json(
        target.with_suffix(".metadata.json"),
        {
            **audit,
            "synthetic": False,
            "trace_replay_eligible": not pivot.isna().any().any(),
            "slot_seconds": slot_seconds,
            "preprocessing": preprocessing,
            "raw_sha256": file_hash(source),
        },
    )
    return target
