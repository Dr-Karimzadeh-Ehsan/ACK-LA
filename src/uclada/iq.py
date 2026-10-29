"""Simultaneous subband measurements from a single documented continuous IQ stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uclada.common import digest, file_hash, write_json
from uclada.data import chronological_split
from uclada.env import TraceEpisode

OFFSETS_HZ = np.array([-800000, -600000, -400000, -200000, 200000, 400000, 600000, 800000])


def subband_power(
    samples: np.ndarray,
    sample_rate_hz: float,
    fft_size: int = 1024,
    offsets_hz: np.ndarray = OFFSETS_HZ,
    bandwidth_hz: float = 180000,
    batch_slots: int = 512,
) -> np.ndarray:
    """Hann-windowed integrated PSD, same sample window for every subband.

    Output units: squared stored IQ units. No absolute power calibration is assumed.
    """
    if sample_rate_hz <= 0 or fft_size < 8 or bandwidth_hz <= 0:
        raise ValueError("Positive sample rate, FFT size and subband width required")
    edges = np.c_[offsets_hz - bandwidth_hz / 2, offsets_hz + bandwidth_hz / 2]
    if np.any(np.abs(edges) >= sample_rate_hz / 2):
        raise ValueError("Requested subband exceeds sampled Nyquist bandwidth")
    if any(max(a[0], b[0]) < min(a[1], b[1]) for i, a in enumerate(edges) for b in edges[i + 1 :]):
        raise ValueError("Allocation subbands must be disjoint")
    slots = len(samples) // fft_size
    frequencies = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate_hz))
    masks = [(frequencies >= low) & (frequencies < high) for low, high in edges]
    if any(not m.any() for m in masks):
        raise ValueError("Frequency resolution cannot support requested subbands")
    window = np.hanning(fft_size)
    normalization = sample_rate_hz * float(np.sum(window**2))
    delta_f = sample_rate_hz / fft_size
    output = np.empty((slots, len(offsets_hz)), dtype=np.float64)
    for start in range(0, slots, batch_slots):
        stop = min(slots, start + batch_slots)
        blocks = np.asarray(samples[start * fft_size : stop * fft_size]).reshape(-1, fft_size)
        transformed = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
        psd = np.abs(transformed) ** 2 / normalization
        output[start:stop] = np.stack([psd[:, m].sum(axis=1) * delta_f for m in masks], axis=1)
    return output


def extract_iq(path: Path, output: Path, discard_seconds: float = 1.0) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if file_hash(path) != receipt["prefix_sha256"]:
        raise ValueError("IQ prefix SHA256 differs from acquisition receipt")
    recording = receipt["metadata"]["recording"]
    fs = float(recording["Sample_Rate_Hz"])
    samples = np.memmap(path, mode="r", dtype="<c8")
    finite = bool(np.isfinite(samples.real).all() and np.isfinite(samples.imag).all())
    if not finite:
        raise ValueError("Nonfinite IQ samples; do not silently impute")
    discard = int(discard_seconds * fs)
    if len(samples) - discard < 1000 * 1024:
        raise ValueError("Insufficient retained IQ samples for chronological replay")
    powers = subband_power(samples[discard:], fs)
    db = 10 * np.log10(np.maximum(powers, 1e-30))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, db, allow_pickle=False)
    metadata = {
        "dataset_id": "iraq_rf",
        "source": "official measured over-the-air complex IQ",
        "recording": recording,
        "source_record": receipt["metadata"]["archive"]["record"],
        "source_archive_url": receipt["source_url"],
        "source_member": receipt["member"],
        "source_prefix_sha256": receipt["prefix_sha256"],
        "source_prefix_bytes": receipt["prefix_bytes"],
        "full_archive_md5_checked": receipt["archive_md5_checked"],
        "member_full_crc_checked": receipt["member_full_crc_checked"],
        "sample_rate_hz": fs,
        "fft_size": 1024,
        "slot_seconds": 1024 / fs,
        "discard_samples": discard,
        "discard_seconds": discard_seconds,
        "freq_centers_hz": (float(recording["Center_Frequency_MHz"]) * 1e6 + OFFSETS_HZ).tolist(),
        "bandwidth_hz": 180000,
        "channels": len(OFFSETS_HZ),
        "slots": len(db),
        "units": "dB relative to squared stored complex-IQ units; NOT calibrated dBm",
        "timestamp_quality": "relative stored sample clock; absolute start is source-estimated",
        "alignment": "identical FFT sample window for all nonoverlapping subbands",
        "timestamp_skew_seconds_by_construction": 0,
        "sample_drop_log_available": False,
        "overflow_assumption": "Stored IQ sequence treated as continuous at declared sample rate; no independent acquisition-overflow log available",
        "missing_retained_samples": 0,
        "finite_samples": finite,
        "fraction_exact_complex_zero": float(np.mean(samples == 0)),
        "max_abs_component": float(max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)))),
        "power_sha256": file_hash(output),
        "synthetic": False,
        "license_id": "CC-BY-4.0",
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def fit_occupancy(
    power_db: np.ndarray,
    train_indices: np.ndarray,
    margin_db: float = 6.0,
    method: str = "global_lower_quantile",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prespecified energy proxy calibration using training only, with explicit assumptions.

    A shared robust low-power level assumes at least one quiet frequency/time region.
    Constant signals on all subbands can invalidate that assumption. Report sensitivity.
    """
    if not len(train_indices) or not np.isfinite(power_db[train_indices]).all():
        raise ValueError("Finite training powers required")
    training = power_db[train_indices]
    if method == "global_lower_quantile":
        floor = np.full(power_db.shape[1], float(np.quantile(training, 0.1)))
    elif method == "per_channel_lower_quantile":
        floor = np.quantile(training, 0.1, axis=0)
    else:
        raise ValueError("Unknown threshold calibration method")
    thresholds = floor + margin_db
    labels = (power_db >= thresholds).astype(np.int8)
    calibration = {
        "fit_split": "train",
        "fit_indices_sha256": digest(train_indices),
        "margin_db": margin_db,
        "method": method,
        "noise_proxy_relative_db": floor.tolist(),
        "threshold_relative_db": thresholds.tolist(),
        "assumption": "A quiet region exists in training; this is a statistical noise proxy, not an independent RF noise calibration",
        "ground_truth_kind": "threshold_measurement_proxy",
        "training_busy_fraction": labels[train_indices].mean(axis=0).tolist(),
        "training_state_changes": np.sum(np.diff(labels[train_indices], axis=0) != 0, axis=0).tolist(),
    }
    return labels, calibration


def prepare_iq(
    power_path: Path, root: Path, margin_db: float = 6.0, threshold_method: str = "global_lower_quantile"
) -> Path:
    power = np.load(power_path, mmap_mode="r", allow_pickle=False)
    metadata = json.loads(power_path.with_suffix(".metadata.json").read_text())
    splits = chronological_split(len(power), embargo=20)
    occupancy, calibration = fit_occupancy(power, splits["train"], margin_db, threshold_method)
    target = root / "data/processed" / f"{power_path.stem}_margin{margin_db:g}_{threshold_method}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        occupancy=occupancy,
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
    )
    metadata.update(
        calibration=calibration,
        embargo_slots=20,
        occupancy_sha256=digest(occupancy),
        source_power_path=str(power_path),
        trace_replay_eligible=True,
        temporal_split="60/20/20 with 20-slot embargo on each side",
        primary_occupant_interpretation="Observed RF activity proxy; incumbent protection is a simulation assumption, with no verified transmitter identity or licensed rights",
    )
    write_json(target.with_suffix(".metadata.json"), metadata)
    # Canonical measured values preserve relative timing/units rather than inventing calibrated UTC/dBm.
    split_names = np.full(len(power), "embargo", dtype=object)
    for name, indices in splits.items():
        split_names[indices] = name
    t, n = power.shape
    canonical = pd.DataFrame(
        {
            "trace_id": power_path.stem,
            "dataset_id": "iraq_rf",
            "site_id": "Baghdad_campaign",
            "session_id": metadata["recording"]["Dataset_Day"],
            "sensor_id": "RTL-SDR V3 campaign receiver",
            "timestamp_utc": pd.NaT,
            "timestamp_quality": metadata["timestamp_quality"],
            "sample_start": np.repeat(np.arange(t) * metadata["fft_size"] + metadata["discard_samples"], n),
            "time_relative_s": np.repeat(
                np.arange(t) * metadata["slot_seconds"] + metadata["discard_seconds"], n
            ),
            "sample_rate_hz": metadata["sample_rate_hz"],
            "freq_center_hz": np.tile(metadata["freq_centers_hz"], t),
            "band_width_hz": metadata["bandwidth_hz"],
            "raw_power_value": np.asarray(power).ravel(),
            "raw_power_unit": "dB relative IQ^2",
            "calibrated_power_dbm": None,
            "noise_floor_dbm": None,
            "occupancy_reference": occupancy.ravel(),
            "occupancy_confidence": None,
            "occupancy_threshold_dbm": None,
            "occupancy_threshold_relative_db": np.tile(calibration["threshold_relative_db"], t),
            "ground_truth_kind": "threshold_measurement_proxy",
            "observed": True,
            "timestamp_skew_ms": 0.0,
            "source_file": metadata["source_member"],
            "source_sha256": metadata["source_prefix_sha256"],
            "preprocess_config_sha256": digest(calibration),
            "split_id": np.repeat(split_names, n),
            "license_id": "CC-BY-4.0",
        }
    )
    canonical.to_parquet(target.with_suffix(".parquet"), index=False, compression="zstd")
    return target


def load_iq_trace(path: Path, split: str) -> TraceEpisode:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    if metadata.get("synthetic") or not metadata.get("trace_replay_eligible"):
        raise ValueError("Not an audited real IQ trace")
    with np.load(path, allow_pickle=False) as data:
        indices = data[split]
        if len(indices) == 0 or np.any(np.diff(indices) != 1):
            raise ValueError("Trace split must be contiguous and nonempty")
        occupancy = data["occupancy"][indices]
    metadata["split_id"] = split
    metadata["split_sample_start_slot"] = int(indices[0])
    return TraceEpisode(occupancy, f"{path.stem}-{split}", "threshold_measurement_proxy", metadata)
