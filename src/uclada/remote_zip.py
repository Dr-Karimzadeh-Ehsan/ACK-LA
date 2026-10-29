"""HTTP range access to official ZIP members without downloading unrelated recordings."""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from uclada.common import file_hash, write_json


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file; every byte range is logged and must be honored by server."""

    def __init__(self, url: str, size: int, block_bytes: int = 8 * 2**20):
        self.url, self.size, self.block_bytes = url, size, block_bytes
        self.position = 0
        self.session = requests.Session()
        self.requests: list[dict[str, Any]] = []
        self.cache_start = 0
        self.cache = b""
        self._fetch(max(0, size - 131072), size)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if target < 0:
            raise ValueError("Negative seek")
        self.position = target
        return target

    def _fetch(self, start: int, end: int) -> None:
        end = min(end, self.size)
        for attempt in range(4):
            begin = time.perf_counter()
            try:
                with self.session.get(
                    self.url,
                    headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"},
                    stream=True,
                    timeout=(20, 90),
                ) as r:
                    if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(
                        f"bytes {start}-"
                    ):
                        raise RuntimeError(f"Server did not honor bounded byte range: {r.status_code}")
                    data = r.content
                    if len(data) != end - start:
                        raise RuntimeError("Truncated byte range")
                    self.requests.append(
                        {
                            "start": start,
                            "end_exclusive": end,
                            "status": r.status_code,
                            "bytes": len(data),
                            "seconds": time.perf_counter() - begin,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                self.cache_start, self.cache = start, data
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        remaining = min(size, self.size - self.position)
        pieces = []
        while remaining > 0:
            if not (self.cache_start <= self.position < self.cache_start + len(self.cache)):
                self._fetch(
                    self.position, self.position + max(self.block_bytes, min(remaining, self.block_bytes))
                )
            offset = self.position - self.cache_start
            available = min(remaining, len(self.cache) - offset)
            pieces.append(self.cache[offset : offset + available])
            self.position += available
            remaining -= available
        return b"".join(pieces)


def zip_inventory(url: str, size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    remote = HTTPRangeFile(url, size)
    with zipfile.ZipFile(remote) as archive:
        rows = [
            {
                "name": m.filename,
                "raw_bytes": m.file_size,
                "compressed_bytes": m.compress_size,
                "compression": m.compress_type,
                "crc32": m.CRC,
                "header_offset": m.header_offset,
            }
            for m in archive.infolist()
        ]
    return rows, remote.requests


def download_prefix(
    url: str, size: int, member: str, target: Path, raw_bytes: int, source_metadata: dict[str, Any]
) -> dict[str, Any]:
    """Keep a documented prefix of one contiguous capture; never claim archive-wide MD5 validation."""
    receipt_path = target.with_suffix(target.suffix + ".receipt.json")
    if target.exists() and receipt_path.exists():
        old = __import__("json").loads(receipt_path.read_text())
        if target.stat().st_size == old["prefix_bytes"] and file_hash(target) == old["prefix_sha256"]:
            return old
        raise RuntimeError("Existing IQ prefix integrity mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    remote = HTTPRangeFile(url, size)
    start = time.perf_counter()
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(member)
        requested = min(raw_bytes, info.file_size)
        written = 0
        with archive.open(info) as source, partial.open("wb") as output:
            while written < requested:
                chunk = source.read(min(2**20, requested - written))
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
        if written != requested:
            raise RuntimeError("Incomplete requested IQ prefix")
        partial.rename(target)
        result = {
            "source_url": url,
            "source_archive_bytes": size,
            "member": member,
            "member_raw_bytes": info.file_size,
            "member_compressed_bytes": info.compress_size,
            "member_official_crc32": info.CRC,
            "member_full_crc_checked": requested == info.file_size,
            "archive_md5_checked": False,
            "prefix_bytes": written,
            "prefix_sha256": file_hash(target),
            "contiguous_sample_prefix": True,
            "prefix_offset_samples": 0,
            "requested_sample_count": written // 8,
            "dtype": "<c8",
            "seconds": time.perf_counter() - start,
            "http_requests": remote.requests,
            "metadata": source_metadata,
        }
    write_json(receipt_path, result)
    return result
