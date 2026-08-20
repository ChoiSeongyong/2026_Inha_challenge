#!/usr/bin/env python3
"""Resume one large ModelScope file with validated parallel HTTP ranges.

The existing ``*.incomplete`` file is treated as a contiguous prefix beginning at
byte zero.  The remaining bytes are downloaded into independent chunks.  The
final file is published atomically only after exact-size and SHA256 validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


CONTENT_RANGE_RE = re.compile(r"^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$", re.I)
PRINT_LOCK = threading.Lock()
STOP = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[bytes]] = set()
ACTIVE_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[parallel-download] {stamp} {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size-mib", type=int, default=256)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    return parser.parse_args()


def parse_content_range(header_path: Path) -> tuple[int, int, int] | None:
    try:
        lines = header_path.read_text(encoding="latin-1", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    found = None
    for line in lines:
        match = CONTENT_RANGE_RE.match(line.strip())
        if match:
            found = tuple(int(value) for value in match.groups())
    return found


def append_valid_fetch(
    partial: Path,
    fetch: Path,
    header: Path,
    range_start: int,
    range_end: int,
    total_size: int,
) -> int:
    if not fetch.exists():
        return 0
    fetched = fetch.stat().st_size
    content_range = parse_content_range(header)
    expected_header = (range_start, range_end, total_size)
    if fetched <= 0 or fetched > range_end - range_start + 1 or content_range != expected_header:
        fetch.unlink(missing_ok=True)
        header.unlink(missing_ok=True)
        return 0
    with partial.open("ab") as destination, fetch.open("rb") as source:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    fetch.unlink(missing_ok=True)
    header.unlink(missing_ok=True)
    return fetched


def download_chunk(
    url: str,
    chunk_path: Path,
    start: int,
    end: int,
    total_size: int,
    retry_delay: float,
) -> Path:
    expected = end - start + 1
    partial = chunk_path.with_suffix(chunk_path.suffix + ".partial")
    fetch = chunk_path.with_suffix(chunk_path.suffix + ".fetch")
    header = chunk_path.with_suffix(chunk_path.suffix + ".headers")

    if chunk_path.exists():
        if chunk_path.stat().st_size == expected:
            return chunk_path
        raise RuntimeError(f"invalid completed chunk size: {chunk_path}")
    if partial.exists() and partial.stat().st_size > expected:
        raise RuntimeError(f"oversized partial chunk: {partial}")

    attempts = 0
    while not STOP.is_set():
        current = partial.stat().st_size if partial.exists() else 0
        if current == expected:
            os.replace(partial, chunk_path)
            return chunk_path

        request_start = start + current
        fetch.unlink(missing_ok=True)
        header.unlink(missing_ok=True)
        command = [
            "curl",
            "-L",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "--range",
            f"{request_start}-{end}",
            "--dump-header",
            str(header),
            "--output",
            str(fetch),
            url,
        ]
        process = subprocess.Popen(command)
        with ACTIVE_LOCK:
            ACTIVE_PROCESSES.add(process)
        try:
            return_code = process.wait()
        finally:
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES.discard(process)

        appended = append_valid_fetch(
            partial, fetch, header, request_start, end, total_size
        )
        attempts += 1
        if appended:
            current += appended
            log(
                f"chunk {start}-{end}: {current}/{expected} bytes "
                f"({100.0 * current / expected:.1f}%), curl_rc={return_code}"
            )
        else:
            log(
                f"chunk {start}-{end}: retry {attempts}; no validated bytes "
                f"(curl_rc={return_code})"
            )
            time.sleep(min(60.0, retry_delay * max(1, attempts)))

    raise InterruptedError(f"stopped while downloading {start}-{end}")


def bytes_downloaded(prefix_size: int, part_dir: Path, expected_size: int) -> int:
    total = prefix_size
    for path in part_dir.glob("chunk_*.bin*"):
        # Include the currently arriving .fetch body so the 10-second progress
        # line remains live.  It is still validated against Content-Range before
        # becoming a durable .partial or completed chunk.
        if path.name.endswith(".headers"):
            continue
        total += path.stat().st_size
    return min(total, expected_size)


def monitor(prefix_size: int, part_dir: Path, expected_size: int) -> None:
    started = time.monotonic()
    initial = bytes_downloaded(prefix_size, part_dir, expected_size)
    while not STOP.wait(10):
        current = bytes_downloaded(prefix_size, part_dir, expected_size)
        elapsed = max(time.monotonic() - started, 0.001)
        speed = max(current - initial, 0) / elapsed
        remaining = expected_size - current
        eta = remaining / speed if speed > 0 else float("inf")
        eta_text = f"{eta / 3600:.2f}h" if eta != float("inf") else "unknown"
        log(
            f"TOTAL {current}/{expected_size} bytes "
            f"({100.0 * current / expected_size:.2f}%), "
            f"average={speed / 1024 / 1024:.2f} MiB/s, ETA={eta_text}"
        )
        if current >= expected_size:
            return


def install_signal_handlers() -> None:
    def stop_handler(signum: int, _frame: object) -> None:
        log(f"received signal {signum}; preserving all partial chunks")
        STOP.set()
        with ACTIVE_LOCK:
            processes = list(ACTIVE_PROCESSES)
        for process in processes:
            process.terminate()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)


def assemble_and_verify(
    prefix: Path,
    chunks: list[tuple[int, int, Path]],
    output: Path,
    expected_size: int,
    expected_sha256: str,
) -> None:
    assembling = output.with_suffix(output.suffix + ".assembling")
    digest = hashlib.sha256()
    written = 0
    log(f"assembling and hashing into {assembling}")
    with assembling.open("wb") as destination:
        for source_path in [prefix, *(chunk[2] for chunk in chunks)]:
            with source_path.open("rb") as source:
                while block := source.read(16 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
                    written += len(block)
        destination.flush()
        os.fsync(destination.fileno())

    actual_sha256 = digest.hexdigest()
    if written != expected_size:
        raise RuntimeError(f"assembled size mismatch: {written} != {expected_size}")
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA256 mismatch: {actual_sha256} != {expected_sha256}; "
            f"prefix and chunks were kept for diagnosis"
        )
    os.replace(assembling, output)
    log(f"verified SHA256={actual_sha256}; published {output}")


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.chunk_size_mib < 1:
        raise ValueError("workers and chunk-size-mib must be positive")
    output = args.output.resolve()
    prefix = (args.prefix or Path(str(output) + ".incomplete")).resolve()
    part_dir = Path(str(output) + ".parallel.parts")
    lock_path = Path(str(output) + ".parallel.lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another parallel downloader already holds the lock") from exc

        if output.exists():
            if output.stat().st_size == args.expected_size:
                log(f"final output already exists: {output}")
                return 0
            raise RuntimeError(f"refusing to overwrite unexpected final file: {output}")
        if not prefix.is_file():
            prefix.touch()
        prefix_size = prefix.stat().st_size
        if prefix_size > args.expected_size:
            raise RuntimeError(f"prefix is larger than expected file: {prefix_size}")

        chunk_bytes = args.chunk_size_mib * 1024 * 1024
        chunks: list[tuple[int, int, Path]] = []
        cursor = prefix_size
        while cursor < args.expected_size:
            end = min(cursor + chunk_bytes, args.expected_size) - 1
            path = part_dir / f"chunk_{cursor:012d}_{end:012d}.bin"
            chunks.append((cursor, end, path))
            cursor = end + 1

        log(
            f"prefix={prefix_size} bytes ({100.0 * prefix_size / args.expected_size:.2f}%); "
            f"remaining_chunks={len(chunks)}, workers={args.workers}"
        )
        install_signal_handlers()
        monitor_thread = threading.Thread(
            target=monitor,
            args=(prefix_size, part_dir, args.expected_size),
            daemon=True,
        )
        monitor_thread.start()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    download_chunk,
                    args.url,
                    path,
                    start,
                    end,
                    args.expected_size,
                    args.retry_delay,
                )
                for start, end, path in chunks
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        STOP.set()
        monitor_thread.join(timeout=1)
        assemble_and_verify(
            prefix, chunks, output, args.expected_size, args.sha256
        )
        shutil.rmtree(part_dir)
        prefix.unlink()
        log("download complete; temporary prefix and chunks removed")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        STOP.set()
        raise SystemExit(130)
