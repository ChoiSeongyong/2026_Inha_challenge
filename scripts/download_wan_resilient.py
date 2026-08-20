#!/usr/bin/env python3
"""Resume Wan weights with a small number of connections.

ModelScope already supports HTTP Range requests for ``.incomplete`` files.
This wrapper deliberately downloads only two files at a time and retries the
same file indefinitely, so a broken connection does not restart the pipeline
or discard the partial files.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
from pathlib import Path

from modelscope_hub.api import HubApi
from modelscope_hub.constants import RepoType


PROJECT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT / "models" / "Wan-AI"
I2V_ROOT = MODEL_ROOT / "Wan2.1-I2V-14B-480P"
T2V_ROOT = MODEL_ROOT / "Wan2.1-T2V-1.3B"

I2V_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
T2V_ID = "Wan-AI/Wan2.1-T2V-1.3B"

DIT_FILES = {
    "diffusion_pytorch_model-00001-of-00007.safetensors": 9847223624,
    "diffusion_pytorch_model-00002-of-00007.safetensors": 9797021016,
    "diffusion_pytorch_model-00003-of-00007.safetensors": 9797041744,
    "diffusion_pytorch_model-00004-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00005-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00006-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00007-of-00007.safetensors": 7061933536,
}
CLIP = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
T5 = "models_t5_umt5-xxl-enc-bf16.pth"
VAE = "Wan2.1_VAE.pth"
MS_CLIENT = HubApi().legacy


def log(message: str) -> None:
    print(f"[wan-resume] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def complete(path: Path, expected: int | None = None) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    return expected is None or path.stat().st_size == expected


def progress(path: Path, expected: int | None) -> str:
    partial = path.with_name(path.name + ".incomplete")
    if path.exists():
        size = path.stat().st_size
    elif partial.exists():
        size = partial.stat().st_size
    else:
        size = 0
    if expected:
        return f"{size}/{expected} bytes ({100.0 * size / expected:.2f}%)"
    return f"{size} bytes"


def download_one(model_id: str, root: Path, filename: str, expected: int | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / filename
    if complete(destination, expected):
        log(f"already valid: {filename} ({destination.stat().st_size} bytes)")
        return

    attempt = 0
    while True:
        attempt += 1
        log(f"start {filename}, attempt={attempt}, current={progress(destination, expected)}")
        try:
            partial = destination.with_name(destination.name + ".incomplete")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = MS_CLIENT._headers()
            if offset:
                headers["Range"] = f"bytes={offset}-"
            response = MS_CLIENT.download_stream(
                model_id,
                RepoType.MODEL,
                filename,
                revision="master",
                headers=headers,
            )
            # This endpoint sometimes returns HTTP 200 with a valid
            # Content-Range header (instead of the usual HTTP 206). Accept
            # that response, but never trust a plain 200: it would truncate
            # the existing partial file.
            content_range = response.headers.get("Content-Range", "")
            range_honored = content_range.startswith(f"bytes {offset}-")
            if offset and not range_honored:
                response.close()
                raise RuntimeError(
                    f"server ignored Range (status={response.status_code}, content_range={content_range!r}, offset={offset})"
                )
            mode = "ab" if offset and range_honored else "wb"
            report_at = time.monotonic() + 30
            report_size = offset
            report_time = time.monotonic()
            with response, partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        now = time.monotonic()
                        if now >= report_at:
                            size = partial.stat().st_size
                            elapsed = max(now - report_time, 1e-6)
                            rate = (size - report_size) / elapsed / (1024 * 1024)
                            log(
                                f"progress {filename}: {progress(destination, expected)}, "
                                f"speed={rate:.2f} MiB/s"
                            )
                            report_size = size
                            report_time = now
                            report_at = now + 30
            if complete(partial, expected):
                partial.replace(destination)
            if complete(destination, expected):
                log(f"validated: {filename} ({destination.stat().st_size} bytes)")
                return
            log(f"stream ended before validation: {filename}; current={progress(destination, expected)}")
        except Exception as exc:  # noqa: BLE001 - network failures are expected here
            log(f"failed: {filename}: {type(exc).__name__}: {exc}; current={progress(destination, expected)}")
        time.sleep(min(60, 5 + attempt * 5))


def main() -> int:
    log("starting; existing .incomplete files are preserved")
    log(f"I2V root: {I2V_ROOT}")
    log("downloading DiT shards with max_workers=2")

    # Two connections is intentional: the previous seven-way download was
    # repeatedly losing long HTTP responses with IncompleteRead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(download_one, I2V_ID, I2V_ROOT, filename, size)
            for filename, size in DIT_FILES.items()
        ]
        for job in concurrent.futures.as_completed(jobs):
            job.result()

    # CLIP was already complete, but validate it before proceeding.
    download_one(I2V_ID, I2V_ROOT, CLIP, 4772359047)
    log("all I2V/CLIP files validated")

    log("downloading T5 and VAE sequentially")
    download_one(T2V_ID, T2V_ROOT, T5, None)
    download_one(T2V_ID, T2V_ROOT, VAE, None)
    log("all Wan public weights validated; no GPU training was started")
    log("after Cosmos finishes, start VACE training with the command from the handoff")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[wan-resume] interrupted; partial files remain on disk", file=sys.stderr, flush=True)
        raise
