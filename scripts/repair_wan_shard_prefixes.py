#!/usr/bin/env python3
"""Repair corrupted Wan safetensors prefixes without redownloading valid suffixes."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

from modelscope_hub.api import HubApi
from modelscope_hub.constants import RepoType
from safetensors import safe_open


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "models/Wan-AI/Wan2.1-I2V-14B-480P"
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"

# The first byte offset proven to match the official remote object.  Only the
# bytes before this boundary came from malformed historical .incomplete files.
REPAIR_PREFIX_BYTES = {
    "diffusion_pytorch_model-00001-of-00007.safetensors": 178_257_920,
    "diffusion_pytorch_model-00002-of-00007.safetensors": 185_597_952,
    "diffusion_pytorch_model-00003-of-00007.safetensors": 1_630_535_680,
    "diffusion_pytorch_model-00005-of-00007.safetensors": 116_391_936,
    "diffusion_pytorch_model-00006-of-00007.safetensors": 149_946_368,
    "diffusion_pytorch_model-00007-of-00007.safetensors": 3_846_176_768,
}

EXPECTED = {
    "diffusion_pytorch_model-00001-of-00007.safetensors": (9_847_223_624, "09c0170242cfe9598208724585196ca18f294928fe25971149e1d7b37b3b51d6"),
    "diffusion_pytorch_model-00002-of-00007.safetensors": (9_797_021_016, "9e9dd8069241f0cbfe8f269e784e915ddaa7f177272b9850ae88c494d418c181"),
    "diffusion_pytorch_model-00003-of-00007.safetensors": (9_797_041_744, "22af36eb620c381d57a2dd5d5e3a001bf90ba0ca32f2d5013d6ca45225b11d5f"),
    "diffusion_pytorch_model-00004-of-00007.safetensors": (9_692_142_864, "5c2be0421b8df4a8e7dd31cdbb6489567438ae14b812b57518e43c6f4e49a755"),
    "diffusion_pytorch_model-00005-of-00007.safetensors": (9_692_142_864, "0ba7f3ab69f414f23e5566e607b9385a8f67907302cbaaf86507b3a826a20dda"),
    "diffusion_pytorch_model-00006-of-00007.safetensors": (9_692_142_864, "d7287e9233cbf67a75fce8ef8be24232ec93d6336f148972465db00008d69e28"),
    "diffusion_pytorch_model-00007-of-00007.safetensors": (7_061_933_536, "d1bc30f07b162c34b90e4ee4a349f81b6e1a3a342187ea9ea153c9e4dbb07676"),
}


def log(message: str) -> None:
    print(f"[wan-prefix-repair] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_prefix(api, filename: str, length: int) -> Path:
    target = ROOT / f".{filename}.prefix-repair"
    if target.exists() and target.stat().st_size > length:
        raise RuntimeError(f"oversized repair file: {target}")

    attempt = 0
    while (target.stat().st_size if target.exists() else 0) < length:
        current = target.stat().st_size if target.exists() else 0
        attempt += 1
        headers = api._headers()
        headers["Range"] = f"bytes={current}-{length - 1}"
        log(f"download {filename}: {current}/{length}, attempt={attempt}")
        try:
            response = api.download_stream(
                MODEL_ID, RepoType.MODEL, filename, revision="master", headers=headers
            )
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or int(match.group(1)) != current or int(match.group(2)) != length - 1:
                response.close()
                raise RuntimeError(
                    f"unexpected Content-Range {content_range!r} for requested offset {current}"
                )
            mode = "ab" if current else "wb"
            report_at = time.monotonic() + 30
            with response, target.open(mode) as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if not chunk:
                        continue
                    remaining = length - handle.tell()
                    if remaining <= 0:
                        break
                    handle.write(chunk[:remaining])
                    if time.monotonic() >= report_at:
                        log(f"progress {filename}: {handle.tell()}/{length}")
                        report_at = time.monotonic() + 30
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:  # network retries are expected
            log(f"retry {filename}: {type(exc).__name__}: {exc}")
            time.sleep(min(60, 5 + attempt * 5))
    if target.stat().st_size != length:
        raise RuntimeError(f"repair prefix has wrong size: {target.stat().st_size} != {length}")
    return target


def verify_file(path: Path, expected_size: int, expected_sha: str) -> None:
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"wrong size for {path.name}: {path.stat().st_size}")
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual_sha}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        key_count = len(list(handle.keys()))
    log(f"verified {path.name}: sha256={actual_sha}, tensors={key_count}")


def main() -> int:
    api = HubApi().legacy
    ROOT.mkdir(parents=True, exist_ok=True)
    for filename, prefix_length in REPAIR_PREFIX_BYTES.items():
        path = ROOT / filename
        expected_size, expected_sha = EXPECTED[filename]
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"base shard missing or wrong size: {path}")
        prefix = download_prefix(api, filename, prefix_length)
        log(f"installing repaired prefix into {filename}")
        with path.open("r+b") as output, prefix.open("rb") as source:
            output.seek(0)
            while chunk := source.read(16 * 1024 * 1024):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_file(path, expected_size, expected_sha)
        prefix.unlink()

    # Verify the untouched fourth shard too, so success certifies all seven.
    fourth = "diffusion_pytorch_model-00004-of-00007.safetensors"
    verify_file(ROOT / fourth, *EXPECTED[fourth])
    log("all seven Wan DiT shards passed official SHA-256 and safetensors validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
