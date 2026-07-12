#!/usr/bin/env python
"""Pre-place AudioSeal weights in torch's cache so its loader works offline / behind a mirror.

AudioSeal's model cards point at hardcoded ``huggingface.co`` URLs downloaded via
``torch.hub`` (keyed by ``sha1(urlparse(url).path)[:24]``). This script downloads the
generator/detector weights from a mirror to those exact cache paths.

    HF_MIRROR=https://hf-mirror.com python scripts/prefetch_audioseal.py
"""

from __future__ import annotations

import os
import urllib.request
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlparse

MIRROR = os.environ.get("HF_MIRROR", "https://hf-mirror.com")
CACHE = Path(os.environ.get("AUDIOSEAL_CACHE_DIR", str(Path.home() / ".cache" / "audioseal")))
FILES = {
    "generator": "https://huggingface.co/facebook/audioseal/resolve/main/generator_base.pth",
    "detector": "https://huggingface.co/facebook/audioseal/resolve/main/detector_base.pth",
}


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        h = sha1(urlparse(url).path.encode()).hexdigest()[:24]
        dst = CACHE / h
        if dst.exists() and dst.stat().st_size > 0:
            print(f"{name}: already cached ({dst})")
            continue
        src = url.replace("https://huggingface.co", MIRROR)
        print(f"{name}: {src} -> {dst}")
        urllib.request.urlretrieve(src, dst)
        print(f"  {dst.stat().st_size} bytes")


if __name__ == "__main__":
    main()
