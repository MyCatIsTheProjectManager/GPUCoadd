"""Compare the original and optimized FITS export conversion on full-size S50 Pro RGB lights.

Includes writing, flushing and SHA-256 read-back verification. Excludes GPU rendering.
Only one temporary FITS is kept at a time; original input lights are not needed or modified.
"""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fits_output import stack_chunks, write_fits
from worker.bridge import ADU_SCALE, stack_pixels


def original_pixels(img, offset):
    """The conversion before export optimization, retained as the benchmark reference."""
    out = (img + np.float32(offset)) / np.float32(ADU_SCALE)
    finite = np.isfinite(out)
    low = int(np.count_nonzero(finite & (out <= 0)))
    high = int(np.count_nonzero(finite & (out > 1)))
    out = np.nan_to_num(out, nan=0.0)
    np.clip(out, np.float32(1 / ADU_SCALE), 1, out=out, where=finite)
    return out.astype(np.float32, copy=False), low, high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lights", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("out_export_bench"))
    args = parser.parse_args()
    if args.lights < 1 or args.rounds < 1:
        parser.error("lights and rounds must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    image = rng.normal(2000, 3000, (3840, 2160, 3)).astype(np.float32)
    image[:80] = np.nan
    image[:, :40] = np.nan
    header = fits.Header({"PRODTYPE": "ALIGNED", "ADUSCALE": ADU_SCALE, "ADUOFFS": 2063.0})
    converters = {"original": original_pixels, "optimized": partial(stack_pixels, count_clipping=False)}
    results = []
    expected = None
    with tempfile.TemporaryDirectory(dir=args.out) as folder:
        path = Path(folder) / "light.fits"
        for round_index in range(args.rounds):
            # Alternate order to reduce the advantage of running after a warmed cache.
            order = list(converters) if round_index % 2 == 0 else list(reversed(converters))
            for name in order:
                conversion_s = 0.0

                def convert(img, offset):
                    nonlocal conversion_s
                    start = time.perf_counter()
                    result = converters[name](img, offset)
                    conversion_s += time.perf_counter() - start
                    return result

                start = time.perf_counter()
                for _ in range(args.lights):
                    write_fits(path, header, (3, 3840, 2160), np.float32, stack_chunks(image[::-1], 2063, convert))
                elapsed = time.perf_counter() - start
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if expected is None:
                    expected = digest
                if digest != expected:
                    raise AssertionError("Exported FITS bytes differ")
                row = dict(round=round_index + 1, mode=name, lights=args.lights, seconds=round(elapsed, 3),
                           conversion_s=round(conversion_s, 3), sha256=digest)
                results.append(row)
                print(json.dumps(row), flush=True)
    report = {"shape": [3840, 2160, 3], "results": results}
    (args.out / "conversion_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
