"""Write raw GRBG lights in other FITS layouts, to check that GPUCoadd reads every layout into the
same engine image.

  python tools/fits_layouts.py SRC_DIR DST_DIR VARIANT [VARIANT ...]

Each variant goes to DST_DIR/<variant>/ with the source file names; the source files are only
read. Variants:

  bottom-up   rows stored bottom first, ROWORDER = 'BOTTOM-UP', BAYERPAT in stored order
  xoff        GRBG declared as RGGB with XBAYROFF = 1
  yoff        GRBG declared as BGGR with YBAYROFF = 1
  f32, f64    BITPIX -32 / -64, the same values as floats
  i32         BITPIX 32, BZERO 0
  bzero       BITPIX 16 with BZERO = the light's minimum + 32768 (needs a range under 65536)
  rggb, bggr, gbrg
              first column, first row, or both left out, BAYERPAT set to match
  rggb-ref, bggr-ref, gbrg-ref
              GRBG, cut to the image GPUCoadd reads from the variant above (two columns
              and/or rows left out): both must stack to identical pixels

The variants without "rggb", "bggr" or "gbrg" in their name read back to exactly the source
lights' pixels.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lightgroups import fits_files  # noqa: E402

CUTS = {"rggb": (0, 1), "bggr": (1, 0), "gbrg": (1, 1)}  # rows, columns left out of a GRBG light
SKIP = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "BZERO", "BSCALE"}
SKIP |= {"BAYERPAT", "ROWORDER", "XBAYROFF", "YBAYROFF"}


def write_raw(path, stored, cards, header=None):
    """A primary-HDU FITS file holding `stored` exactly as given (any of the FITS dtypes) with
    `cards` (BZERO, BAYERPAT, ...), plus the other cards of `header`."""
    stored = np.asarray(stored)
    if stored.dtype == np.uint16:  # FITS holds it as int16 with BZERO 32768
        stored, cards = _int16(stored), {**_bzero(), **cards}
    bitpix = {"u1": 8, "i2": 16, "i4": 32, "i8": 64, "f4": -32, "f8": -64}[stored.dtype.str[1:]]
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = bitpix
    h["NAXIS"] = 2
    h["NAXIS1"] = stored.shape[1]
    h["NAXIS2"] = stored.shape[0]
    for key, value in cards.items():
        h[key] = value
    for card in header.cards if header is not None else ():
        if card.keyword not in SKIP and card.keyword not in h and card.keyword not in ("COMMENT", "HISTORY", ""):
            h.append(card)
    data = stored.astype(stored.dtype.newbyteorder(">"), copy=False).tobytes()
    with open(path, "wb") as f:
        f.write(h.tostring().encode("ascii"))
        f.write(data + b"\0" * (-len(data) % 2880))


def variant(image, name):
    """(stored array, cards) of a GRBG uint16 image (top row first) in layout `name`."""
    grbg = {"BAYERPAT": "GRBG"}
    if name in CUTS or name.endswith("-ref"):
        dy, dx = CUTS[name.removesuffix("-ref")]
        if name.endswith("-ref"):
            dy, dx, pattern = 2 * dy, 2 * dx, "GRBG"
        else:
            pattern = name.upper()
        return _int16(image[dy:, dx:]), {**_bzero(), "BAYERPAT": pattern}
    if name == "bottom-up":
        flipped = image[::-1]
        pattern = "GRBG" if image.shape[0] % 2 else "BGGR"  # the pattern of the first stored row
        return _int16(flipped), {**_bzero(), "BAYERPAT": pattern, "ROWORDER": "BOTTOM-UP"}
    if name == "xoff":
        return _int16(image), {**_bzero(), "BAYERPAT": "RGGB", "XBAYROFF": 1}
    if name == "yoff":
        return _int16(image), {**_bzero(), "BAYERPAT": "BGGR", "YBAYROFF": 1}
    if name == "f32":
        return image.astype(np.float32), grbg
    if name == "f64":
        return image.astype(np.float64), grbg
    if name == "i32":
        return image.astype(np.int32), grbg
    if name == "bzero":
        zero = int(image.min()) + 32768
        if int(image.max()) - zero > 32767:
            raise ValueError("value range too wide for another BZERO")
        return (image.astype(np.int64) - zero).astype(np.int16), {"BZERO": zero, "BSCALE": 1, **grbg}
    raise ValueError("unknown variant " + name)


def _int16(image):
    return (image.astype(np.int32) - 32768).astype(np.int16)


def _bzero():
    return {"BZERO": 32768, "BSCALE": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("variants", nargs="+")
    args = ap.parse_args()
    src = fits_files(args.src)
    dst = Path(args.dst).resolve()
    if any(dst == p.parent.resolve() or p.parent.resolve() in dst.parents for p in src[:1]):
        raise SystemExit("The destination must not be inside the source folder")
    for name in args.variants:
        out = dst / name
        out.mkdir(parents=True, exist_ok=True)
        for p in src:
            with fits.open(p) as hdul:
                header = hdul[0].header
                image = hdul[0].data
                if header.get("BAYERPAT", "").strip() != "GRBG" or image.dtype != np.uint16 or "ROWORDER" in header:
                    raise SystemExit(f"{p}: expected a top-down GRBG 16-bit light")
                stored, cards = variant(image, name)
                write_raw(out / p.name, stored, cards, header)
        print(f"{name}: {len(src)} lights -> {out}")


if __name__ == "__main__":
    main()
