"""Raw Bayer lights in any common FITS layout, read into the engine's layout.

The engine works on GRBG mosaics held as uint16, top row first, with the pattern starting at the
first pixel and even dimensions. A light stored any other way is converted as it is read:

- BAYERPAT (with XBAYROFF/YBAYROFF) describes the rows in the order the file stores them.
  ROWORDER = 'BOTTOM-UP' lights are turned top row first; a missing ROWORDER means top-down.
- When the pattern does not start the GRBG way, the first row and/or column is left out, and a
  last row or column when needed for even dimensions. Lights already in the engine's layout
  (the Seestar's) keep every pixel.
- Values are BZERO + BSCALE x stored and must lie within 0..65535. Whole numbers are kept
  exactly; BITPIX 8 data is multiplied by 256 (lossless) so the 16-bit saturation limit applies;
  float data whose maximum is at most 1 is normalized and is multiplied by 65535; other fractional
  values are rounded to whole numbers.
"""

import math
from pathlib import Path

import numpy as np

ENGINE_PATTERN = "GRBG"
PATTERNS = ("GRBG", "RGGB", "BGGR", "GBRG")
STORED = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
KEYS = ("BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "BZERO", "BSCALE", "BAYERPAT", "XBAYROFF", "YBAYROFF", "ROWORDER")


class Layout:
    """How a light is stored and where the engine's image lies in it, from the header alone.
    reason: None when the light can be read, else why not. A light without BAYERPAT is read as
    GRBG (declared is False); light groups leave such lights out."""

    def __init__(self, header):
        get = header.get
        self.reason = None
        self.bitpix = _int(get("BITPIX"))
        self.naxis = _int(get("NAXIS"))
        self.stored_shape = (_int(get("NAXIS2")) or 0, _int(get("NAXIS1")) or 0)
        self.bzero = _float(get("BZERO"), 0.0)
        self.bscale = _float(get("BSCALE"), 1.0)
        self.pattern = _text(get("BAYERPAT")).upper()
        self.declared = bool(self.pattern)
        self.pattern = self.pattern or ENGINE_PATTERN
        self.xoff = (_int(get("XBAYROFF")) or 0) % 2
        self.yoff = (_int(get("YBAYROFF")) or 0) % 2
        order = _text(get("ROWORDER")).upper()
        self.bottom_up = order == "BOTTOM-UP"
        self.top = self.left = 0
        self.shape = (0, 0)
        H, W = self.stored_shape
        if self.bitpix not in STORED:
            self.reason = f"BITPIX {get('BITPIX')} is not supported"
        elif self.naxis != 2:
            self.reason = "not a single raw image (colour or multi-plane data)" if self.naxis == 3 else "not a 2D image"
        elif self.pattern not in PATTERNS:
            self.reason = f"Bayer pattern {self.pattern} is not supported"
        elif order not in ("", "TOP-DOWN", "BOTTOM-UP"):
            self.reason = f"ROWORDER {order} is not supported"
        elif not (math.isfinite(self.bzero) and math.isfinite(self.bscale) and self.bscale != 0):
            self.reason = "invalid BZERO or BSCALE"
        else:
            self.top, self.left = next(
                (py, px)
                for py in (0, 1)
                for px in (0, 1)
                if all(self.colour(py + i, px + j) == ENGINE_PATTERN[2 * i + j] for i in (0, 1) for j in (0, 1))
            )
            self.shape = ((H - self.top) // 2 * 2, (W - self.left) // 2 * 2)
            if min(self.shape) < 2:
                self.reason = "image too small"

    def colour(self, y, x):
        """The colour of the pixel at row y (top row first) and column x."""
        s = self.stored_shape[0] - 1 - y if self.bottom_up else y
        return self.pattern[2 * ((s + self.yoff) % 2) + (x + self.xoff) % 2]

    @property
    def native(self):
        """True when no row or column is left out and the rows keep their order."""
        return not self.bottom_up and (self.top, self.left) == (0, 0) and self.shape == self.stored_shape

    @property
    def scale(self):
        return 256 if self.bitpix == 8 else 1

    def describe(self):
        """What reading does to this light, for the log ('' when nothing)."""
        notes = []
        if self.bottom_up:
            notes.append("rows turned top first (ROWORDER BOTTOM-UP)")
        H, W = self.stored_shape
        cuts = []
        for what, first, last in (
            ("row", self.top, H - self.top - self.shape[0]),
            ("column", self.left, W - self.left - self.shape[1]),
        ):
            ends = [end for end, cut in (("first", first), ("last", last)) if cut]
            if ends:
                cuts.append(" and ".join(ends) + " " + what)
        if cuts:
            notes.append(", ".join(cuts) + " left out to start on the GRBG layout with even dimensions")
        if self.bitpix == 8:
            notes.append("8-bit values multiplied by 256")
        elif self.bitpix < 0:
            notes.append("float values rounded to whole numbers (data within 0..1 multiplied by 65535)")
        return "; ".join(notes)

    def check(self, path):
        if self.reason is not None:
            raise ValueError(f"Cannot read light {path}: {self.reason}")

    def window(self, a=0, b=None, c=0, d=None):
        """Stored rows and columns (slices) holding engine rows a:b and columns c:d, and whether
        the stored rows run bottom to top."""
        h, w = self.shape
        b = h if b is None else b
        d = w if d is None else d
        if not (0 <= a < b <= h and 0 <= c < d <= w):
            raise ValueError("Invalid raw-light crop")
        H = self.stored_shape[0]
        rows = slice(H - self.top - b, H - self.top - a) if self.bottom_up else slice(self.top + a, self.top + b)
        return rows, slice(self.left + c, self.left + d)

    def float_scale(self, stored, path):
        """65535 for float data within 0..1 (normalized), else 1, judged on the whole light."""
        if self.bitpix > 0:
            return 1
        values = stored.astype(np.float64) * self.bscale + self.bzero
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite pixels in light: {path}. No light was excluded automatically.")
        return 65535 if values.size and values.max() <= 1 else 1

    def convert(self, stored, path, float_scale=1):
        """Engine values (uint16) from stored values, any shape. Rows are not reordered here.
        float_scale: float_scale() of the whole light."""
        if self.bitpix == 16 and self.bzero == 32768 and self.bscale == 1:
            # the stored int16 plus 32768 is its bit pattern with the sign bit flipped
            unsigned = stored.view(stored.dtype.str.replace("i", "u"))
            return np.bitwise_xor(unsigned, np.uint16(0x8000), dtype=np.uint16)
        if self.bitpix > 0 and float(self.bzero).is_integer() and float(self.bscale).is_integer():
            values = stored.astype(np.int64) * int(self.bscale) + int(self.bzero)
            values *= self.scale
        else:
            values = stored.astype(np.float64) * self.bscale + self.bzero
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite pixels in light: {path}. No light was excluded automatically.")
            values = np.rint(values * (self.scale * float_scale))
        if values.size and (values.min() < 0 or values.max() > 65535):
            raise ValueError(
                f"Light {path} has values outside 0..65535 ({values.min():g} to {values.max():g}). "
                "No light was excluded automatically."
            )
        return values.astype(np.uint16)


def _int(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) and f.is_integer() else None


def _float(v, default):
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _text(v):
    return str(v if v is not None else "").strip()


def _card_value(text):
    """A header card's value: quoted strings unquoted, anything else as written."""
    text = text.strip()
    if text.startswith("'"):
        end, out = 1, []
        while end < len(text):
            q = text.find("'", end)
            if q < 0:
                break
            out.append(text[end:q])
            if text[q + 1 : q + 2] == "'":  # '' is an escaped quote
                out.append("'")
                end = q + 2
                continue
            return "".join(out).rstrip()
        return None
    return text.split("/")[0].strip()


def primary_header(path):
    """(data byte offset, {key: value} for the keys a layout needs) of the primary header, or
    None when the file is not a plain FITS file."""
    cards = {}
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2880), b""):
            if len(block) < 2880:
                return None
            for k in range(0, 2880, 80):
                card = block[k : k + 80].decode("ascii", "replace")
                key = card[:8].strip()
                if key == "END":
                    return f.tell(), cards
                if card[8:10] == "= " and key in KEYS:
                    cards[key] = _card_value(card[10:])
    return None


def _direct(path):
    """(Layout, data offset) when the light's data can be read straight from the file."""
    found = primary_header(path)
    if found is None:
        return None
    offset, cards = found
    layout = Layout(cards)
    return (layout, offset) if layout.reason is None else None


def read_light(path, box=None):
    """A light in the engine's layout as uint16 (rows a:b, columns c:d when box = (a, b, c, d)).
    Plain FITS files are read directly; anything else (compressed data, an image in an
    extension) goes through astropy, with the same conversion."""
    box = box or (0, None, 0, None)
    direct = _direct(path)
    if direct is not None:
        layout, offset = direct
        H, W = layout.stored_shape
        dtype = np.dtype(STORED[layout.bitpix])
        stored = np.fromfile(path, dtype=dtype, count=H * W, offset=offset)
        if stored.size != H * W:
            raise ValueError(f"Truncated light: {path}; expected {H * W} pixels, read {stored.size}")
        stored = stored.reshape(H, W)
    else:
        from astropy.io import fits

        with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                raise ValueError(f"No image data in light: {path}")
            layout = Layout(hdu.header)
            layout.check(path)
            stored = np.asarray(hdu.data)
            if stored.shape != layout.stored_shape:
                raise ValueError(f"Light {path}: data shape {stored.shape} does not match its header")
    rows, cols = layout.window(*box)
    part = stored[rows, cols]
    if direct is not None and layout.bitpix == 16 and layout.bzero == 32768 and layout.bscale == 1:
        # The freshly read buffer is ours: convert's int16 + 32768 in place, on the window only
        # (native byte order, then the sign bit flipped). Same values in about half the time;
        # numpy's byte-swapping cast in convert took 12.7 ms per 3840 x 2160 light, this 6.1 ms.
        part = part.view(part.dtype.str.replace("i", "u"))
        if not part.dtype.isnative:
            part.byteswap(inplace=True)
            part = part.view(part.dtype.newbyteorder())
        part ^= np.uint16(0x8000)
        if layout.bottom_up:
            part = part[::-1]
        return np.ascontiguousarray(part)
    if layout.bottom_up:
        part = part[::-1]
    return np.ascontiguousarray(layout.convert(part, path, layout.float_scale(stored, path)))


def direct_source(path):
    """(Layout, data offset, float scale) of a plain FITS light that read_crop can read, or None."""
    direct = _direct(path)
    if direct is None:
        return None
    layout, offset = direct
    float_scale = 1
    if layout.bitpix < 0:
        H, W = layout.stored_shape
        stored = np.fromfile(path, dtype=np.dtype(STORED[layout.bitpix]), count=H * W, offset=offset)
        if stored.size != H * W:
            raise ValueError(f"Truncated light: {path}; expected {H * W} pixels, read {stored.size}")
        float_scale = layout.float_scale(stored, path)
    return layout, offset, float_scale


def read_crop(path, source, box):
    """Engine rows a:b, columns c:d of a plain FITS light (source: direct_source()), through a
    read-only mapping that is closed before returning (only the stored rows of the crop are read)."""
    layout, offset, float_scale = source
    H, W = layout.stored_shape
    dtype = np.dtype(STORED[layout.bitpix])
    if Path(path).stat().st_size < offset + H * W * dtype.itemsize:
        raise ValueError("Truncated light: " + str(path))
    rows, cols = layout.window(*box)
    mapped = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(H, W))
    try:
        part = mapped[rows, cols]
        if layout.bottom_up:
            part = part[::-1]
        return np.ascontiguousarray(layout.convert(part, path, float_scale))
    finally:
        mapped._mmap.close()
