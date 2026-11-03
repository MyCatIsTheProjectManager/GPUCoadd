"""Bounded FITS writing, byte verification and same-directory atomic publication."""

import ctypes
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from astropy.io import fits
from output_store import BLOCK_BYTES, check_space, row_blocks

# Optional native comparison; NumPy remains the portable fallback. Both compare
# every byte. Argument types prevent pointer truncation on 64-bit platforms.
try:
    _runtime = ctypes.CDLL("msvcrt" if os.name == "nt" else None)
    _memcmp = _runtime.memcmp
    _memcmp.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)
    _memcmp.restype = ctypes.c_int
except (AttributeError, OSError):
    _memcmp = None


def image_header(header, shape, dtype):
    template = fits.PrimaryHDU(np.zeros((1,) * len(shape), dtype=dtype), header=header)
    h = template.header.copy()
    for axis, length in enumerate(reversed(shape), 1):
        h[f"NAXIS{axis}"] = int(length)
    # Raw-light checksums cannot describe new image data. Verification below covers
    # the serialized output bytes, independently of FITS pseudo-unsigned scaling.
    for key in ("CHECKSUM", "DATASUM"):
        if key in h:
            del h[key]
    return h


def write_encoded(path, header, shape, body, check_cancel=None):
    """write_fits for float32 pixels already in FITS form (export_kernels): the same header,
    full read-back against body, and publication."""
    path = Path(path)
    check = check_cancel or (lambda: None)
    size = int(np.prod(shape, dtype=object)) * 4
    if body.dtype != np.uint8 or body.nbytes != size:
        raise ValueError("Formatted light does not match the image size")
    check_space([(path.parent, size + 65536)])
    path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = image_header(header, shape, np.float32).tostring().encode("ascii")
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(header_bytes)
            for start in range(0, size, BLOCK_BYTES):
                check()
                stream.write(memoryview(body[start : start + BLOCK_BYTES]))
            stream.write(b"\0" * (-size % 2880))
            stream.flush()
            os.fsync(stream.fileno())
        verify_fits(temp, shape, size, None, check, header_bytes, body)
        check()
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return str(path)


def write_fits(path, header, shape, dtype, chunks, check_cancel=None, before_publish=None, verify_buffer=None):
    """Publish only after a full read-back check. An exclusive RAM byte buffer can
    hold the serialized pixels for exact comparison instead of two hash passes.
    Without that buffer, verification stays bounded and uses SHA-256 as before.
    """
    path = Path(path)
    check = check_cancel or (lambda: None)
    dtype = np.dtype(dtype)
    size = int(np.prod(shape, dtype=object)) * dtype.itemsize
    if verify_buffer is not None:
        if (
            verify_buffer.dtype != np.uint8 or verify_buffer.ndim != 1
            or not verify_buffer.flags.c_contiguous or not verify_buffer.flags.writeable
            or verify_buffer.nbytes < size
        ):
            raise ValueError("FITS verification buffer must be contiguous uint8 storage large enough for the image")
    check_space([(path.parent, size + 65536)])
    path.parent.mkdir(parents=True, exist_ok=True)
    h = image_header(header, shape, dtype)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    os.close(fd)
    temp = Path(name)
    stream = None
    expected = hashlib.sha256() if verify_buffer is None else None
    written = 0
    try:
        stream = fits.StreamingHDU(temp, h)
        for block in chunks:
            check()
            block = np.asarray(block)
            if block.dtype != dtype:
                raise TypeError("FITS chunk dtype changed")
            if dtype == np.dtype(np.uint32):
                physical = np.bitwise_xor(block, np.uint32(0x80000000)).view(np.int32)
                source, stored_dtype = physical, ">i4"
            elif dtype == np.dtype(np.float32):
                source, stored_dtype = block, ">f4"
            else:
                raise TypeError("Unsupported streaming FITS dtype")
            if written + source.nbytes > size:
                raise OSError("FITS data stream exceeds the image size")
            if verify_buffer is None:
                encoded = np.ascontiguousarray(source, dtype=stored_dtype)
                expected.update(memoryview(encoded).cast("B"))
            else:
                encoded = verify_buffer[written:written + source.nbytes].view(stored_dtype).reshape(source.shape)
                encoded[:] = source
            written += encoded.nbytes
            stream.write(encoded)
        if not stream.writecomplete or written != size:
            raise OSError("Incomplete FITS data stream")
        stream.close()
        stream = None
        with temp.open("r+b") as f:
            f.flush()
            os.fsync(f.fileno())
        verify_fits(
            temp, shape, size, expected.hexdigest() if expected is not None else None,
            check, h.tostring().encode("ascii"), verify_buffer,
        )
        check()
        if before_publish is not None:
            before_publish()
        os.replace(temp, path)  # old output survives until the new file is complete and verified
    finally:
        try:
            if stream is not None:
                stream.close()
        finally:
            temp.unlink(missing_ok=True)
    return str(path)


def verify_fits(path, shape, data_bytes, digest, check=lambda: None, header_bytes=None, expected_bytes=None):
    """Verify structure and all pixel bytes: against expected_bytes when given, else against a SHA-256 digest."""
    if expected_bytes is not None and (
        expected_bytes.dtype != np.uint8 or expected_bytes.ndim != 1
        or not expected_bytes.flags.c_contiguous or expected_bytes.nbytes < data_bytes
    ):
        raise ValueError("Expected FITS bytes must be contiguous uint8 storage large enough for the image")
    with fits.open(path, memmap=True, do_not_scale_image_data=True, uint=False) as hdus:
        hdus.verify("exception")
        h = hdus[0].header
        actual = tuple(h[f"NAXIS{i}"] for i in range(len(shape), 0, -1))
        if len(hdus) != 1 or actual != tuple(shape):
            raise OSError("FITS header verification failed")
        offset = hdus[0].fileinfo()["datLoc"]
    expected_size = offset + ((data_bytes + 2879) // 2880) * 2880
    if Path(path).stat().st_size != expected_size:
        raise OSError("FITS file length verification failed")
    got = hashlib.sha256() if expected_bytes is None else None
    with Path(path).open("rb") as f:
        if header_bytes is not None and (len(header_bytes) != offset or f.read(offset) != header_bytes):
            raise OSError("FITS header bytes verification failed")
        f.seek(offset)
        remaining = data_bytes
        # Exact comparison reads into one reusable buffer instead of allocating every block
        scratch = np.empty(min(4 << 20, remaining), np.uint8) if expected_bytes is not None else None
        while remaining:
            check()
            if scratch is None:
                block = f.read(min(4 << 20, remaining))
                if not block:
                    raise OSError("Truncated FITS data")
                got.update(block)
                remaining -= len(block)
                continue
            count = f.readinto(memoryview(scratch)[:min(scratch.nbytes, remaining)])
            if not count:
                raise OSError("Truncated FITS data")
            start = data_bytes - remaining
            expected = expected_bytes[start:start + count]
            equal = (_memcmp(scratch.ctypes.data, expected.ctypes.data, count) == 0 if _memcmp is not None
                     else np.array_equal(scratch[:count], expected))
            if not equal:
                raise OSError("FITS pixel verification failed")
            remaining -= count
    if got is not None and got.hexdigest() != digest:
        raise OSError("FITS pixel verification failed")


def coverage_chunks(coverage, check=lambda: None):
    for a, b in row_blocks(*coverage.shape):
        check()
        yield coverage[a:b]


def stack_chunks(img, offset, convert, check=lambda: None):
    for channel in range(3):
        for a, b in row_blocks(*img.shape[:2]):
            check()
            yield convert(img[a:b, :, channel], offset)[0]


def clipping_counts(img, offset, convert, check=lambda: None):
    low = high = 0
    for channel in range(3):
        for a, b in row_blocks(*img.shape[:2]):
            check()
            _, lo, hi = convert(img[a:b, :, channel], offset)
            low += lo
            high += hi
    return low, high
