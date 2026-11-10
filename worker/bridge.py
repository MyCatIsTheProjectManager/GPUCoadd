"""The GPU worker the GPUCoadd dialogs start: one job (--request) or a session of jobs
(--session), JSON events on stdout."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
import csv
from collections import Counter, deque
from pathlib import Path
import sys
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker.failure_report import build_failure, save_failure
from source_integrity import SourceGuard, SourceFileError, file_stamp

SCHEMA = 2  # 2: analysis.json carries overlap normalization
RUN_KEEP_DAYS, RUN_KEEP_COUNT = 30, 100  # GPUCoadd's own run folders (heat maps, run reports)
LEFTOVER_DAYS = 1  # request files and worker folders left by interrupted runs
# Latin letters Unicode has no decomposition for, so fits_text can't just drop an accent
LATIN_LETTERS = str.maketrans(
    {"đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE"}
)


def work_root(request=None):
    """GPUCoadd's own working area, kept out of the user's output folder: request files, worker
    sessions, heat maps (alignment caches) and run reports. %LOCALAPPDATA%\\GPUCoadd on Windows,
    where the GPU kernel caches also live; a request's "work_dir" overrides it (benchmarks, tests)."""
    if request and request.get("work_dir"):
        return Path(request["work_dir"]).resolve()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "GPUCoadd"


def prune_work(root, keep=()):
    """Delete GPUCoadd's old working files under `root` (never the output folder): run folders
    older than RUN_KEEP_DAYS or beyond the newest RUN_KEEP_COUNT, except those holding a path in
    `keep` (e.g. the heat map a stack is using), and request files and worker folders older than
    LEFTOVER_DAYS (left by interrupted runs; finished ones are removed by the dialog and worker)."""
    import shutil

    now = time.time()
    keep = {Path(k).resolve().parent for k in keep if k}
    runs = root / "runs"
    if runs.is_dir():
        folders = sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for k, p in enumerate(folders):
            if p.resolve() not in keep and (k >= RUN_KEEP_COUNT or now - p.stat().st_mtime > RUN_KEEP_DAYS * 86400):
                shutil.rmtree(p, ignore_errors=True)
    for sub in ("requests", "sessions"):
        if (root / sub).is_dir():
            for p in (root / sub).iterdir():
                if now - p.stat().st_mtime > LEFTOVER_DAYS * 86400:
                    if p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        p.unlink(missing_ok=True)


class Cancelled(Exception):
    pass


def filename_text(text):
    """`text` for a file name: letters, marks and digits of any script stay, the rest becomes _."""
    text = unicodedata.normalize("NFC", text)
    kept = "".join(c if c in "._-" or unicodedata.category(c)[0] in "LMN" else " " for c in text)
    return re.sub(" +", "_", kept).strip("._-")


def fits_text(text):
    """`text` for a FITS header, which holds printable ASCII only: accents are dropped, letters of
    other scripts can't be kept, and a name with nothing left becomes Unknown_Target."""
    plain = unicodedata.normalize("NFKD", text.translate(LATIN_LETTERS))
    return plain.encode("ascii", "ignore").decode("ascii").strip("._- ") or "Unknown_Target"


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def fingerprint(paths, settings, reference, source_guard=None):
    """paths: the lights used (after group selection), in stacking order."""
    # grouping decides which lights are used, their pedestals and expected scales: part of the engine
    engine = hashlib.sha256(
        b"".join(
            (ROOT / f).read_bytes()
            for f in (
                "gpucoadd.py",
                "raw_storage.py",
                "fits_input.py",
                "source_integrity.py",
                "resource_guard.py",
                "output_store.py",
                "fits_output.py",
                "overlap.py",
                "sky_store.py",
                "memory_plan.py",
                "overlap_kernels.py",
                "lightgroups.py",
                "rcd_fused.py",
                "rcd_kernels.py",
                "cuda_kernels.py",
                "integrate_kernels.py",
                "sort_kernels.py",
                "warp_kernels.py",
                "calibrate_kernels.py",
                "tileplan.py",
                "registration_fast.py",
                "feather_kernels.py",
            )
        )
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "engine": engine,
        "files": [
            {
                "path": str(p.resolve()),
                **(source_guard.stamps[str(p.absolute())] if source_guard is not None else file_stamp(p)),
            }
            for p in paths
        ],
        "selection": settings.get("select", 0),
        "reference": reference,
        "transform": settings.get("transform", "similarity"),
    }


def quality_settings(settings):
    choices = {
        "debayer": ("rcd", "bilinear"),
        "interp": ("bicubic", "bilinear"),
        "norm": ("spread", "photometric", "overlap"),
        "weight": ("none", "noise"),
        "transform": ("similarity", "homography"),
        "rejection": ("sigma", "winsorized", "noise"),
    }
    result = {}
    for key, values in choices.items():
        result[key] = settings.get(key, values[0])
        if result[key] not in values:
            raise ValueError(f"Unknown {key} setting: {result[key]}")
    selection = float(settings.get("select", 0))
    if not math.isfinite(selection) or selection < 0:
        raise ValueError("Sharpness limit must be zero (off) or positive")
    result["select"] = selection
    feather = float(settings.get("feather", 0))
    if not math.isfinite(feather) or not 0 <= feather <= 1000:
        raise ValueError("Feather must be 0 (off) to 1000 pixels")
    result["feather"] = feather  # stacking only: not part of the heat map's fingerprint
    result["rgb_equal"] = settings.get("rgb_equal", False)
    if type(result["rgb_equal"]) is not bool:
        raise ValueError("RGB equalization must be true or false")
    if result["rgb_equal"] and result["norm"] != "spread":
        raise ValueError("RGB equalization requires Background spread brightness")
    result["match_gradients"] = settings.get("match_gradients", False)  # stacking only; checked against norm after
    if type(result["match_gradients"]) is not bool:  # the automatic mixed-instrument choice
        raise ValueError("Gradient matching must be true or false")
    return result


def output_settings(settings):
    result = {
        key: settings.get(key, False)
        for key in ("stack_preview", "coverage_fits", "coverage_png", "frames_csv", "aligned_lights", "debayered_lights", "activity_log")
    }
    if any(type(value) is not bool for value in result.values()):
        raise ValueError("Output-file choices must be true or false")
    return result


ADU_SCALE = 2 * 65535.0  # stacked FITS: value = (ADU + offset) / ADU_SCALE, 2x headroom above 16 bits


def stack_pixels(img, offset=0.0, *, count_clipping=True):
    """The stack as written to FITS, in 0..1 as image processing programs expect (some otherwise
    rescale each file from its own min..max on opening): (ADU + offset) / ADU_SCALE, with the
    reference light's camera offset added back so no real pixel reaches 0 (0 means "no data" to
    many stacking and processing tools), and 2x headroom for star cores that brightness matching
    lifts above 65535. Uncovered pixels (NaN) are 0. Returns (pixels, samples clipped low,
    samples clipped high); both counts are expected to be 0. Per-light exports do not
    report these counts and can skip computing them."""
    import numpy as np

    out = (img + np.float32(offset)) / np.float32(ADU_SCALE)
    finite = np.isfinite(out)
    low = high = 0
    if count_clipping:
        low, high = int(np.count_nonzero(finite & (out <= 0))), int(np.count_nonzero(finite & (out > 1)))
    np.clip(out, np.float32(1 / ADU_SCALE), 1, out=out, where=finite)  # a covered pixel never becomes 0
    if not finite.all():
        # Most export pixels are finite. Avoid nan_to_num's three full-size masks
        # and copies just to fill the uncovered border; preserve infinities too.
        invalid = ~finite
        out[invalid] = np.nan_to_num(out[invalid], copy=False, nan=0.0)
    return out.astype(np.float32, copy=False), low, high


def nearest_light(mats, indices, sizes, scales, centre):
    """The light (among indices) whose sensor centre lies nearest `centre` (reference-frame
    pixels), compared on the sky (light pixels x its arcsec/px when known). mats: dst
    (reference) -> src (light) 3x3, None for unaligned lights."""
    import numpy as np

    best, best_d = None, math.inf
    for i in indices:
        if mats[i] is None:
            continue
        p = np.asarray(mats[i], dtype=float) @ (centre[0], centre[1], 1.0)
        w, h = sizes[i]
        d = math.hypot(p[0] / p[2] - w / 2, p[1] / p[2] - h / 2) * (scales[i] or 1.0)
        if d < best_d:
            best, best_d = i, d
    return best


OUTPUT_HEADER_KEYS = (
    "OBJECT",
    "EXPTIME",
    "EXPOSURE",
    "DATE-OBS",
    "FOCALLEN",
    "XPIXSZ",
    "YPIXSZ",
    "XBINNING",
    "YBINNING",
    "FILTER",
    "TELESCOP",
    "INSTRUME",
    "SITELAT",
    "SITELONG",
    "SITEELEV",
)


def header_metadata(header):
    """Small, typed snapshot of output metadata; do not retain whole FITS histories."""
    return {key: header[key] for key in OUTPUT_HEADER_KEYS if key in header}


def observation_header(hdr, paths, indices, ref, centre, pointing, *, headers=None):
    """Observation metadata for plate solving and colour calibration (e.g. Siril's
    plate solver and SPCC), from the lights' own headers: RA/DEC of the light nearest the image
    centre (a search seed, not a solution), the optics of the reference light (whose pixel grid
    the output uses), the time span of the contributing lights, and filter, telescope and site
    when all contributing lights agree (else the reference light's, marked)."""
    from astropy.io import fits

    heads = {i: fits.getheader(paths[i]) if headers is None else headers[i] for i in set(indices) | {ref}}
    ra, dec = pointing[centre] if centre is not None else (float("nan"), float("nan"))
    if math.isfinite(ra) and math.isfinite(dec):
        hdr["RA"] = (ra, "[deg] seed: light nearest the image centre")
        hdr["DEC"] = (dec, "[deg] seed: light nearest the image centre")
    for key in ("FOCALLEN", "XPIXSZ", "YPIXSZ", "XBINNING", "YBINNING"):
        if key in heads[ref]:
            hdr[key] = (heads[ref][key], "reference light (output pixel grid)")
    starts = []
    for i in indices:
        try:
            start = datetime.fromisoformat(str(heads[i].get("DATE-OBS", "")).strip())
        except ValueError:
            continue
        exposure = heads[i].get("EXPTIME", heads[i].get("EXPOSURE"))
        try:
            seconds = float(exposure)
            seconds = seconds if math.isfinite(seconds) and seconds > 0 else 0
            end = start + timedelta(seconds=seconds)
        except (TypeError, ValueError, OverflowError):
            end = start  # unknown duration; never invent an end time or fail the stack's save
        starts.append((start, end))
    if starts:
        hdr["DATE-OBS"] = (min(s for s, _ in starts).isoformat(), "start of the first contributing light")
        hdr["DATE-END"] = (max(e for _, e in starts).isoformat(), "end of the last contributing light")
    for key in ("FILTER", "TELESCOP", "INSTRUME", "SITELAT", "SITELONG", "SITEELEV"):
        values = {str(heads[i][key]).strip() if key in heads[i] else None for i in indices}
        if len(values) == 1 and None not in values:
            hdr[key] = heads[ref][key] if key in heads[ref] else heads[indices[0]][key]
        elif key in heads[ref]:
            hdr[key] = (heads[ref][key], "reference light; the lights differ")


def save_image_outputs(
    folder,
    img,
    coverage,
    hdr,
    basename,
    frames,
    options,
    predicted=False,
    offset=0.0,
    clipping=None,
    check_cancel=None,
    before_publish=None,
    warn=None,
):
    """Export only selected extras; a real stack always retains its FITS image. `folder` is the
    user's output folder, created only when there is something to write. offset: the reference
    light's camera offset (stack_pixels); clipping: a dict that receives the clip counts.
    warn(key, exc, what): told about an extra that could not be saved (see Reporter.warn), so
    the job carries on; without it the failure is raised."""
    import numpy as np
    from gpucoadd import coverage_preview, preview
    from fits_output import write_fits, coverage_chunks, stack_chunks, clipping_counts
    from resource_guard import ResourcePressureError

    check = check_cancel or (lambda: None)
    files = {}

    def save_extra(key, path, write, what):
        try:
            write(path)
            files[key] = str(path)
        except (Cancelled, SourceFileError, ResourcePressureError):  # about the job, not this file
            raise
        except Exception as exc:
            if warn is None:
                raise
            warn(key, exc, what)

    # FITS standard row order: the first stored row is the bottom of the image, so the canvas (top
    # row first, like the lights) is written last row first. ROWORDER says so for readers that
    # otherwise assume top-down; readers that follow the standard show it the right way up.
    hdr = hdr.copy()
    hdr["ROWORDER"] = ("BOTTOM-UP", "FITS standard: first stored row is the bottom")
    if img is not None or options["coverage_fits"] or options["coverage_png"]:
        folder.mkdir(parents=True, exist_ok=True)
    if options["coverage_fits"]:
        coverage_header = hdr.copy()
        coverage_header["PRODTYPE"] = "COVERAGE"
        coverage_header["BUNIT"] = "frames"
        save_extra(
            "coverage_fits",
            folder / (basename + "_coverage.fits"),
            lambda path: write_fits(
                path,
                coverage_header,
                coverage.shape,
                np.uint32,
                coverage_chunks(coverage[::-1], check),
                check,
                before_publish,
            ),
            "The frame count map (FITS)",
        )
    if options["coverage_png"]:
        save_extra(
            "coverage_png",
            folder / (basename + "_coverage.png"),
            lambda path: coverage_preview(coverage, frames, str(path), predicted=predicted),
            "The frame count heat map (PNG)",
        )
    if img is not None:
        path = folder / (basename + ".fits")
        stack_header = hdr.copy()
        stack_header["PRODTYPE"] = "STACK"
        low, high = clipping_counts(img, offset, stack_pixels, check)
        stack_header["ADUSCALE"] = (ADU_SCALE, "ADU = value x ADUSCALE - ADUOFFS")
        stack_header["ADUOFFS"] = (float(offset), "camera offset of the reference light, ADU")
        stack_header["HISTORY"] = "Pixels: (ADU + ADUOFFS) / ADUSCALE in 0..1; uncovered pixels are 0"
        if low or high:
            stack_header["HISTORY"] = f"Clipped samples: {low} low, {high} high"
        if clipping is not None:
            clipping.update(low=low, high=high)
        write_fits(
            path,
            stack_header,
            (3, *img.shape[:2]),
            np.float32,
            stack_chunks(img[::-1], offset, stack_pixels, check),
            check,
            before_publish,
        )
        files["stack_fits"] = str(path)
        if options["stack_preview"]:
            save_extra(
                "stack_preview",
                folder / (basename + "_preview.png"),
                lambda path: preview(img, str(path)),
                "The stack preview (PNG)",
            )
    for path in files.values():
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise OSError(f"Output was not saved successfully: {path}")
    return files


# The optional per-light exports: output key -> (file suffix, name in messages).
# debayered_lights is experimental and not shown in the dialogs.
LIGHT_COPIES = {"aligned_lights": ("aligned", "aligned lights"), "debayered_lights": ("debayered", "debayered lights")}


def light_copy_name(path, used, suffix):
    """File name of a light's copy; lights from two folders can share a name."""
    stem = Path(path).stem
    name, n = f"{stem}_{suffix}.fits", 1
    while name.casefold() in used:
        n += 1
        name = f"{stem}_{n}_{suffix}.fits"
    used.add(name.casefold())
    return name


def light_copy_header(aligned, hdr, light, path, ref, index, paths, pointings, headers, offset):
    """One light's own observation details, on the stack's pixel grid (aligned) or its own."""
    from astropy.io import fits

    header = fits.Header()
    header["PRODTYPE"] = "ALIGNED" if aligned else "DEBAYERED"
    if aligned:
        for key in ("FRAMING", "ORIGINX", "ORIGINY", "RGBEQUAL", "GRADMTCH"):
            header[key] = (hdr[key], hdr.comments[key])
    header["NCOMBINE"] = 1
    observation_header(header, paths, [index], ref if aligned else index, index, pointings, headers=headers)
    for key in ("OBJECT", "EXPTIME", "EXPOSURE", "FILTER", "TELESCOP", "INSTRUME", "SITELAT", "SITELONG", "SITEELEV"):
        if key in light:
            header[key] = fits_text(light[key]) if key == "OBJECT" else light[key]
    header["SRCLIGHT"] = (fits_text(Path(path).name)[:68], "source light")
    header["ROWORDER"] = ("BOTTOM-UP", "FITS standard: first stored row is the bottom")
    header["ADUSCALE"] = (ADU_SCALE, "ADU = value x ADUSCALE - ADUOFFS")
    header["ADUOFFS"] = (float(offset), f"camera offset of the {'reference ' if aligned else ''}light, ADU")
    header["HISTORY"] = (
        "Warped and brightness-matched as stacked, before rejection and weights"
        if aligned
        else "Debayered as stacked, in the light's own position: not aligned or brightness-matched"
    )
    header["HISTORY"] = "Pixels: (ADU + ADUOFFS) / ADUSCALE in 0..1; pixels without data are 0"
    return header


def plan_light_copies(out_dir, key, sizes, reserved, reporter):
    """Whether these copies fit on the results drive, before any work starts; they are only an
    extra, so if not the stack goes ahead without them. sizes: bytes per light, an upper limit as
    some lights may not end up contributing to the selected area. reserved: bytes already planned
    for that drive."""
    from output_store import GIB, check_space

    what = LIGHT_COPIES[key][1]
    try:
        check_space([(out_dir, reserved + sum(sizes))])
    except OSError as exc:
        reporter.warn(key, exc, f"The {what} export")
        return False
    reporter.message(
        f"{what.capitalize()}: up to {len(sizes)} FITS files of {max(sizes) / 2**20:.0f} MiB each, "
        f"{sum(sizes) / GIB:.1f} GiB in total, in the results folder."
    )
    return True


def own_position(i, width, height):
    """render_light's state for a light where it lies on its own sensor: the stack's demosaicing
    with no movement (cubic weights at whole pixels are 1 and 0) and no brightness change."""
    import numpy as np

    one = (np.ones(3, np.float32), np.zeros(3, np.float32))
    return {
        "mats": {i: np.eye(3)},
        "coefs": {i: one},
        "origin": [0, 0],
        "width": width,
        "height": height,
        "sizes": {i: (width, height)},
    }


def save_light_copies(
    key, out_dir, basename, be, info, hdr, paths, pointings, headers, pedestals, workspace, reporter, *,
    overlap=True, verify_exact=True, gpu_format=True
):
    """One optional export, one FITS per contributing light in a folder of its own, after the stack
    is safely saved. A failure stops it with a warning and the job carries on; the finished files
    stay."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from functools import partial
    from threading import Event

    import numpy as np
    from fits_output import stack_chunks, write_encoded, write_fits
    from gpucoadd import ALIGNED_TILE, render_light
    from resource_guard import ResourcePressureError, check_ram

    suffix, what = LIGHT_COPIES[key]
    aligned = key == "aligned_lights"
    ref = info["ref_idx"]
    indices = info["contributing_indices"]
    folder = out_dir / f"{basename}_{suffix}"
    saved, size, tile, used = 0, 0, ALIGNED_TILE, set()
    render_s = write_s = 0.0
    buffers = {}
    # Disk-backed canvases already compete with FITS writes on the work drive.
    overlap = overlap and workspace.mode == "ram" and len(indices) > 1
    writer = None
    verify_buffer = None
    encoder = None
    stopped = Event()
    pending = None
    convert = partial(stack_pixels, count_clipping=False)
    stage = f"saving {what}"
    reporter.progress(stage, 0, len(indices))
    export_started = time.perf_counter()

    def check_write():
        # Resource relief and source checks may touch the GPU: keep Reporter on
        # the main thread. The writer only observes cancellation and shutdown.
        if writer is None:
            reporter.check_cancel()
        if stopped.is_set() or reporter.cancel_path.exists():
            raise Cancelled("Cancellation requested")

    def write_light(canvas, path, header, offset):
        started = time.perf_counter()
        write_fits(
            path, header, (3, canvas.shape[0], canvas.shape[1]), np.float32,
            stack_chunks(canvas[::-1], offset, convert, check_write), check_write,
            None, verify_buffer,
        )
        return path.stat().st_size, time.perf_counter() - started

    def write_formatted(body, shape, path, header):
        started = time.perf_counter()
        write_encoded(path, header, (3, shape[0], shape[1]), body, check_write)
        return path.stat().st_size, time.perf_counter() - started

    def collect():
        nonlocal pending, saved, size, write_s
        while True:
            reporter.check_cancel()
            try:
                count, seconds = pending.result(timeout=0.1)
                break
            except TimeoutError:
                if pending.done():
                    # Distinguish a wait timeout racing completion from a
                    # TimeoutError raised by the writer itself.
                    count, seconds = pending.result()
                    break
        pending = None
        saved += 1
        size += count
        write_s += seconds
        reporter.progress(stage, saved, len(indices))

    try:
        try:
            if verify_exact and workspace.mode == "ram" and indices:
                state = info["light_render"]
                sizes = ([state["width"] * state["height"] * 12] if aligned else
                         [state["sizes"][i][0] * state["sizes"][i][1] * 12 for i in indices])
                largest = max(sizes)
                if gpu_format:
                    from export_kernels import open_encoder

                    encoder = open_encoder(
                        be, largest // 12, 2 if overlap else 1, ADU_SCALE, np.float32(1 / ADU_SCALE), reporter.message
                    )
                # The encoder's host buffers hold the bytes it writes, so they are the read-back too
                if encoder is None and check_ram(largest * (3 if overlap else 2), "light export verification"):
                    try:
                        verify_buffer = workspace.allocate(f"{suffix}_verify", (largest,), np.uint8)
                    except ResourcePressureError:
                        raise
                    except MemoryError:  # optional acceleration: retain streaming SHA-256
                        verify_buffer = None
                    if workspace.mode != "ram":
                        verify_buffer = None
                        overlap = False
            if overlap:
                writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="light-export")
            for k, i in enumerate(indices, 1):
                reporter.check_cancel()
                if pending is not None and pending.done():
                    collect()
                state = info["light_render"] if aligned else own_position(i, *info["light_render"]["sizes"][i])
                shape = (state["height"], state["width"], 3)
                slot = (k - 1) % 2 if overlap else 0
                buffer_key = (slot, shape)
                if encoder is not None:
                    canvas = encoder.canvas(shape[0], shape[1])
                elif buffer_key not in buffers:
                    buffers[buffer_key] = workspace.allocate(f"{suffix}_{len(buffers)}", shape, np.float32)
                if encoder is None:
                    canvas = buffers[buffer_key]
                if writer is not None and workspace.mode != "ram":
                    # An automatic allocation may have fallen back to disk.
                    # Drain before using that canvas, then continue serially.
                    if pending is not None:
                        collect()
                    writer.shutdown(wait=True)
                    writer = None
                    overlap = False
                started = time.perf_counter()
                tile = render_light(be, i, state, canvas, tile)
                offset = pedestals[ref if aligned else i]
                # the host buffer of this slot is free: the writer can only hold the previous one
                body = None if encoder is None else encoder.encode(shape[0], shape[1], offset, slot)
                rendered = time.perf_counter()
                render_s += rendered - started
                path = folder / light_copy_name(paths[i], used, suffix)
                header = light_copy_header(aligned, hdr, headers[i], paths[i], ref, i, paths, pointings, headers, offset)
                # This canvas is distinct from the one the writer owns. Finish the
                # preceding file before handing off this one: at most one pending write.
                if pending is not None:
                    collect()
                reporter.check_cancel()
                job = (write_light, canvas, path, header, offset) if encoder is None else (
                    write_formatted, body, shape, path, header
                )
                if writer is None:
                    count, seconds = job[0](*job[1:])
                    saved += 1
                    size += count
                    write_s += seconds
                    reporter.progress(stage, saved, len(indices))
                else:
                    pending = writer.submit(*job)
            if pending is not None:
                collect()
        finally:
            stopped.set()
            if writer is not None:
                writer.shutdown(wait=True, cancel_futures=True)
            # A render/cancel failure can arrive after the preceding file was published.
            # Count that completed file before reporting the partial export.
            if pending is not None and not pending.cancelled():
                try:
                    count, seconds = pending.result()
                except BaseException:
                    pass  # preserve the original main-thread or writer exception
                else:
                    saved += 1
                    size += count
                    write_s += seconds
    except (Cancelled, SourceFileError, ResourcePressureError):  # about the job, not these files
        raise
    except Exception as exc:
        reporter.warn(key, exc, f"Part of the {what} ({saved} of {len(indices)} saved)")
    finally:
        formatted = encoder is not None
        encoder = canvas = body = None  # the writer has joined: its buffers go before the GPU cache
        be.torch.cuda.empty_cache()
    return {
        "folder": str(folder),
        "saved": saved,
        "total": len(indices),
        "bytes": size,
        "render_s": render_s,
        "write_s": write_s,
        "elapsed_s": time.perf_counter() - export_started,
        "overlap": overlap,
        "formatting": "gpu" if formatted else "cpu",
        "verification": "exact" if verify_buffer is not None or formatted else "sha256",
        "tile": tile,
    }


def image_output_manifest(files):
    roles = {
        "stack_fits": ("Stacked image", "FITS"),
        "stack_preview": ("Stack preview", "PNG"),
        "coverage_fits": ("Frame count map", "FITS"),
        "coverage_png": ("Frame count heat map", "PNG"),
    }
    return [
        {"role": key, "label": label, "format": fmt, "path": files[key]}
        for key, (label, fmt) in roles.items()
        if key in files
    ]


def output_warning(output, exc, what):
    """Why an optional output (activity log, frames CSV) was not saved. An OSError is the
    environment refusing it (permission, full drive, lock, disconnected folder); anything else
    is a fault in GPUCoadd's own code, reported as such rather than blamed on the setup."""
    environment = isinstance(exc, (OSError, MemoryError))
    return {
        "output": output,
        "kind": "environment" if environment else "internal",
        "message": f"{what} was not saved"
        + (
            f": {exc}. Closing other programs lets it be saved next time."
            if isinstance(exc, MemoryError)
            else f": {exc}"
            if isinstance(exc, OSError) and str(exc).startswith("Not enough free disk space")
            else ": the folder or drive refused it (permission, free space, a lock or a disconnected drive)."
            if environment
            else ": this looks like a GPUCoadd bug; please share the run report."
        )
        + " The rest of the result is unaffected.",
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_optional(path, write):
    """Write an optional file through a temporary file in its folder, published only when
    complete, so a failure never leaves a partial file under the final name."""
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write(temp)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def adopt_kept_lights(be, reporter):
    """A kept set reused by this job: its messages and storage notices now go to this job, and
    lights released to file reads during an earlier job are read back into RAM while RAM allows
    (between jobs only; see GPUBackend.restore_ram)."""
    be.log, be.storage_notice = reporter.message, reporter.storage
    restored = be.restore_ram() if hasattr(be, "restore_ram") else 0
    if restored:
        reporter.message(f"RAM is free again: {restored} lights that were read from their files are held in RAM again.")
    return restored


def save_optional(files, key, path, write, reporter, what):
    """Save an optional output; listed in `files` only when saved, else a warning (see Reporter.warn)."""
    try:
        write_optional(path, write)
        files[key] = str(path)
    except Exception as exc:
        reporter.warn(key, exc, what)


def image_identity(paths, indices, generated=None, *, headers=None):
    """Name by real contributing lights; do not invent target/exposure metadata."""
    from astropy.io import fits

    selected_headers = [fits.getheader(paths[i]) if headers is None else headers[i] for i in indices]
    targets = [str(h.get("OBJECT", "")).strip() for h in selected_headers if str(h.get("OBJECT", "")).strip()]
    if targets:
        target = Counter(targets).most_common(1)[0][0] if len(set(targets)) == 1 else "Mixed_Targets"
    else:
        parent = paths[0].parent
        while (
            parent.name.casefold() in ("lights", "light", "subs", "s50pro", "s50", "seestar s50")
            and parent.parent != parent
        ):
            parent = parent.parent
        target = parent.name or "Unknown_Target"
    target = filename_text(target)[:80] or "Unknown_Target"
    exposures = []
    dates = []
    for h in selected_headers:
        try:
            exposure = float(h.get("EXPTIME", h.get("EXPOSURE", float("nan"))))
        except (ValueError, TypeError):
            exposure = float("nan")
        exposures.append(exposure if math.isfinite(exposure) and exposure > 0 else None)
        date = str(h.get("DATE-OBS", ""))[:10]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
            dates.append(date)
    common = (
        exposures[0] if exposures and all(e is not None and abs(e - exposures[0]) < 1e-6 for e in exposures) else None
    )
    total = sum(exposures) if exposures and all(e is not None for e in exposures) else None
    if common is not None:
        exposure_tag = f"{len(indices)}x{common:g}sec"
    elif total is not None:  # e.g. S50 Pro 30 s + S50 10 s: 230x30sec+210x10sec, largest total first
        counts = Counter(round(e, 6) for e in exposures)
        exposure_tag = "+".join(
            f"{c}x{e:g}sec" for e, c in sorted(counts.items(), key=lambda ec: (-ec[0] * ec[1], -ec[0]))
        )
    else:
        exposure_tag = f"{len(indices)}frames"
    generated = generated or datetime.now().astimezone()
    observation_dates = sorted(set(dates))
    date_tag = (
        (observation_dates[0] + ("_" + observation_dates[-1] if len(observation_dates) > 1 else "") + "_")
        if observation_dates
        else ""
    )
    basename = f"{target}_{exposure_tag}_{date_tag}{generated:%Y-%m-%d_%H%M%S}_gpucoadd"
    return {
        "basename": basename,
        "target": target,
        "frames": len(indices),
        "exposure_seconds": common,
        "summed_input_exposure_s": total,
        "observation_dates": observation_dates,
        "generated_at": generated.isoformat(),
    }


# Plain explanations logged when a stage starts, so a step that appears only for some sets
# (mosaics, two telescopes, Overlap matching) is not a mystery.
STAGE_NOTES = (
    (
        "linking light groups",
        "The lights of the other light groups are aligned to the reference through one well-matched light per group.",
    ),
    (
        "neighbour alignment",
        "Some lights could not be matched to the reference directly, usually because they share little sky with "
        "it (mosaic panels, drift). They are aligned through overlapping lights that are already aligned. This is "
        "normal and does not lower quality.",
    ),
    (
        "preparing overlap matching, step 1 ",
        "Overlap matching compares the lights where they overlap to balance their brightness. It runs in "
        "numbered steps; each step's progress starts again from zero.",
    ),
    (
        "saving aligned lights",
        "The stack is already saved. Each light is now written to its own FITS file on the stack's pixels.",
    ),
    (
        "saving debayered lights",
        "The stack is already saved. Each light is now written to its own colour FITS file, not aligned.",
    ),
)


class Reporter:
    def __init__(self, folder, cancel_path, save_log=True):
        self.start = time.perf_counter()
        self.stage = None
        self.stage_start = self.start
        self.stage_times = {}
        self.recent_events = deque(maxlen=80)
        self.source_guard = None
        self.last_source_check = 0
        self.source_check_interval = 5
        self.last_resource_check = 0
        self.resource_check = None
        self.storage_notices = {}
        self.last_logged = -1
        self.cancel_path = Path(cancel_path)
        self.warnings = []
        self.log_file = open(folder / "activity.jsonl", "a", encoding="utf-8", buffering=1) if save_log else None

    def check_cancel(self):
        if self.cancel_path.exists():
            raise Cancelled("Cancellation requested")
        if self.source_guard is not None and time.perf_counter() - self.last_source_check >= self.source_check_interval:
            self.validate_sources()
        if time.perf_counter() - self.last_resource_check >= 2:
            if self.resource_check is not None:
                self.resource_check()
            else:
                from resource_guard import check_ram

                check_ram(stage=self.stage or "worker preparation", relieve=True)
            self.last_resource_check = time.perf_counter()

    def validate_sources(self):
        if self.source_guard is not None:
            started = time.perf_counter()
            self.source_guard.validate()
            self.last_source_check = time.perf_counter()
            # Slow/large folders should not spend a sizeable fraction of the job on stat().
            # Mandatory checks before publication/cache updates are never throttled.
            self.source_check_interval = max(5, (self.last_source_check - started) * 100)

    def emit(self, event_type, /, live=True, **fields):
        """One event to the interface (stdout) and the activity log; live=False: log only.
        Positional-only type, so event fields may themselves be called "kind"."""
        event = {
            "type": event_type,
            "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": time.perf_counter() - self.start,
            **fields,
        }
        self.recent_events.append(event)
        line = json.dumps(safe_json(event), allow_nan=False)
        if live:
            print(line, flush=True)
        if self.log_file is not None:
            try:
                self.log_file.write(line + "\n")
            except OSError as exc:  # the log is optional: stop writing it, never fail the run
                self.close_log()
                self.warn("activity_log", exc, "The activity log")

    def close_log(self):
        log, self.log_file = self.log_file, None
        if log is not None:
            try:
                log.close()
            except OSError:
                pass

    def warn(self, output, exc, what):
        """An optional output was not saved: a warning, never a change of the run's outcome."""
        warning = output_warning(output, exc, what)
        self.warnings.append(warning)
        self.emit("warning", **warning)
        return warning

    def message(self, text):
        self.emit("log", message=text)

    def storage(self, component, mode, reason, **details):
        notice = {"component": component, "mode": mode, "reason": reason, **details}
        self.storage_notices[component] = notice
        self.emit("storage", **notice)
        if component == "output":  # the last decision of a job: say where everything is held
            from memory_plan import plan_summary

            self.message(plan_summary(self.storage_notices))

    def progress(self, stage, done, total):
        self.check_cancel()
        now = time.perf_counter()
        if stage != self.stage:
            if self.stage is not None:
                duration = now - self.stage_start
                self.stage_times[self.stage] = self.stage_times.get(self.stage, 0) + duration
                self.emit("stage_finished", stage=self.stage, duration_s=duration)
            self.stage, self.stage_start = stage, now
            self.last_logged = -1
            note = next((text for prefix, text in STAGE_NOTES if stage.startswith(prefix)), None)
            self.emit("stage_started", stage=stage, message=f"Started {stage}" + (f". {note}" if note else ""))
        elapsed = now - self.stage_start
        # This is a stage estimate only. Tile/frame costs vary across the field.
        eta = elapsed * (total - done) / done if done >= 3 and elapsed >= 1 else None
        self.emit("progress", stage=stage, done=done, total=total, stage_elapsed_s=elapsed, stage_eta_s=eta)
        percent = int(100 * done / total) if total else 0
        if done == total or percent >= self.last_logged + 10:
            self.last_logged = percent
            self.message(
                f"{stage}: {done}/{total}" + (f"; estimated {eta:.0f}s left in this stage" if eta is not None else "")
            )


def shared_sky_cached(analysis, select):
    """Whether a heat map's cache holds an Overlap matching solution stack() can use (the same test)."""
    import overlap
    from gpucoadd import retained_lights

    norm = analysis.get("normalization")
    return (
        norm is not None
        and norm.get("version") == overlap.VERSION
        and norm.get("reference") == analysis.get("ref_idx")
        and set(map(str, retained_lights(analysis, select))) <= set(norm.get("a", {}))
    )


def light_folders(request):
    """One folder ("lights_dir", as before) or several ("lights_dirs")."""
    folders = request.get("lights_dirs") or ([request["lights_dir"]] if request.get("lights_dir") else [])
    if not folders or not all(isinstance(f, str) and f for f in folders):
        raise ValueError("Select at least one light folder")
    for f in folders:  # Seestar's "<target>_sub" folders hold thumbnails or copies of the lights
        if Path(f).name.casefold().endswith("_sub"):
            raise ValueError(
                f'Seestar "_sub" folders hold copies of the lights and are never read: {f}. '
                'Choose the target folder or its "lights" folder.'
            )
    return folders


def scan_lights(request, reporter):
    """Read every light's header once: groups, which lights are used, pedestals, pointings."""
    from astropy.io import fits
    import lightgroups

    paths = lightgroups.light_paths(light_folders(request))
    guard = SourceGuard(paths)
    reporter.message(f"Reading {len(paths)} light headers (instrument, filter, exposure, pointing).")
    lights = []
    for p in paths:
        reporter.check_cancel()
        header = guard.read(p, fits.getheader)
        light = lightgroups.describe(p, header)
        light["output_header"] = header_metadata(header)
        lights.append(light)
    exclude = list(request.get("exclude_groups") or [])
    known = {lightgroups.group_key(light) for light in lights}
    stale = [k for k in exclude if k not in known]
    if stale:  # a choice saved before the files changed: report it, do not fail
        reporter.message(f"Ignoring {len(stale)} left-out group(s) that no longer match any lights.")
    grouping = lightgroups.plan(lights, [k for k in exclude if k in known])
    for g in grouping["groups"]:
        state = "main group" if g["main"] else "used" if g["used"] else "left out: " + g["reason"]
        reporter.message(
            f"Light group: {g['label']}; pedestal {', '.join(f'{v:g}' for v in g['pedestal'])} "
            f"({g['pedestal_source']}); {state}."
        )
        if g["used"] and g["conversion"]:
            reporter.message(f"Reading {g['label']}: {g['bayer']}, " + "; ".join(g["conversion"]) + ".")
    if not grouping["used"]:
        raise ValueError("No light group is selected for stacking")
    reporter.source_guard = guard.select([lights[i]["path"] for i in grouping["used"]])
    reporter.validate_sources()
    return lights, grouping


def group_names(groups):
    """Short readable name per group key for logs and the engine ("S50 Pro IRCUT 30s"), made
    unique with the folder name (then a number) when two groups would read the same."""

    def short(g):
        exp = "" if g["exposure_s"] is None else f" {g['exposure_s']:g}s"
        return f"{g['telescope']} {g['filter']}{exp}"

    count = Counter(short(g) for g in groups)
    names, seen = {}, set()
    for g in groups:
        name = (
            short(g)
            if count[short(g)] == 1
            else f"{short(g)} ({Path(g['folder']).parent.name}/{Path(g['folder']).name})"
        )
        base, k = name, 2
        while name in seen:
            name, k = f"{base} #{k}", k + 1
        seen.add(name)
        names[g["key"]] = name
    return names


def light_weight(value):
    """A light's weight as one number for reports: its green-channel weight when the engine
    weighted per channel (the full per-channel values stay in the report's `weights`)."""
    return float(value[1]) if isinstance(value, (list, tuple)) else float(value)


def weight_report(info):
    """Preserve full engine weights and their input-order indices in saved/live results.

    Indices refer to the selected inputs, not rows of a crop's loaded-light array.
    The CSV's existing scalar weight remains the green-channel summary.
    """
    return {
        "weights": info.get("weights"),
        "weighted_indices": info.get("weighted_indices"),
        "weights_per_channel": bool(info.get("weights_per_channel", False)),
    }


def group_report(grouping, used, info, operation):
    """Per-group rows for the report: the scan table plus what happened to the group's lights
    (aligned, sharpness, retained, contributing), its brightness term and, with noise weights,
    its mean weight and "main-group light equivalents" (its stacked lights' total weight in
    units of a mean main-group light: what the group adds, where it covers)."""
    pos = {i: k for k, i in enumerate(used)}  # input light -> stack index
    retained, dropped, failed = set(info["retained_indices"]), set(info["dropped"]), set(info["alignment_failed"])
    contributing = None if info.get("contributing_indices") is None else set(info["contributing_indices"])
    weight, rgb = {}, {}
    if info.get("weights") is not None and info.get("weighted_indices") is not None:
        weight = {i: light_weight(w) for i, w in zip(info["weighted_indices"], info["weights"])}
        if info.get("weights_per_channel"):
            rgb = {i: [float(v) for v in w] for i, w in zip(info["weighted_indices"], info["weights"])}
    terms = (info.get("normalization") or {}).get("group_terms") or {}
    main = next(g for g in grouping["groups"] if g["main"])
    main_w = [weight[pos[i]] for i in main["indices"] if i in pos and pos[i] in weight]
    unit = float(sum(main_w) / len(main_w)) if main_w else None
    main_rgb = [rgb[pos[i]] for i in main["indices"] if i in pos and pos[i] in rgb]
    unit_rgb = [sum(v[c] for v in main_rgb) / len(main_rgb) for c in range(3)] if main_rgb else None
    rows = []
    for g in grouping["groups"]:
        row = {
            k: g[k]
            for k in (
                "key",
                "label",
                "telescope",
                "filter",
                "exposure_s",
                "lights",
                "total_exposure_s",
                "main",
                "used",
                "reason",
                "pedestal",
                "pedestal_source",
            )
        }
        if g["used"]:
            k = [pos[i] for i in g["indices"]]
            row.update(
                aligned=sum(j not in failed for j in k),
                sharpness_rejected=sum(j in dropped for j in k),
                retained=sum(j in retained for j in k),
            )
            if contributing is not None:
                row["contributing"] = sum(j in contributing for j in k)
            name = group_names(grouping["groups"])[g["key"]]
            if name in terms:
                row["brightness_vs_star_prior"] = terms[name]["factor"]
            w = [weight[j] for j in k if j in weight]
            if w and unit:
                row["mean_weight_vs_main"] = float(sum(w) / len(w) / unit)
                row["main_light_equivalents"] = float(sum(w) / unit)
            wc = [rgb[j] for j in k if j in rgb]
            if (
                wc and unit_rgb and all(unit_rgb)
            ):  # per-channel weights: the group's mean per channel vs the main group's
                row["mean_weight_rgb_vs_main"] = [
                    float(sum(v[c] for v in wc) / len(wc) / unit_rgb[c]) for c in range(3)
                ]
        rows.append(row)
    return rows


def backend_geometry(shapes):
    """Bayer-mask allocation needed by a loaded subset, independent of its first light.

    Per-light dimensions remain in backend.sizes. Keep the largest mask dimensions
    in the reuse key: retaining larger masks could change the working-memory budget.
    """
    return max(h for h, w in shapes), max(w for h, w in shapes)


def execute(request, reporter, folder, session=None):
    if request.get("operation") == "scan":
        return _execute(request, reporter, folder, session, None)
    from output_store import OutputWorkspace

    with OutputWorkspace(
        folder,
        request["output_dir"],
        output_settings(request.get("outputs", {})),
        reporter.message,
        reporter.check_cancel,
        storage_notice=reporter.storage,
    ) as workspace:
        import resource_guard

        reporter.resource_check = workspace.check_resources
        # low RAM never stops a job: the lights this job holds in RAM can be read from their files
        # instead (registered once they are loaded), and otherwise one warning is logged
        resource_guard.set_handlers(warn=reporter.message)
        try:
            return _execute(request, reporter, folder, session, workspace)
        finally:
            reporter.resource_check = None
            pool = getattr(reporter, "registration_pool", None)
            if pool is not None:  # register() normally shut it down already
                pool.shutdown(wait=False, cancel_futures=True)
                reporter.registration_pool = None
            resource_guard.set_handlers()


def _execute(request, reporter, folder, session, workspace):
    """folder: this run's folder in GPUCoadd's work area (heat map, report); the user's output
    folder (request["output_dir"]) receives only the stack and the extras they chose."""
    import cv2
    import numpy as np
    from astropy.io import fits
    from gpucoadd import GPUBackend, frames_for_region, load_frames, retained_lights, sensor_edge, stack, write_png

    operation = request["operation"]
    if operation not in ("analyze", "stack", "scan"):
        raise ValueError("Unknown operation")
    lights, grouping = scan_lights(request, reporter)
    if operation == "scan":  # group table for the interface; no pixels read
        return {
            "status": "complete",
            "operation": operation,
            "folder": str(folder),
            "files": {},
            "groups": grouping["groups"],
            "input_frames": len(lights),
            "used_frames": len(grouping["used"]),
        }
    paths = [lights[i]["path"] for i in grouping["used"]]
    pedestals = [grouping["pedestal"][i] for i in grouping["used"]]
    pixel_scales = [grouping["pixel_scale"][i] for i in grouping["used"]]
    main_lights = set(grouping["main_lights"])
    reference_pool = [k for k, i in enumerate(grouping["used"]) if i in main_lights]
    names = group_names(grouping["groups"])
    group_of = {i: names[g["key"]] for g in grouping["groups"] for i in g["indices"]}
    light_groups = [group_of[i] for i in grouping["used"]]
    mixed = len(set(light_groups)) > 1
    settings = request.get("settings", {})
    quality = quality_settings(settings)
    auto = {}
    if mixed:
        if quality["rgb_equal"]:
            raise ValueError(
                "RGB equalization requires Background spread. Mixed light groups use Overlap matching; "
                "turn RGB equalization off."
            )
        # Lights of different instruments differ in noise and colour response: brightness must
        # come from their overlaps (spread matching would scale whole groups by their noise) and
        # weights from their noise (equal weights: 20-65% more noise than the Pro alone, SH2-142).
        # Noise-aware rejection left more noise than sigma clipping on SH2-142 with S50 lights
        # added (red +0.5%, blue +0.2% with each channel's own noise; +1.4% and +0.6% when it was
        # scaled from green), where sigma clipping gave the expected gain in all channels. Sigma
        # or winsorized stay the user's choice.
        want = {
            "norm": "overlap",
            "weight": "noise",
            "rejection": "sigma" if quality["rejection"] == "noise" else quality["rejection"],
        }
        auto = {k: v for k, v in want.items() if quality[k] != v}
        quality.update(want)
        if auto:
            reporter.message(
                "Mixed instruments: using Overlap matching and noise weights"
                + (", sigma clipping instead of noise-aware rejection" if "rejection" in auto else "")
                + " ("
                + ", ".join(f"{k} was {settings.get(k, 'default')}" for k in auto)
                + ")."
            )
    outputs = output_settings(request.get("outputs", {}))
    if quality["match_gradients"] and quality["norm"] != "overlap":
        raise ValueError("Gradient matching requires Overlap matching (automatic for mixed light groups).")
    threads = int(settings.get("threads", 8))
    tile = int(settings.get("tile", 0))
    selection = quality["select"]
    if threads < 1 or (tile != 0 and tile < 16) or not math.isfinite(selection) or selection < 0:
        raise ValueError("Invalid worker count, tile size, or frame selection setting")
    ref = request.get("reference") or ""
    ref_index = None
    if ref:
        ref_path = Path(ref).resolve()
        try:
            ref_index = [str(p.resolve()).casefold() for p in paths].index(str(ref_path).casefold())
        except ValueError as exc:
            raise ValueError(
                "The optional reference must be one of the used lights (selected folders and groups)"
            ) from exc
        ref = str(ref_path)
    signature = fingerprint(paths, settings, ref, reporter.source_guard)
    reporter.message(
        f"Source integrity checks active for {len(paths)} lights; changes stop the operation without excluding files."
    )
    # capture pointings guide overlapping-frame matching
    pointings = [lights[i]["pointing"] for i in grouping["used"]]
    cached = None
    if operation == "stack":
        cache_path = Path(request["analysis_file"])
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") != signature:
            old = cached.get("signature", {})
            if old.get("engine") == signature["engine"]:
                previous = {item["path"]: item for item in old.get("files", [])}
                for item in signature["files"]:
                    if item["path"] in previous and previous[item["path"]] != item:
                        raise SourceFileError(
                            f"Source light changed since the heat map was built: {item['path']}. "
                            "Rebuild the heat map before stacking. No light was excluded."
                        )
            raise ValueError("Lights, reference, selection, or engine changed. Build the heat map again.")
        reporter.message("Validated cached alignment; star detection and registration will be reused.")
    else:  # registration will run: its workers import the engine while the lights load
        from gpucoadd import start_registration_pool

        reporter.registration_pool = start_registration_pool(threads)
    reporter.message(f"{len(paths)} lights; work folder for this run: {folder}")
    region = request.get("region") if operation == "stack" else None
    if operation == "stack" and region is None:
        raise ValueError("Select framing on the heat map before stacking")
    first = lights[grouping["used"][0]]
    load = list(range(len(paths)))
    prepare_sky = False
    if cached is not None:
        if quality["norm"] == "overlap" and not shared_sky_cached(cached["analysis"], selection):
            # First Overlap matching stack from a heat map built without it: the solve needs every
            # retained light (the same lights and reference as at heat-map time); the lights that
            # cannot reach the selected area are freed after the solve, before stacking.
            prepare_sky = True
            load = retained_lights(cached["analysis"], selection)
            reporter.message(
                f"Preparing Overlap matching from all {len(load)} retained lights "
                "(first stack with it from this heat map; saved to the heat map for later stacks)."
            )
        else:
            # Lights whose footprint cannot reach the selected area are not loaded; stack() checks
            # again after loading and refuses to run if any light that can reach it is missing.
            load = frames_for_region(
                cached["analysis"],
                len(paths),
                first["width"],
                first["height"],
                selection,
                region,
                sensor_edge(quality["debayer"], quality["interp"]),
                include_ref=quality["norm"] != "overlap",
            )
            reporter.message(f"Loading {len(load)} of {len(paths)} lights; the others cannot reach the selected area.")
    if operation == "stack":
        reserved = region[2] * region[3] * 12  # the stack itself
        for key in LIGHT_COPIES:
            sizes = (
                [region[2] * region[3] * 12] * len(load)
                if key == "aligned_lights"
                else [lights[grouping["used"][k]]["width"] * lights[grouping["used"][k]]["height"] * 12 for k in load]
            )
            outputs[key] = outputs[key] and plan_light_copies(Path(request["output_dir"]), key, sizes, reserved, reporter)
            reserved += sum(sizes) if outputs[key] else 0
    else:
        outputs.update(dict.fromkeys(LIGHT_COPIES, False))
    reporter.progress("loading lights", 0, len(load))
    load_start = time.perf_counter()

    def inputs():
        for i, raw in enumerate(load_frames([str(paths[k]) for k in load], source_guard=reporter.source_guard), 1):
            reporter.check_cancel()
            yield raw
            if i % 8 == 0 or i == len(load):
                reporter.progress("loading lights", i, len(load))

    reporter.message("Quality settings: " + json.dumps(quality))
    if not load:
        raise ValueError("No lights reach the selected area. Choose a covered area on the heat map.")
    # Per-light sizes drive the math; only maximum mask dimensions constrain reuse.
    # Dropping the first light alone must not force a reload. Dropping the largest
    # mask still reloads to match a fresh worker's memory budget.
    shapes = [(lights[grouping["used"][k]]["height"], lights[grouping["used"][k]]["width"]) for k in load]
    reuse_key = {
        "signature": signature,
        "pedestals": pedestals,
        "interp": quality["interp"],
        "debayer": quality["debayer"],
        "sensor_setup": backend_geometry(shapes),
    }
    be = session.acquire(reuse_key, load) if session is not None else None
    if be is not None:
        adopt_kept_lights(be, reporter)
    import torch
    from memory_plan import start_summary

    # measured after a kept backend is reused or released, so its memory is counted correctly
    if torch.cuda.is_available():
        reporter.message(start_summary(*torch.cuda.mem_get_info()))
    reused = be is not None
    if reused:
        from_files = getattr(be, "released", 0)
        reporter.storage(
            "raw",
            "mixed" if from_files else be.storage_mode,
            "normal"
            if be.storage_mode == "gpu"
            else "capacity"
            if os.environ.get("GPUCOADD_STORAGE", "auto") == "auto"
            else "forced",
        )
        reporter.message(
            f"Reusing {len(load)} prepared lights from {be.storage_mode}"
            + (f" ({from_files} read from their files for each tile)" if from_files else "")
            + "; crops are uploaded only when needed."
        )
        reporter.progress("loading lights", len(load), len(load))
    else:
        reporter.message("Preparing GPU kernels and loading lights; new kernels compile once after an update.")
        if session is not None:
            if torch.cuda.is_initialized():
                torch.cuda.reset_peak_memory_stats()
        be = GPUBackend(
            inputs(),
            [pedestals[k] for k in load],
            len(paths),
            quality["interp"],
            quality["debayer"],
            indices=None if len(load) == len(paths) else load,
            paths=[paths[k] for k in load],
            sizes=shapes,
            log=reporter.message,
            storage_notice=reporter.storage,
        )
        reporter.progress("loading lights", len(load), len(load))
        if session is not None:
            session.remember(be, reuse_key)
    load_s = time.perf_counter() - load_start
    be.source_paths = {i: str(p) for i, p in enumerate(paths)}
    import resource_guard

    resource_guard.set_handlers(relieve=getattr(be, "release_ram", None), warn=reporter.message)
    reporter.message(
        f"{'Reused prepared lights' if reused else 'Prepared lights'} in {load_s:.1f}s; raw storage: {be.storage_mode}; GPU: {be.torch.cuda.get_device_name(0)}; RCD: {be.rcd_kind}; rejection: {be.clip_kind}; sort: {be.sort_kind}; warp: {be.warp_kind}; calibration: {be.calibration_kind}"
    )
    captured = {}

    def capture(analysis):
        captured.update(analysis)
        if operation == "stack" and prepare_sky and "normalization" in analysis:
            reporter.validate_sources()
            # saved as soon as the solve has succeeded, atomically (write_json replaces the file)
            cached["analysis"]["normalization"] = analysis["normalization"]
            write_json(cache_path, cached)
            reporter.message("Saved the overlap matching solution to the heat map.")

    img, engine_s, info = stack(
        be,
        tile,
        threads,
        transform=quality["transform"],
        select=selection,
        norm=quality["norm"],
        weight=quality["weight"],
        framing="union",
        rejection=quality["rejection"],
        feather=quality["feather"] if operation == "stack" else 0,
        rgb_equal=quality["rgb_equal"],
        match_gradients=quality["match_gradients"] if operation == "stack" else False,
        coverage_only=operation == "analyze",
        reference_idx=ref_index,
        analysis_cache=None if cached is None else cached["analysis"],
        on_analysis=capture,
        region=region,
        progress=reporter.progress,
        log=reporter.message,
        pointings=pointings,
        # solved at heat-map time only when it will be used: mixed telescopes
        # always, otherwise when Overlap matching is selected (else on first use)
        prepare_overlap=operation == "analyze" and (mixed or quality["norm"] == "overlap"),
        pixel_scales=pixel_scales,
        reference_pool=reference_pool,
        groups=light_groups if mixed else None,
        workspace=workspace,
        registration_pool=getattr(reporter, "registration_pool", None),
    )
    peak_vram = be.torch.cuda.max_memory_reserved() / 2**30
    reporter.validate_sources()
    reporter.progress("saving outputs", 0, 1)
    canvas = dict(info["canvas"])
    if region is not None:
        canvas["framing"] = "selected"
        canvas["source_framing"] = "union"
    failed = info["alignment_failed"]
    dropped = info["dropped"]
    contributing = info["contributing_indices"]
    if operation == "stack" and not contributing:
        raise ValueError("The selected area has no contributing frames. Select an area covered by the heat map.")
    input_headers = [lights[i]["output_header"] for i in grouping["used"]]
    identity = image_identity(
        paths, contributing if operation == "stack" else info["retained_indices"], headers=input_headers
    )
    groups_out = group_report(grouping, grouping["used"], info, operation)
    hdr = fits.Header()
    hdr["FRAMING"] = "selected" if operation == "stack" else "union"
    hdr["ORIGINX"] = (canvas["origin_xy"][0], "canvas top-left x in reference light pixels")
    hdr["ORIGINY"] = (canvas["origin_xy"][1], "canvas top-left y (light rows top-down)")
    hdr["NCOMBINE"] = len(contributing) if contributing is not None else info["registered"]
    hdr["RGBEQUAL"] = (operation == "stack" and quality["rgb_equal"], "RGB matched to reference green background")
    hdr["GRADMTCH"] = (
        operation == "stack" and quality["match_gradients"],
        "gradient planes matched (Overlap matching)",
    )
    hdr["OBJECT"] = fits_text(identity["target"])
    if identity["exposure_seconds"] is not None:
        hdr["EXPTIME"] = identity["exposure_seconds"]
    if identity["summed_input_exposure_s"] is not None:
        hdr["TOTINPUT"] = (identity["summed_input_exposure_s"], "Input sum, not per-pixel exposure")
    hdr["COVTYPE"] = "GEOMETRY" if operation == "analyze" else "FINITE"
    used = grouping["used"]
    named = contributing if contributing is not None else info["retained_indices"]
    centre = nearest_light(
        (cached["analysis"] if cached is not None else captured)["mats"],
        named,
        [(lights[k]["width"], lights[k]["height"]) for k in used],
        [lights[k]["pixel_scale"] for k in used],
        (canvas["origin_xy"][0] + canvas["width"] / 2, canvas["origin_xy"][1] + canvas["height"] / 2),
    )
    observation_header(hdr, paths, named, info["ref_idx"], centre, pointings, headers=input_headers)
    hdr["HISTORY"] = "Frame counts before pixel rejection; geometry prediction when COVTYPE=GEOMETRY"
    for g in groups_out:
        if g["used"]:
            n = g.get("contributing", g["retained"])
            exp = "?" if g["exposure_s"] is None else f"{g['exposure_s']:g}"
            hdr["HISTORY"] = (
                f"Group {g['telescope']} {g['filter']} {exp}s: {n} lights"
                + (
                    f", {g['main_light_equivalents']:.1f} main-light equivalents"
                    if "main_light_equivalents" in g
                    else ""
                )
            )[:72]
    out_dir = Path(request["output_dir"]).resolve()
    workspace.check_resources(include_exports=True)
    for key in getattr(workspace, "skipped_previews", []):
        outputs[key] = False
        what = {"stack_preview": "The stack preview (PNG)", "coverage_png": "The frame count heat map (PNG)"}[key]
        reporter.warn(key, MemoryError(workspace.preview_shortage), what)
    clipping = {}
    files = save_image_outputs(
        out_dir,
        img,
        info["coverage"],
        hdr,
        identity["basename"],
        info["registered"],
        outputs,
        predicted=operation == "analyze",
        offset=pedestals[info["ref_idx"]],
        clipping=clipping,
        check_cancel=reporter.check_cancel,
        before_publish=reporter.validate_sources,
        warn=reporter.warn,
    )
    if clipping.get("low") or clipping.get("high"):
        reporter.message(f"Saving the stack clipped {clipping['low']} samples at 0 and {clipping['high']} at the top.")
    result = {
        "status": "complete",
        "operation": operation,
        "folder": str(folder),
        "canvas": canvas,
        "retained": info["registered"],
        "input_frames": len(paths),
        "groups": groups_out,
        "reference": str(paths[info["ref_idx"]]),
        "reference_override": bool(ref),
        "load_upload_s": load_s,
        "engine_s": engine_s,
        "stages_s": dict(be.timer.t),
        "peak_vram_reserved_gib": peak_vram,
        "region": region,
        "reused_alignment": cached is not None,
        "coverage_kind": info["coverage_kind"],
        "loaded_lights": len(load),
        "tile_plan": info.get("tile_plan"),
    }
    result["settings"] = {**quality, "threads": threads, "tile": tile}
    result.update(weight_report(info))
    result["automatic_settings"] = sorted(auto)
    result.update(output_options=outputs, files=files)
    if clipping:
        result["export_clipping"] = clipping
    result.update(
        identity=identity,
        aligned=len(paths) - len(failed),
        alignment_failed=len(failed),
        sharpness_rejected=len(dropped),
        alignment_failed_files=[str(paths[i]) for i in failed],
        sharpness_rejected_files=[str(paths[i]) for i in dropped],
        contributing=len(contributing) if contributing is not None else None,
        registration=info["registration"],
        normalization=info.get("normalization"),
    )
    details = info["registration"].get("frames", [])
    if outputs["frames_csv"]:
        frames_csv = out_dir / (identity["basename"] + "_frames.csv")

        def write_frames(target):
            with open(target, "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "file",
                        "group",
                        "status",
                        "fwhm_px",
                        "weight",
                        "alignment",
                        "anchor",
                        "matched_stars",
                        "rms_px",
                        "reason",
                        "transform",
                    ],
                )
                label_of = {i: g["label"] for g in grouping["groups"] for i in g["indices"]}
                weight_of = {
                    i: light_weight(w) for i, w in zip(info.get("weighted_indices") or [], info.get("weights") or [])
                }
                writer.writeheader()
                for i, p in enumerate(paths):
                    d = details[i] if i < len(details) else {}
                    status = "alignment_failed" if i in failed else "sharpness_rejected" if i in dropped else "retained"
                    if status == "retained" and contributing is not None and i not in contributing:
                        status = "outside_selected_area"
                    writer.writerow(
                        {
                            "file": str(p),
                            "group": label_of[grouping["used"][i]],
                            "status": status,
                            "fwhm_px": info["fwhm_px"][i],
                            "weight": weight_of.get(i, ""),
                            "alignment": d.get("method", ""),
                            "anchor": d.get("anchor", ""),
                            "matched_stars": d.get("matched_stars", ""),
                            "rms_px": d.get("rms_px", ""),
                            "reason": d.get("reason", ""),
                            "transform": d.get("transform", "")
                            + (f" (homography rejected: {d['homography_rejected']})" if "homography_rejected" in d else ""),
                        }
                    )

        # optional: the stack is already saved, so a refused CSV is a warning, not a failed run
        save_optional(files, "frames_csv", frames_csv, write_frames, reporter, "The frame report (CSV)")
    reporter.message(
        f"Frame report: {len(paths) - len(failed)} aligned, {len(failed)} failed alignment, "
        f"{len(dropped)} rejected for sharpness, {info['registered']} retained."
    )
    if failed:
        reporter.message(
            f"Attention: {len(failed)} lights could not be aligned. See the run report for the file list; these were not rejected for image quality."
        )
    if operation == "analyze":
        reporter.validate_sources()
        analysis_file = folder / "analysis.json"
        write_json(
            analysis_file,
            {
                "signature": signature,
                "analysis": captured,
                "canvas": canvas,
                "lights_dirs": light_folders(request),
                "exclude_groups": sorted(request.get("exclude_groups") or []),
            },
        )
        # Heat-map-only image has no legend: every display pixel maps to sky framing.
        scale = min(1, 1100 / canvas["width"], 750 / canvas["height"])
        size = (max(1, round(canvas["width"] * scale)), max(1, round(canvas["height"] * scale)))
        source = workspace.float_coverage(info["coverage"])
        small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        del source
        colors = cv2.applyColorMap(np.uint8(np.clip(small / info["registered"], 0, 1) * 255), cv2.COLORMAP_TURBO)
        colors[small == 0] = 0
        heatmap = folder / "heatmap.png"
        write_png(heatmap, colors)
        result.update(analysis_file=str(analysis_file), heatmap=str(heatmap))
        files.update(analysis_cache=str(analysis_file), interactive_heatmap=str(heatmap))
    else:
        result["stack_file"] = files["stack_fits"]
        reporter.message(
            f"{len(contributing)} retained lights contribute pixels to the selected area; named output accordingly."
        )
    reporter.validate_sources()
    reporter.progress("saving outputs", 1, 1)
    for key, (suffix, what) in LIGHT_COPIES.items():
        if outputs[key]:
            copies = save_light_copies(
                key,
                out_dir,
                identity["basename"],
                be,
                info,
                hdr,
                paths,
                pointings,
                input_headers,
                pedestals,
                workspace,
                reporter,
            )
            result[key] = copies
            if copies["saved"]:
                files[key] = copies["folder"]
            reporter.message(f"Saved {copies['saved']} {what} ({copies['bytes'] / 2**30:.1f} GiB) in {copies['folder']}")
            reporter.validate_sources()
    reporter.message(
        f"Completed {operation}: {info['registered']}/{len(paths)} lights, {canvas['width']} x {canvas['height']} pixels"
    )
    result["storage"] = be.storage_info()
    result["source_integrity"] = {
        "method": "file identity, size and timestamps (not content hashes)",
        "lights": len(paths),
        "periodic_interval_s": reporter.source_check_interval,
        "fits_publication_guard_enabled": True,
    }
    result["output_storage"] = workspace.info
    result["reused_gpu_lights"] = reused and be.storage_mode == "gpu"
    result["reused_prepared_lights"] = reused
    if session is not None and operation == "analyze":
        # Hold useful raws while framing, not failed/blurred lights (keep the optional reference).
        be.keep_only(set(info["retained_indices"]) | {captured["ref_idx"]})
    return result


def _run_request(request_path, session=None):
    request_path = Path(request_path).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    started = datetime.now(timezone.utc).isoformat()
    # This run's own files (heat map, report, live activity log) go to GPUCoadd's work area, not
    # the output folder, which receives only the stack and the extras the user chose.
    root = work_root(request)
    prune_work(root, keep=[request.get("analysis_file")])
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + request["operation"]
    folder, k = root / "runs" / name, 1
    while True:  # unique even for runs started within the same microsecond
        try:
            folder.mkdir(parents=True)
            break
        except FileExistsError:
            folder, k = root / "runs" / f"{name}_{k}", k + 1
    outputs = output_settings(request.get("outputs", {}))
    reporter = Reporter(folder, request_path.with_suffix(".cancel"), save_log=outputs["activity_log"])
    reporter.emit("started", started_utc=started, folder=str(folder), message="Operation started")
    result = {}
    try:
        if session is not None:
            session.validate_source()
        result = execute(request, reporter, folder, session)
    except Cancelled:
        result = {"status": "cancelled", "folder": str(folder), "operation": request["operation"]}
        reporter.message("Cancelled. Existing lights and earlier results are unchanged.")
    except Exception as exc:
        result = {"status": "error", "folder": str(folder), "operation": request["operation"], "error": str(exc)}
        failure = build_failure(
            exc,
            request,
            stage=reporter.stage,
            started=started,
            elapsed=time.perf_counter() - reporter.start,
            events=reporter.recent_events,
        )
        path, errors = save_failure(failure, folder, root)
        result.update(failure=failure, error_report=path, error_report_save_errors=errors)
        reporter.emit("error", message=str(exc), traceback=traceback.format_exc())
    result.update(
        started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        wall_s=time.perf_counter() - reporter.start,
        worker_pid=os.getpid(),
    )
    result["storage_notices"] = list(reporter.storage_notices.values())
    result.setdefault("files", {})["report"] = str(folder / "report.json")
    result["request_file"] = str(request_path)  # lets the interface match the answer to its job
    result["output_options"] = outputs
    result["image_outputs"] = image_output_manifest(result["files"])
    if session is not None:
        if result["status"] != "complete":
            session.release()
        result["worker_cache"] = session.idle()
    # Everything that can still fail happens before the outcome is announced: the interface acts
    # on the first "finished" it reads, so nothing after it may change or replace the result.
    # The optional activity log is finished and copied first; a refused copy is a warning.
    if reporter.log_file is not None:
        copy_to = None
        if outputs["activity_log"] and request["operation"] != "scan":
            name = (result.get("identity") or {}).get("basename") or folder.name
            copy_to = Path(request["output_dir"]).resolve() / (name + "_activity.jsonl")
            result["files"]["activity_log"] = str(copy_to)
        result["warnings"] = list(reporter.warnings)
        reporter.emit("finished", live=False, **result)  # the log ends with the outcome
        log_ok = reporter.log_file is not None
        reporter.close_log()
        if copy_to is not None:
            del result["files"]["activity_log"]  # listed again only once the copy is saved
            if log_ok:  # else writing it failed during the run, already a warning
                import shutil

                save_optional(
                    result["files"],
                    "activity_log",
                    copy_to,
                    lambda target: shutil.copyfile(folder / "activity.jsonl", target),
                    reporter,
                    "The activity log copy in the results folder",
                )
    result["warnings"] = list(reporter.warnings)
    write_json(folder / "report.json", result)
    write_json(request_path.with_suffix(".result.json"), result)
    reporter.emit("finished", **result)
    return 1 if result["status"] == "error" or (session is not None and result["status"] == "cancelled") else 0


def run_request(request_path, session=None):
    # Includes failures before the Reporter exists and failures while saving the final report.
    try:
        return _run_request(request_path, session)
    except Exception as exc:
        request = {"request_file": str(request_path)}
        try:
            request.update(json.loads(Path(request_path).read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            pass
        root = work_root(request)
        failure = build_failure(exc, request, stage="worker setup or final reporting")
        path, errors = save_failure(failure, root / "errors", root)
        result = dict(
            status="error",
            operation=request.get("operation"),
            request_file=str(request_path),
            error=str(exc),
            failure=failure,
            error_report=path,
            error_report_save_errors=errors,
        )
        if session is not None:
            try:
                session.release()
            except Exception:
                pass
        try:
            write_json(Path(request_path).with_suffix(".result.json"), result)
        except OSError:
            pass
        print(json.dumps({"type": "finished", **result}, default=str), flush=True)
        return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--request")
    mode.add_argument("--session")
    ap.add_argument("--idle-timeout", type=float, default=300)
    args = ap.parse_args()
    if args.session:
        if not math.isfinite(args.idle_timeout) or args.idle_timeout <= 0:
            ap.error("idle timeout must be positive")
        from worker.worker_session import serve

        return serve(args.session, run_request, args.idle_timeout)
    return run_request(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
