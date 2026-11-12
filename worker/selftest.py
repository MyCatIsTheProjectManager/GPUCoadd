"""End-to-end GPU self-test, run by the setup program after installing (and usable any time):
renders six synthetic star fields, shifted against each other, and runs them through the real
pipeline (star detection, alignment, every CUDA kernel, stacking). Passes when every light
aligns, the stars land where the shifts put them and stay as sharp as they were rendered.
Prints one JSON object per line ({"step", "message"} while running, then {"result": ...}).
Reads and writes no lights; the GPU kernels it compiles are cached in GPUCoadd's work folder."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FRAMES = 6
WIDTH, HEIGHT = 1024, 768
SIGMA = 1.6  # star size as rendered (FWHM 3.8 px)
SHIFTS = [(0.0, 0.0), (7.3, -4.6), (-5.2, 6.9), (11.8, 3.1), (-9.4, -8.2), (3.7, 10.4)]


def say(**event):
    print(json.dumps(event), flush=True)


def render(np, rng, stars, dx, dy):
    """A raw GRBG light of the sky shifted by (dx, dy): grey stars on a flat background, with noise."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    img = np.full((HEIGHT, WIDTH), 900.0)
    for x, y, flux in stars:
        cx, cy = x - dx, y - dy
        if -10 < cx < WIDTH + 10 and -10 < cy < HEIGHT + 10:
            x0, x1 = max(0, int(cx) - 8), min(WIDTH, int(cx) + 9)
            y0, y1 = max(0, int(cy) - 8), min(HEIGHT, int(cy) + 9)
            gx, gy = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
            img[y0:y1, x0:x1] += flux / (2 * np.pi * SIGMA**2) * np.exp(-(gx**2 + gy**2) / (2 * SIGMA**2))
    img += rng.normal(0, np.sqrt(img))
    return np.clip(img, 0, 65535).astype(np.uint16)


def measure(np, image, x, y):
    """Centroid and second-moment FWHM of a star in the green channel around (x, y)."""
    g = image[..., 1]
    x0, y0 = int(round(x)) - 6, int(round(y)) - 6
    box = g[y0 : y0 + 13, x0 : x0 + 13].astype(np.float64)
    if box.shape != (13, 13) or not np.isfinite(box).all():
        return None
    box = np.clip(box - np.median(g[y0 - 10 : y0 + 23, x0 - 10 : x0 + 23]), 0, None)
    yy, xx = np.mgrid[:13, :13]
    total = box.sum()
    cx, cy = (box * xx).sum() / total, (box * yy).sum() / total
    var = ((box * (xx - cx) ** 2).sum() + (box * (yy - cy) ** 2).sum()) / (2 * total)
    return x0 + cx, y0 + cy, 2.3548 * np.sqrt(var)


def main():
    started = time.perf_counter()
    say(step="gpu", message="Checking the NVIDIA GPU and CUDA")
    from worker import hardware_check

    hw = hardware_check.check()
    if not hw.get("supported"):
        say(result="failed", summary=hw.get("summary"), next_step=hw.get("next_step"), error=hw.get("error", ""))
        return 1
    say(step="gpu", message=f"GPU ready: {hw.get('gpu', 'NVIDIA GPU')}")

    import numpy as np
    from gpucoadd import GPUBackend, stack

    rng = np.random.default_rng(20261030)
    stars = [(rng.uniform(20, WIDTH - 20), rng.uniform(20, HEIGHT - 20), rng.lognormal(10.0, 0.8)) for _ in range(220)]
    say(step="render", message=f"Rendering {FRAMES} synthetic lights ({WIDTH} x {HEIGHT})")
    raws = [render(np, rng, stars, dx, dy) for dx, dy in SHIFTS[:FRAMES]]

    say(step="stack", message="Aligning and stacking them (the first run also compiles the GPU kernels)")
    analysis = {}
    be = GPUBackend(iter(raws), 100, FRAMES, "bicubic", "rcd")
    image, _, info = stack(be, 256, 2, framing="union", on_analysis=analysis.update, log=lambda _: None)

    say(step="check", message="Checking star positions and sharpness")
    problems = []
    if info["registered"] != FRAMES:
        problems.append(f"only {info['registered']} of {FRAMES} lights aligned")
    rdx, rdy = SHIFTS[info["ref_idx"]]
    ox, oy = info["canvas"]["origin_xy"]
    errors, widths = [], []
    for x, y, flux in sorted(stars, key=lambda s: -s[2])[:40]:
        px, py = x - rdx - ox, y - rdy - oy  # where the star must land in the stack
        m = measure(np, image, px, py)
        if m is not None:
            errors.append(np.hypot(m[0] - px, m[1] - py))
            widths.append(m[2])
    if len(errors) < 20:
        problems.append(f"only {len(errors)} test stars measurable in the stack")
    else:
        err, width = float(np.median(errors)), float(np.median(widths))
        rendered = 2.3548 * SIGMA
        if err > 0.3:
            problems.append(f"stars land {err:.2f} px from where they should (limit 0.3 px)")
        if width > 1.25 * rendered:
            problems.append(f"stars are {width:.2f} px wide, rendered at {rendered:.2f} px (limit +25%)")
    if not np.isfinite(image).any():
        problems.append("the stack has no data")
    # Every stage ran, but a kernel that could not be built leaves its stage on a slower path
    # (the GPUCOADD_* switches are for testing and are not a fault). Feathering builds its kernel
    # on first use, which this stack does not reach, so it is built here.
    from gpucoadd import kernel_error
    from worker.bridge import KERNEL_STAGES

    kernels = be.kernel_status()
    try:
        import feather_kernels

        feather_kernels.module()
    except Exception as exc:
        kernels["feather"] = {"reason": "build failed", "error": kernel_error(exc)}
    slow = {
        stage: kernels[stage]["error"]
        for stage in KERNEL_STAGES
        if (kernels.get(stage) or {}).get("reason") == "build failed"
    }
    seconds = time.perf_counter() - started
    if problems:
        say(result="failed", summary="The GPU self-test found a problem: " + "; ".join(problems) + ".",
            next_step="Update the NVIDIA driver and run the self-test again. If it still fails, send the setup log.",
            seconds=round(seconds, 1))
        return 1
    if slow:
        names = ", ".join(KERNEL_STAGES[stage] for stage in slow)
        say(result="slow", summary=f"{FRAMES} synthetic lights aligned and stacked correctly, but these GPU kernels "
            f"could not be built, so their stages run on slower paths: {names}.",
            next_step="GPUCoadd works. An NVIDIA driver update may fix the build; run the self-test again after it.",
            error="; ".join(f"{stage}: {error}" for stage, error in slow.items()), gpu=hw.get("gpu"),
            seconds=round(seconds, 1))
        return 0
    say(result="ok", summary=f"{FRAMES} synthetic lights aligned and stacked; stars within "
        f"{float(np.median(errors)):.2f} px of their true positions, {float(np.median(widths)):.2f} px wide.",
        gpu=hw.get("gpu"), seconds=round(seconds, 1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # report every failure in the same form, never a bare traceback
        say(result="failed", summary="The GPU self-test could not run.", next_step="Send the setup log.",
            error=f"{type(exc).__name__}: {exc}")
        sys.exit(1)
