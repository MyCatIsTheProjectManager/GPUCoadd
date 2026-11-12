# GPUCoadd technical reference

How GPUCoadd works, what was measured, and how to build and test it. The [README](../README.md) is
the quick start; the [guide](GUIDE.md) describes the controls. Links to tests and measurement records
refer to the source repository; the installed copy contains only the program and its documents.

## Runtime

- Python 3.11, PyTorch 2.11.0+cu128 and 23 other pinned packages
  ([requirements-lock.txt](../requirements-lock.txt)).
- NVIDIA GPU with compute capability 7.5 or newer. Setup advises a driver update below 570.
- [worker/selftest.py](../worker/selftest.py) runs six synthetic Bayer lights through alignment and
  stacking. It passes when all six align, at least 20 stars are measurable, the median position
  error is at most 0.3 px and the median star width is within 25% of the rendered width. If a GPU
  kernel can't be built, it reports `slow` and setup still completes.
- Window: [GPUCoadd/gpucoadd_qt](../GPUCoadd/gpucoadd_qt) (PyQt6), opened on its own by
  [start_gpucoadd.pyw](../start_gpucoadd.pyw) (the Start menu entry runs it with `pythonw.exe`) or
  from Siril's Scripts menu by [GPUCoadd/GPUCoadd.py](../GPUCoadd/GPUCoadd.py). Both start the same
  worker, [worker/bridge.py](../worker/bridge.py), which runs the engine, [gpucoadd.py](../gpucoadd.py).
  On its own, Open Selected Image uses the program Windows has for `.fits` files, or shows the file
  in Explorer.
- Tested with Siril 1.4.4 on Windows 10 (22H2), with an RTX 3060 12 GB and NVIDIA driver 616.92.

## Pipeline

1. Read the lights and subtract each light's camera offset.
2. Detect stars and measure their brightness and width.
3. Match stars to the reference light (triangle matching, astroalign) and fit a similarity or
   homography transform. A homography is kept only where its matched stars cover most of the
   field and it stays close to the similarity; otherwise the similarity is used. Lights that
   can't match the reference directly are chained through already aligned neighbours; there is
   no global bundle adjustment.
4. Optionally drop blurry lights: FWHM above the median plus a chosen multiple of its spread.
5. Compute brightness corrections (see Normalization).
6. For each output tile: debayer and warp every light that reaches it, apply the corrections,
   reject outliers and average.

There is no dark, flat or bias calibration. After the offset subtraction, each light is multiplied
by an array of ones that marks where a master flat will go; it changes no pixel value.

| CPU | GPU (CUDA through PyTorch) |
|---|---|
| Reading FITS; checking that lights don't change | Camera-offset subtraction |
| Star matching and transform fitting (worker processes) | Star detection, star photometry, background statistics |
| Star widths, from small windows cut out on the GPU | Debayering (RCD or bilinear) and warping |
| Reference choice, frame selection, correction coefficients | Overlap matching's pairwise comparisons |
| Overlap matching's least-squares solve (SciPy) | Applying corrections, rejection and averaging |
| Frame count map; PNG encoding | Aligned-light rendering and FITS byte formatting |
| Writing, flushing and verifying files | |

`--backends cpu` runs the same pipeline on the CPU, as a reference for tests.

## Defaults

RCD debayering, bicubic resampling, Background spread normalization, equal weights, sigma clipping,
similarity transform, RGB background equalization on. Blurry-light rejection, feathering and
gradient matching off. On the command line, RGB equalization needs `--rgb-equal`.

## Reference selection

`choose_reference` in [gpucoadd.py](../gpucoadd.py):

1. Candidates: lights near the centre of the covered sky, at least 10 lights or 5% of the pool.
   Without pointing data, all lights.
2. Keep lights with finite FWHM, star flux at least 90% of the candidate median, and background at
   most 1.1 times the candidate median and 1.5 times the pool median.
3. Pick the sharpest; ties go to the one nearest the centre, then mid-session.

If none qualifies, the search widens to the whole pool, then falls back to sharpness alone.
`GPUCOADD_REFERENCE=session` selects the middle light of the session instead.

## Normalization and weighting

- **Background spread:** per-channel median and spread matched to the reference.
- **Star brightness:** scale from measured star flux ratios.
- **Overlap matching:** differences on shared sky, solved across all lights
  ([overlap.py](../overlap.py)). Gradient matching (experimental) adds a fitted plane per light.

All are scale and offset corrections, optionally with a plane. RGB equalization matches each
channel's background to the reference light's green; it is not colour calibration. Noise weights
are per channel, after scaling. With two telescopes the worker uses Overlap matching and noise
weights, and replaces noise-aware rejection with sigma clipping.
[lightgroups.py](../lightgroups.py) refuses to combine groups with different filter labels; filters
not recorded in the headers can't be detected.

## Memory and storage

[memory_plan.py](../memory_plan.py) places each job's data when it starts, by speed benefit: lights
first (GPU, else RAM, else read again from their files), then Overlap matching data, then the output
canvas (RAM, else disk). Lights not held on the GPU are uploaded in the crops each tile needs.

- GPU: free memory minus max(512 MB, 8% of total) for the desktop, and minus a working space of at
  least max(3 GB, 25% of total) for star detection and tiles.
- RAM: all available RAM minus 128 MB.

If other programs take RAM during a job, lights are read again from their files and further buffers
move to disk. Storage location never changes a pixel;
[test_combinations.py](../tests/test_combinations.py) checks options and storage layouts pairwise,
including tile and crop equivalence. Allocation failures or a full disk can still stop a job.

[source_integrity.py](../source_integrity.py) checks each light's identity, size and timestamps
before and after reading and at later checkpoints. Lights are never modified.

## FITS output

- Pixels: `(ADU + ADUOFFS) / ADUSCALE` with `ADUSCALE = 2 * 65535`, clipped to a positive floor and 1;
  uncovered pixels are 0. Clipping counts are recorded.
- Rows are stored bottom-up (`ROWORDER = 'BOTTOM-UP'`).
- `NCOMBINE`: contributing lights. `TOTINPUT`: summed exposure, not exposure per pixel.
- `RA`/`DEC`: pointing of the light nearest the canvas centre, a seed for plate solving only.
- `OBJECT` is ASCII (accents dropped); file names keep any script.

[fits_output.py](../fits_output.py) writes each file to a temporary file in the same folder, flushes
it, checks header bytes, shape and length, reads back every pixel byte, then publishes it with
`os.replace`. Stacks are compared against a SHA-256 of the written bytes. Aligned lights are
compared against the bytes still in RAM, block by block through one reusable buffer with `memcmp`
(NumPy as fallback). Optional outputs that can't be saved become warnings; the stack is kept and
both windows show the warning above the log.

## Aligned lights

A separate pass after the stack is saved. `render_light` warps one light at the stack's absolute
coordinates, in 2048 px tiles that halve if GPU memory runs out, so the files hold the stack's
samples before rejection. With equal weights and no feathering, averaging them with the same
rejection reproduces the stack ([tests](../tests/test_aligned_lights.py)).

With the canvas in RAM, the light stays on the GPU and one CUDA kernel
([export_kernels.py](../export_kernels.py)) produces the finished big-endian FITS bytes, rounding in
float32 as NumPy does. Only those bytes are copied to pinned host memory. One background thread
writes and verifies the previous light while the next one renders. This needs about 24 bytes of GPU
memory per pixel and two pinned buffers of 12 bytes per pixel. When they don't fit, formatting runs
on the CPU with identical bytes; a canvas on disk exports one light at a time.

The worker can also export debayered lights (each light in colour, unaligned). The option is hidden
in both windows until dark and flat calibration exists.

## Measurements

RTX 3060 12 GB, Intel i7-6700K, NVMe SSD. Single observations, not guarantees.

| Target | Lights | Layout | Canvas | Heat map | Stack |
|---|---|---|---|---:|---:|
| [IC 63](../baseline/results/reference_trial_ic63.json) | 31 S50 Pro | Single field | 2428 × 3986 | 6.8 s | 6.9 s |
| [68 Cygni](../baseline/results/reference_trial_68cyg.json) | 279 S50 | Mosaic | 2643 × 4280 | 13.0 s | 10.1 s |
| [NGC 6823](../baseline/results/reference_trial_ngc6823.json) | 498 S50 | Mosaic | 2836 × 4049 | 16.8 s | 14.5 s |
| [SH2-142](../baseline/results/reference_trial_sh2_142.json) | 351 S50 Pro | Single field | 4313 × 4553 | 17.5 s | 23.6 s |
| [IC 1396](../baseline/results/reference_trial_ic1396.json) | 182 S50 Pro | Mosaic | 10636 × 10685 | 15.1 s | 25.6 s |
| [NGC 7023](../baseline/results/reference_trial_ngc7023.json) | 230 S50 Pro + 210 S50 | Two telescopes | 4523 × 4575 | 107.4 s | 28.1 s |
| [SH2-114](../baseline/results/reference_trial_sh2_114.json) | 5194 S50 | Mosaic | 3037 × 4419 | 173.0 s | 161.1 s |

An S50 Pro light has four times the pixels of an S50 light. With two telescopes, the Overlap
matching fit runs during the heat map, which is why NGC 7023's heat map is long.

Light export changes; each kept every file byte-identical:

| Change | Lights | Before | After | Record |
|---|---|---:|---:|---|
| Write the previous light while rendering the next | 31 S50 Pro, aligned + debayered | 30.7 + 25.6 s | 26.0 + 22.8 s | [pipeline](../baseline/results/export_pipeline_20261031.json) |
| Compare read-back with bytes in RAM instead of SHA-256 | same | 25.5 + 22.1 s | 12.3 + 10.2 s | [verification](../baseline/results/export_verification_20261031.json) |
| Format FITS bytes on the GPU | 337 S50 Pro, aligned + debayered | 125.8 + 117.6 s | 58.9 + 57.4 s | [GPU formatting](../baseline/results/export_gpu_formatting_20261031.json) |
| Reusable read-back buffer and `memcmp` | 31 S50 Pro, both exports | 10.4 s | 7.5 s | [read-back](../baseline/results/export_exact_readback_20261101.json) |

For large exports, writing to the drive dominates: about 84% of a 53 GiB aligned export was spent
waiting for the drive to complete writes, with rendering overlapped
([storage record](../baseline/results/export_storage_limit_20261102.json)).

## Limitations

- No dark, flat or bias calibration; no colour calibration.
- Experimental choices, marked so in the window: Star brightness and Overlap matching, noise-based
  weights, noise-aware rejection, homography alignment and gradient matching.
- Output names have one-second timestamps and can collide between simultaneous jobs.
- Installer upgrades replace files in place; cancelling an upgrade doesn't restore the previous
  version. Uninstall removes the runtime folders it created, including files added inside them.
- Siril fails to open files from paths above about 256 bytes of UTF-8.
- Working files are in `%LOCALAPPDATA%\GPUCoadd`; run folders older than 30 days or beyond the
  newest 100 are removed.

## Development

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
& .\.venv\Scripts\python.exe worker\selftest.py
```

Open the window:

```powershell
& .\.venv\Scripts\pythonw.exe start_gpucoadd.pyw
```

Tests (CUDA tests skip without a GPU):

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
```

Environment switches, for tests (first value is the default):

| Variable | Values | Selects |
|---|---|---|
| `GPUCOADD_STORAGE` | `auto`, `gpu`, `ram`, `disk` | Where raw lights are held |
| `GPUCOADD_SKY_STORAGE` | `auto`, `gpu`, `ram`, `disk` | Where Overlap matching's low-resolution lights are held |
| `GPUCOADD_CANVAS` | `auto`, `ram`, `disk` | Where the output canvas is held |
| `GPUCOADD_REFERENCE` | `measured`, `session` | Automatic reference rule |
| `GPUCOADD_REGISTRATION` | unset, `reference` | astroalign's own triangle preparation |
| `GPUCOADD_WEIGHTS` | `rgb`, `green` | Noise weights per channel or from green |
| `GPUCOADD_TILE_LIGHTS` | unset, `all` | Every light in every tile |
| `GPUCOADD_RCD` | unset, `jiterator`, `reference` | RCD implementation |
| `GPUCOADD_CALIBRATE`, `_WARP`, `_SORT`, `_CLIP`, `_OVERLAP` | unset, `torch` | torch instead of the CUDA kernel |
| `GPUCOADD_SYNC_TIMING` | unset, `0` | `0`: stage timers don't wait for the GPU |

Build the setup program from the pinned download manifest (`--offline`). It ships the program, the
README, this reference, the guide, the licence and the notices; no tests, tools or measurements:

```powershell
& .\.venv\Scripts\python.exe tools\make_setup.py --offline
```

Stack one lights folder from the command line (`gpucoadd.py --help` lists the options):

```powershell
& .\.venv\Scripts\python.exe gpucoadd.py 'path/to/lights' --backends gpu --framing union --out out/union
```
