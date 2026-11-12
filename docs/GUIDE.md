# GPUCoadd user guide

GPUCoadd opens the same window from the Start menu or from Siril, with the same steps, settings and
results; this guide covers both. Installation is described in the [README](../README.md), implementation details and
measurements in the [technical reference](TECHNICAL.md).

## Opening GPUCoadd

**On its own:** open **GPUCoadd** from the Start menu. Setup adds the entry; it runs the window with
GPUCoadd's own Python. After an update, close the window and open it again.

**Siril** (1.4.0 or newer, with Python scripting): once, add the `GPUCoadd` folder inside the
install folder under **Preferences > Scripts**, and restart Siril if GPUCoadd doesn't appear. Then
choose **Scripts > Python Scripts > GPUCoadd > GPUCoadd.py**. Use the installed folder, not a copy:
the script finds the engine and its Python environment next to it. Installs before 0.9.1 used a
folder named `siril`; replace that path with the new one. Siril stays usable while the window is
open, and its console reports when GPUCoadd opens, finishes a heat map or stack, or fails.

Opening the window never starts work. Processing runs in GPUCoadd's own Python environment (`.venv`
in the install folder); **Engine settings** can point to another one.

### GPU check

When the window opens, a short NVIDIA/CUDA check runs before any light is read. The buttons that
start work stay disabled until it passes. On failure, the window shows an explanation with
copyable details; correct the Python path under **Engine settings** or click **Check again**.

When the check passes, the log shows free GPU memory, free RAM, free disk space and what they allow,
for example: *Sets up to 7.0 GB of lights (about 453 S50 Pro or 1,815 S50 lights) stay on the GPU;
larger sets use RAM up to 29.9 GB, then disk storage automatically.* A job may use all RAM that is
free when it starts, except 128 MB.

An amber panel (**Action recommended: free up resources**) appears only when less than 2 GB of RAM
is free, less than 10 GB is free on a drive GPUCoadd uses, or other programs hold more than a
quarter of the GPU memory. It names what to free; then click **Check again**. It never blocks work.

## Steps

1. **Target folder.** Choose the folder of one target, the one that contains `lights` (choosing
   `lights` itself also works). Other `lights` folders one level down, such as a second
   telescope's, are listed unticked; check their filter before ticking them, because the same filter
   name can describe different filters. **Add Folder…** adds lights from elsewhere. Seestar `_sub`
   folders are never read. Results are saved in the target folder unless **Another folder** is
   ticked.
2. **Check folders (optional).** **Check Folders** reads only file headers and fills the light-group
   table: telescope, filter, exposure, number of lights and a note. Untick a group to leave it out.
   Groups with a different filter from the main group can't be ticked.
3. **Alignment reference (optional).** Automatic, or a chosen light. Automatic picks a sharp light
   with a dark, clear sky near the middle of the covered sky; the log states which one and why.
4. **Build heat map.** **Build Heat Map** finds the stars, aligns the lights and shows how many
   lights cover each part of the sky: black is none, blue to red is few to many. The heat map stays
   while the window is open, so any number of crops and settings can be stacked from it.
5. **Framing.** **Full Frame** starts ticked. Drag to draw a crop, drag inside it to move it, or
   enter X, Y, Width and Height. Tick **Full Frame** to return to the whole canvas.
6. **Stack.** Check the results folder and the optional **Image Quality**, **Engine settings** and
   **Output Files** sections in the right column, then click **Stack Full Frame** or **Stack Selected Area**. The heat
   map's alignment is reused; stacking doesn't align again.
7. **Results.** The summary lists the saved images. Select one and click **Open Selected Image**.
   From Siril, the image opens in Siril with an autostretched display (the data is unchanged). On its
   own, it opens in the program Windows uses for `.fits` files, or Explorer shows the file. The stack
   is linear and needs a stretch.

Folders are restored when the window is reopened while Siril keeps running; after a restart of Siril,
and each time the window is opened from the Start menu, it starts empty.

Progress, timings, notices, the option sections and the activity log are on the right, with
**Save Activity Log**, **Cancel Operation** and **Close** below. Time estimates are per stage.
Cancelling stops at the next checkpoint. If the worker hasn't stopped after 20 s, the button
becomes **Force Stop**: it ends the worker and removes the job's unfinished files.

### Stage circles

Each stage has a circle that fills as it progresses. Some stages appear only when needed, and the
log explains why:

- **Linking groups** (two telescopes or several light groups): aligns the other groups to the
  reference through one well-matched light per group.
- **Neighbour alignment** (mosaics, drift): lights that share too little sky with the reference are
  aligned through overlapping lights that are already aligned. Quality is unaffected.
- **Overlap matching** runs in numbered steps (2 when its data fits on the GPU, 10 in RAM or on
  disk); its circle restarts for each step.

## Keeping lights ready

**Keep lights ready** (on by default) keeps the prepared lights between the heat map and the stack,
so the stack starts sooner. Small sets stay on the GPU; larger ones use RAM or are read again from
their files, which must stay in place. The line next to **Release GPU** shows what is held.
**Release GPU**, **Close** or five idle minutes free it; the heat map and framing remain.

Changing folders, the reference or alignment settings starts fresh, as do errors and cancelling.
With the box unticked, every step starts fresh.

## Large sets

Each job measures free GPU memory and RAM when it starts and places data by speed benefit: lights
first (GPU, else RAM, else read again from their files), then Overlap matching data, then the output
(RAM, else disk). Part of the GPU memory and 128 MB of RAM always stay free. The log states the plan,
for example *Memory plan: lights in GPU memory (2.0 GB); output canvas in RAM (0.9 GB).*

When data goes to disk, an amber notice says so (**Disk-backed processing: same image quality**).
The result is identical; it takes longer and needs free space in GPUCoadd's work folder. If RAM was
short, the notice states how much was free and how much was needed.

If other programs take RAM during a job, GPUCoadd reads lights again from their files and moves
further buffers to disk without changing any setting. A PNG preview that doesn't fit in memory is
skipped with a warning. Allocation failures or a full disk can still stop a job.

## Image Quality

The three option sections start collapsed and open one at a time; each setting has a tooltip. **Restore Recommended Defaults**
sets RCD, bicubic, Background spread, equal weights, sigma clipping and similarity, turns Equalize
RGB backgrounds on, and turns off blurry-light rejection, feathering and gradient matching.

| Setting | Default | Other choices |
|---|---|---|
| Colour method | RCD | Bilinear |
| Resampling | Bicubic | Bilinear |
| Brightness | Background spread | Star brightness (experimental), Overlap matching (experimental) |
| Frame weights | Equal | Noise-based, per colour channel (experimental) |
| Rejection | Sigma clipping | Winsorized (gentler), noise-aware (experimental) |
| Alignment | Similarity | Homography (experimental) |
| Leave out blurry lights | Off | On, sharpness limit 2.5 or custom |
| Feather edges | 0 (off) | Up to 1000 px |
| Equalize RGB backgrounds | On | Off |
| Match gradients | Off | On (experimental) |

- **Mosaics or changing haze:** Overlap matching with noise-based weights and noise-aware rejection
  compares lights on the sky they share and weights hazy lights down. It gave the most signal on
  mosaics taken over several nights.
- **Leave out blurry lights:** off, so every aligned light is used. When on, lower limits drop more
  lights (focus drift, wind, poor seeing).
- **Feather edges** blends each light in over this many pixels from its edge, hiding steps between
  lights. Wider feathering adds noise where few lights overlap.
- **Equalize RGB backgrounds** (Background spread with one light group only) matches every channel
  to the reference light's green background. It changes colour balance and is not colour
  calibration. It switches off with other brightness methods or mixed groups and returns when
  available again, unless it was turned off manually.
- **Match gradients** (Overlap matching only) removes smooth background differences between
  overlapping lights. Weak or noisy overlaps can change faint structure. It doesn't replace flats:
  without them, the lights' vignetting makes it add a broad dome across a mosaic.

Stacks with both telescopes switch to Overlap matching and noise weights automatically, and
noise-aware rejection becomes sigma clipping. The log lists the changes.

Changing the alignment, the blurry-light filter, the folders or the group ticks makes the heat map
out of date; other settings don't. A new blurry-light limit reuses the previous alignment. Every
run saves its settings in its `report.json`.

## Engine settings

The Python executable, the number of registration workers and the tile size. **Automatic tiles** (on
by default) use 512 px and shrink to 256, 128 or 64 px when GPU memory is tight. Tile size never
changes the framing or the result.

## Output files

The results folder receives the stacked FITS and the extras ticked under **Output Files** (all off
by default):

- stack preview (PNG)
- frame count heat map (PNG)
- frame count map (FITS, `_coverage` added to the name)
- per-light report (CSV: status, sharpness, weight, alignment method, matched stars, residual)
- aligned lights, one FITS per light
- detailed activity log

If an extra can't be saved, the job still finishes and the stack is kept. An amber panel above the
log names each missing file and the reason.

### Aligned lights

A stack normally keeps no aligned copies of the lights, which keeps its disk use small. **Aligned
lights (FITS)** saves each light that went into the stack, after the stack, to a folder next to it
named after the stack with `_aligned` added, for example
`Light_NGC 7380_30.0s_LP_20261003-213800_aligned.fits`.

Each file has the size of the framing, about 12 MB per megapixel:

| Framing | Each light | 337 lights |
|---|---:|---:|
| S50 Pro reference frame (2160 × 3840) | about 100 MB | about 33 GB |
| 4313 × 4553 canvas | about 236 MB | about 80 GB |

Cropping on the heat map reduces this. The log shows the size before stacking; if the files don't
fit on the drive, only the stack is saved and a warning says why.

- The files hold the pixels that went into the stack: warped, debayered and brightness-matched with
  the chosen settings, before rejection. With equal weights and no feathering, averaging them with
  the same rejection gives the stack.
- All files share the stack's framing and line up pixel for pixel. Areas a light doesn't cover are 0.
- Feathering and frame weights only affect how lights are combined, so they aren't in these files.
- Each header has that light's time, pointing and exposure; `SRCLIGHT` names the original file.
- Cancelling while they are saved keeps the stack and the finished files.

<!-- Debayered lights: experimental, not shown.

### Debayered lights

**Debayered lights (FITS)** saves each light that went into the stack in colour, using the Colour
method setting, to a folder next to the stack with `_debayered` added to the name. They aren't
aligned or brightness-matched and keep their original size: about 100 MB per S50 Pro light and
25 MB per S50 light. Without darks and flats, only the camera offset is removed; a thin band around
the edge is 0. `ADUOFFS` is that light's camera offset.
-->

### File names

Stacks are named after the target (`OBJECT`), the number of lights, the exposure, the observation
dates and the time of stacking, for example
`HIP_106890_172x30sec_2026-10-03_2026-10-22_124500_gpucoadd.fits`. Mixed exposures list each one,
largest total first (`NGC_7023_205x30sec+203x10sec_...`). Without `OBJECT`, the folder name is used.

File names keep letters of any language. The `OBJECT` header allows only ASCII, so `Čačak` becomes
`Cacak` there.

Siril can fail to open a file from a very long path; the file is still saved.

### FITS contents

- `NCOMBINE`: number of contributing lights. `TOTINPUT`: summed input exposure, not the exposure at
  every pixel.
- Pixels are `(ADU + ADUOFFS) / ADUSCALE` in 0..1. `ADUOFFS` is the reference light's camera offset,
  so no real pixel is 0; `ADUSCALE` is 2 × 65535, leaving headroom for bright star cores. Uncovered
  areas are 0.
- For plate solving and colour calibration (for example plate solving and SPCC in Siril): `RA`/`DEC` (pointing of the light nearest the centre, a starting point only),
  `FOCALLEN`, `XPIXSZ`/`YPIXSZ`, `DATE-OBS`/`DATE-END`, and `FILTER`, `TELESCOP` and the site when
  the lights agree.
- Rows follow the FITS standard (first row at the bottom), declared with `ROWORDER = 'BOTTOM-UP'`.
  `ORIGINX`/`ORIGINY` give the canvas's top-left corner in the reference light's pixels.

GPUCoadd doesn't colour-calibrate; that is done on the stack in Siril or another program.

## Frame counts and mosaics

Each light is matched to the reference first and, if that fails, to overlapping lights that are
already aligned, nearest pointing first. Every accepted match needs at least ten star pairs spread
over the field, a plausible scale and a low residual.

The run's `report.json` keeps the counts apart:

- `input_frames`: lights in the selected folders and ticked groups
- `groups`: per light group, its counts, brightness match and value in main-group lights
- `aligned` / `alignment_failed`: placed / not placed on the common grid
- `sharpness_rejected`: aligned lights left out as too blurry
- `retained`: lights kept across the full field
- `contributing`: retained lights with data inside the crop (stacks only)
- `kernels`: per GPU stage, the path used and why (e.g. a kernel that couldn't be built)

A small crop can have fewer contributing lights than the field retains, especially in a mosaic.
That is not an alignment failure.

## Errors

A failure shows a plain summary and a suggested next step. **Error Details…** opens the full report,
with **Copy Error Details** and **Export Error Log…**. Reports are saved even with the activity log
off. They contain local paths, settings and recent messages; review them before sharing.

Lights are checked for changes before and after reading, during processing and before saving. If a
light changes, disappears or can't be read, processing stops and the report names it; no light is
left out silently. Lights must not be changed while a job runs.

After an update that changes the engine, the heat map must be built again before stacking.

## Files GPUCoadd keeps

GPUCoadd writes nothing into light folders, and with **Another folder** ticked nothing into the
target folder, including aligned lights, which go to the results folder. Its own files are in
`%LOCALAPPDATA%\GPUCoadd`: each run's `report.json`, the heat map with its alignment, error reports,
and the folders the window restores. Temporary request and worker files are deleted when done. Run
folders are removed after 30 days or beyond the newest 100, except the heat map in use. Stacking
again after closing the window needs a new heat map.
