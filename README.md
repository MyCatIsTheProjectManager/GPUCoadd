# GPUCoadd

## What it does

GPUCoadd stacks the lights from a Seestar S50 or S50 Pro into one image, in its own window or from
inside Siril. It shows a heat map of how many lights cover each part of the sky, allows a crop to be selected,
and saves the stack as a linear FITS, ready for stretching and colour calibration in Siril or another image processing program.
Optionally, it also saves every light aligned to the stack.

<img width="1641" height="891" alt="GPUCoadd window" src="https://github.com/user-attachments/assets/d709ec3f-71b2-40f4-a3d3-21390f311199" />

## Built on

- [astroalign](https://github.com/quatrope/astroalign) (Martin Beroiz and contributors, MIT): star
  matching; part of its code is adapted in `registration_fast.py`.
- [RCD](https://github.com/LuisSR/RCD-Demosaicing) (Luis Sanz Rodriguez, GPL-3.0-or-later): the
  debayering algorithm, rewritten for PyTorch and CUDA.
- [PyTorch](https://pytorch.org), [Astropy](https://www.astropy.org), [NumPy](https://numpy.org),
  [SciPy](https://scipy.org), [OpenCV](https://opencv.org) and, for the window,
  [PyQt6](https://www.riverbankcomputing.com/software/pyqt/).
- Developed with the help of Claude, an AI model by Anthropic, used for writing and reviewing code,
  writing tests, and measuring and comparing results.

[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt) lists every library with its licence, and the
published methods GPUCoadd implements.

## What runs on the CPU

- Reading the light files and checking that they don't change during a job
- Matching stars between lights (alignment)
- Measuring star sharpness, from small cut-outs found on the GPU
- Solving the brightness corrections across all lights for Overlap matching
- Counting how many lights cover each pixel (the heat map)
- Writing the FITS and PNG files, and reading the FITS back to verify every byte

## What runs on the GPU

- Subtracting the camera offset
- Finding stars and measuring their brightness
- Measuring each light's background, and comparing lights where they overlap (Overlap matching)
- Debayering (RCD or bilinear)
- Warping each light onto the stack's pixel grid
- Applying the brightness corrections, rejecting outliers and averaging
- Rendering aligned lights and turning them into FITS bytes

Lights that don't fit in GPU memory are kept in RAM or read again from their files. They still go
through the GPU in pieces.

## Limitations

- **No calibration frames yet.** Darks, flats and bias aren't applied; only the camera offset is
  subtracted. Calibrated lights are planned.
- **Work in progress.** This is a beta. When a job fails, **Error Details** shows the explanation and
  a shareable report.
- **Windows and NVIDIA only.** It needs a GTX 16xx, RTX 20xx or newer GPU. AMD and Intel GPUs aren't
  supported.
- **One filter per stack.** Added folders must be taken through the same filter; this isn't always
  detectable from the headers.
- **No colour calibration.** Done on the stack in Siril or another program.
- Experimental choices, marked so in the window: Star brightness and Overlap matching, noise-based
  weights, noise-aware rejection, homography alignment and gradient matching.

Original lights are never modified.

## Requirements

- Windows 10 or 11 (64-bit)
- An NVIDIA GTX 16xx, RTX 20xx or newer GPU with an up-to-date driver
- Siril 1.4 with Python scripting, only to open GPUCoadd from inside Siril
- Internet access and about 8 GB of free disk space for installation, plus space for the results

## Install

Run `GPUCoadd-0.10.0-beta-windows.exe`, choose an install folder, and wait for the GPU test to pass.
Setup installs everything it needs, including Python, and adds **GPUCoadd** to the Start menu.

- **On its own:** open **GPUCoadd** from the Start menu.
- **From Siril (optional):** under **Preferences > Scripts**, add the `GPUCoadd` folder inside the
  install folder. Restart Siril, then choose **Scripts > Python Scripts > GPUCoadd > GPUCoadd.py**.

Both open the same window.

## Make a stack

1. Choose the **Target folder**: the folder that contains the `lights` folder.
2. Click **Build Heat Map**.
3. Keep **Full Frame**, or drag on the map to select a crop.
4. Click **Stack Full Frame** or **Stack Selected Area**.
5. Choose the finished image under **Saved images**, then click **Open Selected Image**.

The stack is saved in the target folder unless another results folder is chosen.

The [user guide](docs/GUIDE.md) explains every setting. The [technical reference](docs/TECHNICAL.md)
covers how it works, setting it up from the command line and running the tests.

## Licence

Copyright (C) 2026 MyCatIsTheProjectManager

GPUCoadd is free software, licensed under the GNU General Public License version 3 or any later
version (GPL-3.0-or-later). It comes with no warranty. The full text is in [LICENSE](LICENSE).
