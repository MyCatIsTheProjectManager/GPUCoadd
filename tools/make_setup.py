"""Builds the GPUCoadd setup program: dist/GPUCoadd-<version>-windows.exe.

1. Resolves every pinned package of requirements-lock.txt to the exact Windows / Python 3.11 file
   (PyPI, and download.pytorch.org for the CUDA build of torch) and a standalone Python 3.11 at
   least two weeks old (python-build-standalone), with URL, size and SHA-256: setup/packages.json.
2. Packs the runtime files (engine, worker, the window and its launcher) with the README, guide and technical
   reference into app.zip; no tests, tools or measurement data.
3. Wraps setup/Setup.ps1, packages.json and app.zip into one self-extracting .exe with Windows'
   own IExpress. The setup downloads the packages on the user's PC and verifies each one.

  python tools/make_setup.py            resolve (network) and build
  python tools/make_setup.py --offline  build with the existing setup/packages.json
"""

import argparse
import hashlib
import json
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup"
BUILD = Path(tempfile.gettempdir()) / "gpucoadd_setup_build"  # IExpress cannot handle spaces in paths
DIST = ROOT / "dist"
SHIP = ["LICENSE", "THIRD-PARTY-NOTICES.txt", "VERSION", "requirements-lock.txt", "start_gpucoadd.pyw",
        "README.md", "docs/GUIDE.md", "docs/TECHNICAL.md"]
SHIP_DIRS = ["worker", "GPUCoadd"]


def context():
    """Python 3.11.4 cannot read some Windows certificate stores; pip's own bundle always works."""
    try:
        return ssl.create_default_context()
    except ssl.SSLError:
        from pip._vendor import certifi

        return ssl.create_default_context(cafile=certifi.where())


CTX = None


def fetch(url, method="GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "GPUCoadd-make-setup"})
    return urllib.request.urlopen(req, timeout=60, context=CTX)


def standalone_python():
    """The newest python-build-standalone release at least 14 days old with a Python 3.11 build."""
    releases = json.load(fetch("https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=30"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    for rel in releases:
        if rel["draft"] or rel["prerelease"]:
            continue
        if datetime.fromisoformat(rel["published_at"].replace("Z", "+00:00")) > cutoff:
            continue
        for a in rel["assets"]:
            n = a["name"]
            if re.fullmatch(r"cpython-3\.11\.\d+\+\d+-x86_64-pc-windows-msvc-install_only_stripped\.tar\.gz", n):
                digest = (a.get("digest") or "").removeprefix("sha256:")
                if not digest:
                    sums = fetch(next(x["browser_download_url"] for x in rel["assets"] if x["name"] == "SHA256SUMS"))
                    digest = next(line.split()[0] for line in sums.read().decode().splitlines() if line.endswith(n))
                version = n.split("-")[1].split("+")[0]
                return {"name": "Python", "version": version, "file": n, "url": a["browser_download_url"],
                        "size": a["size"], "sha256": digest, "source": "github.com (python-build-standalone)"}
    raise SystemExit("no python-build-standalone release with Python 3.11 is at least 14 days old")


def best_wheel(files):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    rank = {str(t): k for k, t in enumerate(sys_tags())}  # this machine: Windows, CPython 3.11, 64-bit
    best = None
    for f in files:
        if not f["filename"].endswith(".whl"):
            continue
        _, _, _, tags = parse_wheel_filename(f["filename"])
        score = min((rank[str(t)] for t in tags if str(t) in rank), default=None)
        if score is not None and (best is None or score < best[0]):
            best = (score, f)
    return best and best[1]


def pypi_package(name, version):
    data = json.load(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    f = best_wheel(data["urls"])
    if f is None:
        raise SystemExit(f"{name} {version}: no wheel for Windows / Python 3.11")
    return {"name": data["info"]["name"], "version": version, "file": f["filename"], "url": f["url"],
            "size": f["size"], "sha256": f["digests"]["sha256"], "source": "pypi.org"}


def pytorch_package(name, version, index):
    page = fetch(f"{index.rstrip('/')}/{name}/").read().decode()
    files = []
    for href in re.findall(r'href="([^"]+)"', page):
        url, _, frag = href.partition("#sha256=")
        filename = urllib.request.unquote(url.rsplit("/", 1)[-1])
        if filename.startswith(f"{name}-{version}-"):
            files.append({"filename": filename, "url": urllib.parse.urljoin(index.rstrip("/") + f"/{name}/", url),
                          "sha256": frag})
    f = best_wheel(files)
    if f is None:
        raise SystemExit(f"{name} {version}: no wheel for Windows / Python 3.11 on {index}")
    size = int(fetch(f["url"], "HEAD").headers["Content-Length"])
    return {"name": name, "version": version, "file": f["filename"], "url": f["url"], "size": size,
            "sha256": f["sha256"], "source": "download.pytorch.org"}


def resolve():
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    index = re.search(r"--extra-index-url\s+(\S+)", lock).group(1)
    packages = []
    for line in lock.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name, version = line.split("==")
        print(f"  {name} {version}", flush=True)
        packages.append(pytorch_package(name, version, index) if "+" in version else pypi_package(name, version))
    python = standalone_python()
    manifest = {
        "about": "Files the GPUCoadd setup downloads and verifies (SHA-256). Made by tools/make_setup.py.",
        "requirements_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "python": python,
        "packages": packages,
    }
    (SETUP / "packages.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    total = python["size"] + sum(p["size"] for p in packages)
    print(f"Python {python['version']} + {len(packages)} packages: {total / 1e9:.2f} GB to download")


def shipped_files():
    # tracked files and new ones not committed yet; anything git ignores stays out
    listed = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    tracked = subprocess.run(listed, cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    keep = []
    for f in tracked:
        if Path(f).suffix.lower() == ".md" and f not in SHIP:  # other Markdown stays in the repository
            continue
        top = f.split("/")[0]
        if (f.endswith(".py") and "/" not in f) or f in SHIP or top in SHIP_DIRS:
            keep.append(f)
    return sorted(set(keep) | {"VERSION"})


def build():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    manifest = json.loads((SETUP / "packages.json").read_text(encoding="utf-8"))
    lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    if manifest["requirements_sha256"] != hashlib.sha256(lock.encode()).hexdigest():
        raise SystemExit("setup/packages.json does not match requirements-lock.txt: run without --offline")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    files = shipped_files()
    with zipfile.ZipFile(BUILD / "app.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(ROOT / f, f)
    shutil.copy(SETUP / "Setup.ps1", BUILD / "Setup.ps1")
    shutil.copy(SETUP / "packages.json", BUILD / "packages.json")
    # IExpress removes its temporary files when this returns, so the setup runs to the end here
    (BUILD / "start.cmd").write_text(
        f'@powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1" -Version "{version}"\r\n',
        encoding="ascii",
    )
    DIST.mkdir(exist_ok=True)
    final = DIST / f"GPUCoadd-{version}-windows.exe"
    target = BUILD / "setup.exe"
    names = ["Setup.ps1", "packages.json", "app.zip", "start.cmd"]
    sed = "\r\n".join(
        [
            "[Version]", "Class=IEXPRESS", "SEDVersion=3",
            "[Options]", "PackagePurpose=InstallApp", "ShowInstallProgramWindow=1", "HideExtractAnimation=1",
            "UseLongFileName=1", "InsideCompressed=0", "CAB_FixedSize=0", "CAB_ResvCodeSigning=0", "RebootMode=N",
            "InstallPrompt=%InstallPrompt%", "DisplayLicense=%DisplayLicense%", "FinishMessage=%FinishMessage%",
            "TargetName=%TargetName%", "FriendlyName=%FriendlyName%", "AppLaunched=%AppLaunched%",
            "PostInstallCmd=%PostInstallCmd%", "AdminQuietInstCmd=%AdminQuietInstCmd%",
            "UserQuietInstCmd=%UserQuietInstCmd%", "SourceFiles=SourceFiles",
            "[Strings]", "InstallPrompt=", "DisplayLicense=", "FinishMessage=", f"TargetName={target}",
            f"FriendlyName=GPUCoadd {version} setup", "AppLaunched=start.cmd", "PostInstallCmd=<None>",
            "AdminQuietInstCmd=", "UserQuietInstCmd=",
            *[f'FILE{k}="{n}"' for k, n in enumerate(names)],
            "[SourceFiles]", f"SourceFiles0={BUILD}\\",
            "[SourceFiles0]", *[f"%FILE{k}%=" for k in range(len(names))],
            "",
        ]
    )
    (BUILD / "setup.sed").write_text(sed, encoding="ascii")
    target.unlink(missing_ok=True)
    subprocess.run(["iexpress.exe", "/N", "/Q", str(BUILD / "setup.sed")], check=True)
    if not target.is_file():
        raise SystemExit("IExpress did not produce the setup program")
    shutil.move(target, final)
    target = final
    print(f"{target} ({target.stat().st_size / 1e6:.1f} MB): {len(files)} GPUCoadd files, "
          f"{len(manifest['packages'])} packages to download")


def main():
    global CTX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="reuse setup/packages.json instead of resolving it")
    args = ap.parse_args()
    if not args.offline:
        CTX = context()
        print("Resolving packages:", flush=True)
        resolve()
    build()


if __name__ == "__main__":
    sys.exit(main())
