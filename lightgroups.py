"""Light groups for mixed-instrument stacking: lights from one or more folders are grouped by
folder, telescope, filter, exposure, gain, sensor size and Bayer pattern (FITS headers). One
main group sets the filter; groups with another filter are listed but never stacked with it
(LP and IR are different data, not more of the same). To stack another filter, leave out the
groups of the current one. Folders stay apart because a header cannot show an external filter
(an IRCUT light may have been taken through one), so each folder can be switched off."""

import math
import re
from pathlib import Path

from fits_input import PATTERNS, Layout

SUPPORTED_BAYER = PATTERNS  # read into the engine's GRBG layout (fits_input)
LIGHT_SUFFIXES = (".fit", ".fits", ".fts")


def _number(h, *keys):
    for k in keys:
        try:
            v = float(h[k])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v):
            return v
    return None


def _text(h, key):
    return str(h.get(key, "") or "").strip()


def telescope_name(telescop):
    """'S50 Pro_00000000' -> 'S50 Pro' (Seestar appends a unit id)."""
    return re.sub(r"_[0-9a-fA-F]{6,}$", "", telescop).strip() or "Unknown telescope"


def fits_files(folder):
    """The FITS files (.fit, .fits, .fts) directly in `folder`, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in LIGHT_SUFFIXES and p.is_file())


def light_paths(folders):
    """The FITS lights of each folder (sorted by name within a folder, folders in the given
    order); a file reached twice (same folder listed twice) is used once."""
    seen, paths = set(), []
    for folder in folders:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise ValueError(f"Light folder not found: {folder}")
        found = fits_files(folder)
        if not found:
            raise ValueError(f"No FITS lights (.fit, .fits, .fts) found in {folder}")
        for p in found:
            key = str(p.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                paths.append(p)
    return paths


def describe(path, h):
    """One light's grouping attributes from its FITS header."""
    exposure = _number(h, "EXPTIME", "EXPOSURE")
    gain = _number(h, "GAIN")
    ra, dec = _number(h, "RA"), _number(h, "DEC")
    pixel, focal = _number(h, "XPIXSZ"), _number(h, "FOCALLEN")
    scale = 206.265 * pixel / focal if pixel and focal and pixel > 0 and focal > 0 else None
    layout = Layout(h)
    sensor = (int(h.get("NAXIS1", 0) or 0), int(h.get("NAXIS2", 0) or 0))
    width, height = sensor if layout.reason else layout.shape[::-1]
    bias = _number(h, "BIAS")
    return {
        "path": Path(path),
        "folder": str(Path(path).parent),
        "telescope": _text(h, "TELESCOP"),
        "filter": _text(h, "FILTER").upper() or "UNKNOWN",
        "exposure": exposure,
        "gain": gain,
        "width": width,  # the image the engine reads (see fits_input)
        "height": height,
        "sensor": sensor,  # as stored in the file
        "bayer": _text(h, "BAYERPAT").upper() or "UNKNOWN",
        "unusable": layout.reason or (None if layout.declared else "no Bayer pattern in the header (BAYERPAT)"),
        "conversion": layout.describe() if layout.reason is None else "",
        "bias": None if bias is None else bias * layout.scale,
        "pixel_scale": scale,  # arcsec per pixel from the optics (None if not in the header)
        "pointing": [float("nan") if ra is None else ra, float("nan") if dec is None else dec],
        "date": _text(h, "DATE-OBS")[:10],
    }


def group_key(light):
    exp = "?" if light["exposure"] is None else f"{light['exposure']:g}"
    gain = "?" if light["gain"] is None else f"{light['gain']:g}"
    return "|".join(
        [
            light["folder"],
            light["telescope"],
            light["filter"],
            exp,
            gain,
            f"{light['sensor'][0]}x{light['sensor'][1]}",
            light["bayer"],
        ]
        # lights that cannot be read form their own group, so the others can still be stacked
        + ([light["unusable"]] if light["unusable"] else [])
    )


def plan(lights, exclude=()):
    """Group `lights` (describe() records, in input order) and decide which groups are used.

    exclude: group keys the user switched off. Returns
    {"groups": [...], "main": key, "used": [light index...], "pedestal": {light index: ADU}}.
    Group order: by first appearance; used lights keep their input order."""
    exclude = set(exclude or ())
    groups = {}
    for i, light in enumerate(lights):
        groups.setdefault(group_key(light), []).append(i)
    unknown = exclude - set(groups)
    if unknown:
        raise ValueError("Unknown light group(s): " + "; ".join(sorted(unknown)))

    def area_time(key):  # sky area x total exposure: the group that defines the picture
        g = [lights[i] for i in groups[key]]
        return sum(l["width"] * l["height"] * (l["exposure"] or 0) for l in g)

    supported = [k for k in groups if not lights[groups[k][0]]["unusable"]]
    pool = [k for k in supported if k not in exclude] or supported
    if not pool:
        reasons = sorted({lights[i]["unusable"] for idx in groups.values() for i in idx if lights[i]["unusable"]})
        raise ValueError(
            "No lights GPUCoadd can read: raw Bayer images (" + ", ".join(SUPPORTED_BAYER) + ") are needed; "
            + "; ".join(reasons)
        )
    main = max(pool, key=lambda k: (area_time(k), -list(groups).index(k)))
    main_filter = lights[groups[main][0]]["filter"]

    # Pedestal (BIAS): each light's own header value; when missing, the same telescope's value
    # at the same gain from any light in the request; else 0. A constant offset is absorbed by
    # brightness matching; the pedestal only has to keep the raw data positive for debayering.
    known = {}
    for light in lights:
        if light["bias"] is not None:
            known.setdefault((light["telescope"], light["gain"]), light["bias"])

    table, used, pedestal = [], [], {}
    for key, idx in groups.items():
        first = lights[idx[0]]
        reason = None
        if first["unusable"]:
            reason = first["unusable"]
        elif key in exclude:
            reason = "left out by the user"
        elif first["filter"] != main_filter:
            reason = f"filter {first['filter']} differs from the main group's {main_filter}"
        has_bias = [lights[i]["bias"] is not None for i in idx]
        if all(has_bias):
            source = "BIAS header"
        elif (first["telescope"], first["gain"]) in known:
            source = "BIAS of the same telescope and gain in other lights"
        else:
            source = "none found (0)"
        for i in idx:
            b = lights[i]["bias"]
            if b is None:
                b = known.get((lights[i]["telescope"], lights[i]["gain"]), 0.0)
            pedestal[i] = float(b)
        if reason is None:
            used.extend(idx)
        exposures = [lights[i]["exposure"] for i in idx]
        total = sum(exposures) if all(e is not None for e in exposures) else None
        table.append(
            {
                "key": key,
                "label": label(first, len(idx)),
                "folder": first["folder"],
                "indices": list(idx),
                "telescope": telescope_name(first["telescope"]),
                "telescope_id": first["telescope"],
                "filter": first["filter"],
                "exposure_s": first["exposure"],
                "gain": first["gain"],
                "sensor": list(first["sensor"]),
                "bayer": first["bayer"],
                "conversion": sorted({lights[i]["conversion"] for i in idx} - {""}),
                "lights": len(idx),
                "total_exposure_s": total,
                "dates": sorted({lights[i]["date"] for i in idx if lights[i]["date"]}),
                "pedestal": sorted({pedestal[i] for i in idx}),
                "pedestal_source": source,
                "main": key == main,
                "used": reason is None,
                "reason": reason,
            }
        )
    used.sort()
    # Pixel scale per light (registration expects the ratio): the header's, else the group's,
    # else the main group's (a light without optics data is taken to be the main instrument)
    main_scale = next((lights[i]["pixel_scale"] for i in groups[main] if lights[i]["pixel_scale"]), 1.0)
    pixel_scale = {}
    for row, idx in zip(table, groups.values()):
        known_scale = next((lights[i]["pixel_scale"] for i in idx if lights[i]["pixel_scale"]), main_scale)
        for i in idx:
            pixel_scale[i] = lights[i]["pixel_scale"] or known_scale
        row["pixel_scale"] = pixel_scale[idx[0]]
    return {
        "groups": table,
        "main": main,
        "main_lights": list(groups[main]),
        "used": used,
        "pedestal": pedestal,
        "pixel_scale": pixel_scale,
    }


def label(light, count):
    exp = "?" if light["exposure"] is None else f"{light['exposure']:g} s"
    gain = "" if light["gain"] is None else f", gain {light['gain']:g}"
    folder = Path(light["folder"])
    where = f"{folder.parent.name}/{folder.name}" if folder.name.casefold() in ("lights", "light") else folder.name
    return (
        f"{telescope_name(light['telescope'])}, {light['filter']}, {count} x {exp}{gain}, "
        f"{light['sensor'][0]} x {light['sensor'][1]} ({where})"
    )
