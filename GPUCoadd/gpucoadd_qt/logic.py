"""The GPUCoadd window's behaviour without its widgets: folder discovery, light groups, requests to
the worker, worker events, stage circles, framing, notices and texts. tests/test_window_logic.py
checks it against tests/fixtures/dialog_contract.json. Nothing
here imports Qt or any host program's modules."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re

PROJECT_DIR = Path(__file__).resolve().parents[2]
BRIDGE = PROJECT_DIR / "worker" / "bridge.py"
HARDWARE_CHECK = PROJECT_DIR / "worker" / "hardware_check.py"
DEFAULT_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
IDLE_RELEASE_S = 270  # a kept worker idle longer than this is restarted before the next job (it stops at 300)


def work_dir():
    """GPUCoadd's own working area (request files, worker sessions, heat maps, run reports), kept
    out of the results folder: %LOCALAPPDATA%/GPUCoadd, as for the worker."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return norm_path(base) + "/GPUCoadd"


def duration(seconds):
    seconds = max(0, round(seconds))
    return f"{seconds // 60}m {seconds % 60}s"


def seconds_text(s):
    return f"{s:.1f} sec" if s < 60 else duration(s)


def exposure_text(seconds):
    minutes = round(seconds / 60)
    h, m = divmod(minutes, 60)
    return f"{h} h {m} min" if h else f"{m} min" if minutes else f"{round(seconds)} s"


def stamp():
    return datetime.now().strftime("%H:%M:%S")


# Folder convention: a target folder holds a "lights" folder of subs, and stacked results are saved
# in the target folder, next to it.
def norm_path(path):
    return re.sub(r"/+$", "", str(path or "").replace("\\", "/"))


def parent_path(path):
    path = norm_path(path)
    i = path.rfind("/")
    return path[:i] if i > 0 else path


def base_name(path):
    path = norm_path(path)
    return path[path.rfind("/") + 1 :]


def is_sub_copy_folder(path):
    """Seestar's "<target>_sub" folders hold thumbnails or copies of the lights: never read them."""
    return re.search(r"_sub$", base_name(path), re.IGNORECASE) is not None


# The engine's lightgroups.LIGHT_SUFFIXES, repeated: this module runs in the host's Python and must
# not import the engine (lightgroups imports numpy). tests/test_window_logic.py keeps them equal.
LIGHT_SUFFIXES = (".fit", ".fits", ".fts")


def no_lights_message(folder):
    return f"No FITS lights (.fit, .fits, .fts) in {folder}."


def has_lights(folder):
    try:
        with os.scandir(norm_path(folder)) as entries:
            return any(e.is_file() and e.name.lower().endswith(LIGHT_SUFFIXES) for e in entries)
    except OSError:
        return False


def is_dir(path):
    return os.path.isdir(norm_path(path))


def child_folders(folder):
    try:
        with os.scandir(norm_path(folder)) as entries:
            names = [e.name for e in entries if e.is_dir()]
    except OSError:
        return []
    return sorted(norm_path(folder) + "/" + name for name in names)


def find_light_folders(chosen):
    """The chosen folder as {target, lights, offered} or {error}. A target folder uses its "lights"
    folder; a picked lights folder (named "lights", or holding FITS lights itself) still works, with
    its parent as the target. Other "lights" folders one level down (a second telescope, e.g.
    "s50pro/lights") are only offered: the same filter name can hide different filters."""
    chosen = norm_path(chosen)
    if not chosen:
        return {"error": "Choose a target folder."}
    if is_sub_copy_folder(chosen):
        return {
            "error": 'Seestar "_sub" folders hold thumbnails or copies of the lights, so GPUCoadd never reads them. '
            'Choose the target folder (the one with "lights") instead.'
        }
    if re.fullmatch(r"lights", base_name(chosen), re.IGNORECASE) or (
        has_lights(chosen) and not is_dir(chosen + "/lights")
    ):
        target, lights = parent_path(chosen), chosen
    elif is_dir(chosen + "/lights"):
        target, lights = chosen, chosen + "/lights"
    else:
        return {
            "error": f'No "lights" folder in {chosen}. Choose the target folder that holds your lights folder, '
            "or the lights folder itself."
        }
    if not has_lights(lights):
        return {"error": no_lights_message(lights)}
    offered = []
    for child in child_folders(target):
        candidate = child + "/lights"
        if (
            not is_sub_copy_folder(child)
            and candidate.lower() != lights.lower()
            and is_dir(candidate)
            and has_lights(candidate)
        ):
            offered.append(candidate)
    return {"target": target, "lights": lights, "offered": offered}


class Folders:
    """The target folder, the light folders used (in order), those found but unticked, and the
    light groups the user switched off."""

    def __init__(self):
        self.target = ""
        self.folders = []
        self.offered = []
        self.excluded = []

    def set(self, folders, excluded=None, offered=None, target=None):
        """folders: used light folders; offered: found but unticked (kept unless given); target:
        the target folder, else kept, or the first folder's parent when there is none yet.
        Returns the hint shown under the group table."""
        used = []
        for f in folders:
            f = norm_path(f)
            if f and f not in used:
                used.append(f)
        others = []
        for g in self.offered if offered is None else offered:
            g = norm_path(g)
            if g and g not in used and g not in others:
                others.append(g)
        if target is not None:
            self.target = norm_path(target)
        elif not self.target and used:
            self.target = parent_path(used[0])
        if not used and not others and target is None:
            self.target = ""
        self.folders, self.offered = used, others
        # keep the choice for groups of folders that are still listed (a group key starts with its folder)
        previous = self.excluded if excluded is None else excluded
        self.excluded = [
            key for key in previous if any(key.replace("\\", "/").startswith(f + "|") for f in used)
        ]
        if not used:
            return "Choose a target folder to begin."
        more = f" {len(others)} more light folder(s) found in the target, not included: tick to add." if others else ""
        return "Check Folders lists the light groups found; Build Heat Map uses them." + more

    def toggle(self, folder, checked):
        folders, offered = list(self.folders), list(self.offered)
        if checked and folder not in folders:
            folders.append(folder)
            if folder in offered:
                offered.remove(folder)
        elif not checked and folder in folders:
            folders.remove(folder)
            offered.append(folder)
        return self.set(folders, None, offered)

    def remove(self, folder):
        return self.set([f for f in self.folders if f != folder], None, [f for f in self.offered if f != folder])

    def rows(self):
        """(folder, ticked, text) for the folder list."""
        return [(f, True, f) for f in self.folders] + [
            (f, False, f + "  (found; tick to include)") for f in self.offered
        ]

    def exclude(self, key, checked):
        if checked and key in self.excluded:
            self.excluded.remove(key)
        if not checked and key not in self.excluded:
            self.excluded.append(key)


def add_folder_problem(folder):
    """Why a folder picked with Add Folder cannot be used, or None."""
    if is_sub_copy_folder(folder):
        return 'Seestar "_sub" folders hold copies of the lights; add a "lights" folder instead.'
    if not has_lights(folder):
        return no_lights_message(norm_path(folder))
    return None


def group_rows(groups):
    """Rows of the light-group table and the hint below it (None without groups)."""
    rows, used, lights, used_groups = [], 0, 0, 0
    for g in groups:
        if g.get("used"):
            note = ("Main group: sets the filter and the alignment reference" if g.get("main") else
                    "Stacked with the main group")
        else:
            reason = g.get("reason") or ""
            note = reason[:1].upper() + reason[1:]
        if g.get("main_light_equivalents") is not None and not g.get("main"):
            note += f" · worth {g['main_light_equivalents']:.1f} main-group lights"
        rows.append(
            {
                "key": g.get("key"),
                # a group with another filter than the main group can never be ticked
                "checkable": bool(g.get("used")) or g.get("reason") == "left out by the user",
                "checked": bool(g.get("used")),
                "texts": [
                    g.get("label", "") + (" (main)" if g.get("main") else ""),
                    str(g.get("lights")),
                    exposure_text(g["total_exposure_s"]) if g.get("total_exposure_s") is not None else "?",
                    note,
                ],
            }
        )
        lights += g.get("lights") or 0
        if g.get("used"):
            used += g.get("lights") or 0
            used_groups += 1
    if not groups:
        return rows, None
    hint = f"{used} of {lights} lights in {used_groups} group(s) will be stacked."
    if used_groups > 1:
        hint += (
            " Mixed instruments: Overlap matching and noise weights are used automatically "
            "(noise-aware rejection becomes sigma clipping)."
        )
    return rows, hint


# Image-quality choices, in the order the dialog lists them.
CHOICES = {
    "debayer": ["rcd", "bilinear"],
    "interp": ["bicubic", "bilinear"],
    "norm": ["spread", "photometric", "overlap"],
    "weight": ["none", "noise"],
    "transform": ["similarity", "homography"],
    "rejection": ["sigma", "winsorized", "noise"],
}
DEFAULT_SHARPNESS = 2.5
# debayered_lights: experimental, not shown.
OUTPUT_KEYS = ["stack_preview", "coverage_png", "coverage_fits", "frames_csv", "aligned_lights",  # "debayered_lights",
               "activity_log"]


def quality_settings(choice, selection, sharpness, feather, rgb_equal, match_gradients):
    """The request's settings from the controls: `choice` maps each CHOICES key to its index."""
    settings = {}
    for key in ("debayer", "interp", "norm", "weight", "transform", "rejection"):
        index = choice.get(key, 0)
        settings[key] = CHOICES[key][index] if 0 <= index < len(CHOICES[key]) else CHOICES[key][0]
    settings.update(select=sharpness if selection else 0, feather=feather, rgb_equal=rgb_equal,
                    match_gradients=match_gradients)
    return settings


def quality_controls(settings):
    """Control values for saved settings ({} gives the recommended defaults)."""
    choice = {key: values.index(settings[key]) if settings.get(key) in values else 0 for key, values in CHOICES.items()}
    select = settings.get("select")
    feather = settings.get("feather")
    return {
        "choice": choice,
        "selection": isinstance(select, (int, float)) and select > 0,  # off unless a saved choice turned it on
        "sharpness": select if isinstance(select, (int, float)) and select > 0 else DEFAULT_SHARPNESS,
        "feather": feather if isinstance(feather, (int, float)) and feather > 0 else 0,
        "rgb_equal": settings.get("rgb_equal") is not False,  # on unless a saved choice turned it off
        "match_gradients": settings.get("match_gradients") is True,
    }


def option_availability(norm_index, groups, rgb_equal, rgb_auto_off, match_gradients):
    """RGB equalization needs Background spread and one light group; gradient matching needs
    Overlap matching (or several groups, which use it). An option turned off only because it was
    unavailable comes back when it is available again. Returns the new state and log messages."""
    used = sum(1 for g in groups if g.get("used"))
    messages = []
    rgb_available = norm_index == 0 and used <= 1
    if not rgb_available and rgb_equal:
        rgb_equal, rgb_auto_off = False, True
        messages.append("RGB equalization turned off: it requires Background spread and a single light group.")
    elif rgb_available and rgb_auto_off:
        rgb_equal, rgb_auto_off = True, False
        messages.append("RGB equalization turned back on.")
    gradients_available = norm_index == 2 or used > 1
    if not gradients_available and match_gradients:
        match_gradients = False
        messages.append("Gradient matching turned off: it requires Overlap matching.")
    return {
        "rgb_equal": rgb_equal,
        "rgb_auto_off": rgb_auto_off,
        "rgb_available": rgb_available,
        "match_gradients": match_gradients,
        "gradients_available": gradients_available,
        "messages": messages,
    }


def output_plan_text(options):
    images = ["stacked image (FITS)"]
    if options.get("coverage_fits"):
        images.append("frame count map (FITS)")
    if options.get("stack_preview"):
        images.append("stack preview (PNG)")
    if options.get("coverage_png"):
        images.append("frame count heat map (PNG)")
    aligned = " Aligned lights add one FITS per light, about 12 MB per megapixel of your framing."
    # Debayered lights: experimental, not shown.
    # debayered = " Debayered lights add one colour FITS per light, about 100 MB per S50 Pro light."
    return (
        f"Next stack saves {len(images)} image file(s): {', '.join(images)}."
        + (aligned if options.get("aligned_lights") else "")
        # + (debayered if options.get("debayered_lights") else "")
    )


# (status line, log line) when the heat map goes out of date
HEAT_MAP_OUTDATED = (
    "Inputs changed: click Build Heat Map again before stacking.",
    "Heat map marked out of date. No analysis started automatically.",
)
HEAT_MAP_LIMIT_CHANGED = (
    "Sharpness limit changed: click Build Heat Map again before stacking (alignment is reused).",
    "Heat map marked out of date by the sharpness limit; the next heat map reuses its star detection and "
    "alignment. No analysis started automatically.",
)


def build_request(operation, folders, excluded, output_dir, work, reference, settings, outputs, threads, tile,
                  analysis_file=None, region=None, previous_analysis=None):
    """The request file's content, field for field in the order of tests/fixtures/dialog_contract.json.
    previous_analysis: a heat map's analysis.json outdated only by the sharpness limit; the worker
    reuses its star detection and alignment when the lights, reference and transform still match."""
    request = {
        "operation": operation,
        "lights_dirs": list(folders),
        "exclude_groups": list(excluded),
        "output_dir": output_dir,
        "work_dir": work,
        "reference": reference,
        "settings": dict(settings),
        "outputs": dict(outputs),
    }
    request["settings"]["threads"] = threads
    request["settings"]["tile"] = tile
    if operation == "stack":
        request["analysis_file"] = analysis_file
        request["region"] = list(region)
    elif operation == "analyze" and previous_analysis:
        request["previous_analysis"] = previous_analysis
    return request


def finish_warnings(operation, warnings):
    """(status line, panel text) for a completed job's warnings: requested outputs that were not
    saved, and GPU stages that ran on a slower path because their kernel could not be built
    (kind "performance"; the results are complete either way)."""
    outputs = [w for w in warnings if w.get("kind") != "performance"]
    slow = [w for w in warnings if w.get("kind") == "performance"]
    parts = []
    if outputs:
        title = ("Stack saved, but requested outputs are missing or incomplete:"
                 if operation == "stack" else "Completed with output warnings:")
        parts.append("\n\n".join([title] + [w["message"] for w in outputs]))
    if slow:
        parts.append("\n\n".join(
            ["Completed, but slower than it should be:"] + [w["message"] for w in slow]
            + ["The results are complete. The run report (report.json, under \"kernels\") has each stage's "
               "path and error; an NVIDIA driver update may fix the build."]
        ))
    if outputs:
        status = ("Stack saved with warnings: check missing outputs below."
                  if operation == "stack" else "Completed with warnings: see below.")
    else:
        status = ("Stack saved, but slower than it should be: see below."
                  if operation == "stack" else "Completed, but slower than it should be: see below.")
    return status, "\n\n".join(parts)


def job_problem(operation, folders, results_folder, custom_output, python, reference_on, reference, analysis, region):
    """Why a job cannot start, or None (the checks in the order the window shows them)."""
    if not folders:
        return 'Choose a target folder (the one holding "lights") first, or tick a light folder.'
    if not results_folder:
        return ('Choose a results folder, or untick "Another folder" to save in the target folder.' if custom_output
                else "Choose a target folder first: results are saved there.")
    if not os.path.isfile(python):
        return "Python executable not found. Check Engine settings."
    if reference_on and not reference.strip():
        return "Choose a reference light, or turn off the optional reference."
    if operation == "stack" and (not analysis or not region):
        return "Build the heat map and select framing first."
    return None


def split_events(buffer, chunk):
    """Worker output so far: complete lines as events (non-JSON lines become {"type": "text"}) and
    the unfinished rest."""
    lines = (buffer + chunk).split("\n")
    rest = lines.pop()
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError
        except ValueError:
            event = {"type": "text", "message": line.rstrip("\r")}
        events.append(event)
    return events, rest


def same_request(event, request_file):
    """Results name their request file; older workers (no name) are matched by the busy state alone."""
    if not event.get("request_file"):
        return True

    def norm(path):
        return str(path or "").replace("\\", "/").lower()

    return norm(event["request_file"]) == norm(request_file)


def request_files(request_file):
    """The request, its result and any cancel flag: only messages to the worker, removed after the job."""
    base = re.sub(r"\.json$", "", request_file)
    return [request_file, base + ".result.json", base + ".cancel"]


def cancel_file(request_file):
    return re.sub(r"\.json$", ".cancel", request_file)


def result_file(request_file):
    return re.sub(r"\.json$", ".result.json", request_file)


# Stage overview: one circle per stage of the running operation that fills with the stage's
# progress. Stages the worker reports are grouped under short names.
STAGE_GROUPS = [
    ("loading lights", "Loading lights"),
    ("frame analysis", "Detecting stars"),
    ("registration", "Registering"),
    ("linking light groups", "Linking groups"),
    ("neighbour alignment", "Neighbour alignment"),
    ("preparing overlap matching", "Overlap matching"),
    ("frame count", "Frame count"),
    ("stacking", "Stacking"),
    ("saving outputs", "Saving"),
]
PLANNED_STAGES = {
    "analyze": ["Loading lights", "Detecting stars", "Registering", "Frame count", "Saving"],
    "stack": ["Loading lights", "Stacking", "Saving"],
    "scan": ["Checking folders"],
}


def stage_name(stage):
    for prefix, name in STAGE_GROUPS:
        if stage.startswith(prefix):
            return name
    return stage[:1].upper() + stage[1:]


class Stages:
    """Stage circles: name, state (pending, running, done, failed, skipped), fraction, start/end
    (seconds), step ("3/10") and note. `now` is passed in so tests control time."""

    def __init__(self):
        self.stages = []

    def reset(self, operation, start, now):
        self.stages = [dict(name=n, state="pending", fraction=0.0) for n in PLANNED_STAGES.get(operation, [])]
        if start and operation == "scan":
            self.stages[0].update(state="running", start=now)

    def event(self, stage, done, total, now):
        name = stage_name(stage)
        current = next((s for s in self.stages if s["name"] == name), None)
        if current is None:  # not planned (e.g. Overlap matching): before the first stage that has not started
            current = dict(name=name, state="pending", fraction=0.0)
            at = next((i for i, s in enumerate(self.stages) if s["state"] == "pending"), len(self.stages))
            self.stages.insert(at, current)
        for s in self.stages:  # a new stage ends the one running before it
            if s is not current and s["state"] == "running":
                s.update(state="done", end=now)
        if current["state"] != "running":
            current.update(state="running", end=None)
            if not current.get("start"):
                current["start"] = now
        current["fraction"] = done / total if total else 0.0
        # numbered steps ("step 3 of 10") restart the circle's progress; show which step this is
        step = re.search(r"step (\d+) of (\d+)", stage)
        current["step"] = f"{step.group(1)}/{step.group(2)}" if step else None

    def finish(self, status, now):
        for s in self.stages:
            if s["state"] == "running":
                s["end"] = now
                if status == "complete":
                    s["state"] = "done"
                else:
                    s.update(state="failed", note="Cancelled" if status == "cancelled" else "Failed")
            elif s["state"] == "pending":
                s.update(state="skipped", note="Not needed" if status == "complete" else "Not run")

    @staticmethod
    def time_text(stage, now):
        end = stage.get("end") or now
        start = stage.get("start")
        if stage["state"] == "running":
            return (f"Step {stage['step']} · " if stage.get("step") else "Working: ") + seconds_text(end - start)
        if stage["state"] == "done":
            return "Finished in: " + seconds_text(end - (start or end))
        if stage["state"] == "failed":
            return stage.get("note", "") + (f" after {seconds_text(end - start)}" if start else "")
        if stage["state"] == "skipped":
            return stage.get("note") or "Not needed"
        return "Waiting"

    @staticmethod
    def fill(stage):
        if stage["state"] == "done":
            return 1.0
        if stage["state"] == "running":
            return max(0.0, min(1.0, stage.get("fraction") or 0.0))
        return 0.0

    def as_dicts(self):
        return [dict(s) for s in self.stages]


STORAGE_NAMES = {"raw": "light storage", "sky": "overlap matching data", "output": "output buffers"}


class StorageNotices:
    """The amber notice when data is held in RAM or on disk instead of the GPU."""

    def __init__(self):
        self.notices = {}

    def reset(self):
        self.notices = {}

    def add(self, notice):
        """Returns (title, message), or None when nothing needs saying."""
        if notice.get("component") in STORAGE_NAMES:
            self.notices[notice["component"]] = notice
        return self.text()

    def text(self):
        disk, ram = [], []
        budget_disk = budget_ram = False
        forced_disk = forced_ram = True
        for key, name in STORAGE_NAMES.items():
            item = self.notices.get(key)
            if not item:
                continue
            if item.get("mode") in ("disk", "mixed"):
                disk.append(name)
                budget_disk = budget_disk or item.get("reason") == "capacity"
                forced_disk = forced_disk and item.get("reason") == "forced"
            elif item.get("mode") == "ram" and key != "output" and item.get("reason") != "normal":
                ram.append(name)
                budget_ram = budget_ram or item.get("reason") == "capacity"
                forced_ram = forced_ram and item.get("reason") == "forced"
        output = self.notices.get("output")
        low_ram = output and output.get("reason") == "capacity" and output.get("free_ram_bytes") is not None
        if disk and low_ram:
            # the user can act on this: closing other programs keeps the output in RAM next time
            return (
                "Low on RAM: using disk storage (same image quality)",
                f"Only {gb(output['free_ram_bytes'])} of RAM was free and the output needed "
                f"{gb(output.get('needed_ram_bytes') or 0)}, so this operation uses temporary disk storage for: "
                f"{', '.join(disk)}. Image quality and settings are unchanged; it may take longer. "
                "To keep it in RAM, close other programs and run it again.",
            )
        if disk:
            lead = ("To stay within the available GPU/RAM budgets, " if budget_disk else
                    "As selected, " if forced_disk else "For this operation, ")
            return (
                "Disk-backed processing: same image quality",
                lead + "this operation uses disk-backed storage. Processing settings and image quality are unchanged; "
                f"processing may take longer. Used for: {', '.join(disk)}.",
            )
        if ram:
            lead = ("To stay within the available GPU memory budget, " if budget_ram else
                    "As selected, " if forced_ram else "For this operation, ")
            return (
                "RAM-backed processing: same image quality",
                lead + "some data is held in system RAM. Processing settings and image quality are unchanged; "
                f"processing may take longer. Used for: {', '.join(ram)}.",
            )
        return None

    def restore(self, result):
        """The notices of a finished (or saved) result; older reports carry only storage modes."""
        if result.get("storage_notices") is not None:
            for notice in result["storage_notices"]:
                self.add(notice)
        else:
            fields = {
                "raw": result.get("storage"),
                "output": result.get("output_storage"),
                "sky": (result.get("normalization") or {}).get("storage"),
            }
            for key, field in fields.items():
                if field and field.get("mode"):
                    normal = field["mode"] == "gpu" or (key == "output" and field["mode"] == "ram")
                    self.add({"component": key, "mode": field["mode"], "reason": "normal" if normal else "selected"})
        return self.text()


def gb(value):
    return f"{value / 1073741824:.1f} GB"


# Framing: the heat map drawn into a widget, and crops in canvas pixels [x, y, width, height].
def display_rect(width, height, image_width, image_height):
    if not image_width or not image_height:
        return (0.0, 0.0, float(width), float(height))
    s = min((width - 12) / image_width, (height - 12) / image_height)
    w, h = image_width * s, image_height * s
    return ((width - w) / 2, (height - h) / 2, w, h)


def sky_point(rect, canvas, x, y):
    cw, ch = canvas["width"], canvas["height"]
    return (
        max(0.0, min(cw, (x - rect[0]) * cw / rect[2])),
        max(0.0, min(ch, (y - rect[1]) * ch / rect[3])),
    )


def inside(rect, x, y):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


def drag_start(region, canvas, point):
    """The crop being moved when a drag starts inside it, else None (a new crop is drawn)."""
    a = region
    cropped = a and (a[0] != 0 or a[1] != 0 or a[2] != canvas["width"] or a[3] != canvas["height"])
    if cropped and a[0] <= point[0] < a[0] + a[2] and a[1] <= point[1] < a[1] + a[3]:
        return list(a)
    return None


def drag_region(start, moving, canvas, point):
    """The crop while dragging from `start` to `point`: `moving` is the crop being moved, if any."""
    cw, ch = canvas["width"], canvas["height"]
    if moving:
        return [
            max(0, min(cw - moving[2], moving[0] + js_round(point[0] - start[0]))),
            max(0, min(ch - moving[3], moving[1] + js_round(point[1] - start[1]))),
            moving[2],
            moving[3],
        ]
    x0 = min(cw - 1, math.floor(min(start[0], point[0])))
    y0 = min(ch - 1, math.floor(min(start[1], point[1])))
    return [
        x0,
        y0,
        max(1, math.ceil(max(start[0], point[0])) - x0),
        max(1, math.ceil(max(start[1], point[1])) - y0),
    ]


def js_round(value):
    """Math.round: halves go up (Python's round goes to even)."""
    return math.floor(value + 0.5)


def clamp_region(values, canvas):
    """X, Y, Width, Height as typed, kept on the canvas."""
    v = [int(x) for x in values]
    v[0] = min(v[0], canvas["width"] - 1)
    v[1] = min(v[1], canvas["height"] - 1)
    v[2] = min(v[2], canvas["width"] - v[0])
    v[3] = min(v[3], canvas["height"] - v[1])
    return v


def full_frame(canvas):
    return [0, 0, canvas["width"], canvas["height"]]


def is_full_frame(region, canvas, edited):
    return not edited and list(region) == full_frame(canvas)


def stack_caption(full, region):
    size = f" ({region[2]} × {region[3]})" if region else ""
    return ("Stack Full Frame" if full else "Stack Selected Area") + size


# Results and texts.
def check_analysis(result):
    """A heat-map result's problems before it replaces the current map (the image itself is decoded
    by the dialog)."""
    c = result.get("canvas")

    def whole(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 1 and v == int(v)

    if not isinstance(c, dict) or not whole(c.get("width")) or not whole(c.get("height")):
        raise ValueError("Heat-map report has invalid image dimensions.")
    heatmap = result.get("heatmap")
    if not isinstance(heatmap, str) or not heatmap.strip():
        raise ValueError("Heat-map report has no preview image path.")
    if not os.path.isfile(heatmap):
        raise ValueError(f"Heat-map image not found: {heatmap}")
    return heatmap


def map_info(result):
    text = (
        f"{result['retained']} / {result['input_frames']} lights retained · Black: none · Blue: fewer · Red: "
        f"{result['retained']} frames · Drag to choose output framing."
    )
    if result.get("alignment_failed") is not None:
        text += (
            f" Alignment failures: {result['alignment_failed']}; sharpness exclusions: "
            f"{result.get('sharpness_rejected')}."
        )
    return text


def result_images(result):
    files = result.get("image_outputs") or []
    if not files and result.get("stack_file"):
        files = [{"label": "Stacked image", "format": "FITS", "path": result["stack_file"], "role": "stack_fits"}]
    return files


def result_summary(result, files):
    target = result["identity"]["target"] + " · " if result.get("identity") else ""
    names = "; ".join(f"{f['label']} ({f['format']})" for f in files)
    return (
        f"Last completed stack: {target}{result['canvas']['width']} × {result['canvas']['height']} pixels · "
        f"{result['contributing']} contributing lights · {duration(result['wall_s'])}. Saved {len(files)} "
        f"image file(s): {names}."
    )


def cache_text(cache):
    if cache.get("available_lights") and cache.get("storage_mode") != "gpu":
        source = "RAM" if cache["storage_mode"] == "ram" else "files"
        return (f"{cache['available_lights']} lights ready from {source} · GPU crops uploaded as needed · "
                "release after 5 idle min")
    if cache.get("resident_lights"):
        return (
            f"{cache['resident_lights']} lights ready · {cache.get('resident_vram_gib', 0):.2f} GiB · "
            "release after 5 idle min"
        )
    return "Worker ready · no GPU lights held"


def group_summary(groups):
    """One line per used group after a stack with several light groups."""
    used = [g for g in groups if g.get("used")]
    if len(used) < 2:
        return []
    lines = []
    for g in used:
        count = g["contributing"] if g.get("contributing") is not None else g.get("retained")
        text = f"{g['label']}: {count} lights"
        if g.get("main_light_equivalents") is not None:
            text += f", worth {g['main_light_equivalents']:.1f} main-group lights"
        prior = g.get("brightness_vs_star_prior")
        if prior:
            text += (
                f"; brightness matched where lights overlap (R {prior[0]:.2f}, G {prior[1]:.2f}, "
                f"B {prior[2]:.2f} × its star estimate)"
            )
        lines.append(text + ".")
    return lines


def operation_title(operation):
    return {"analyze": "Build Heat Map", "scan": "Check Folders"}.get(operation, "Stack Selected Area")


def status_prefix(operation):
    return {"analyze": "Building heat map: ", "scan": "Checking folders: "}.get(operation, "Stacking selected area: ")


def starting_status(operation):
    return {"analyze": "Starting heat-map analysis…", "scan": "Checking light folders…"}.get(
        operation, "Starting selected-area stack with cached alignment…"
    )


def run_summary_line(result):
    options = result.get("output_options") or {}
    if options and not options.get("activity_log"):
        tail = ". Detailed activity file was disabled."
    elif (result.get("files") or {}).get("activity_log"):
        tail = ", with timestamped activity log."
    else:
        tail = "."
    return f"Run summary saved in {result.get('folder')}{tail}"


def fallback_failure(result, operation, context):
    """A failure report for problems the worker could not report itself (start errors, crashes);
    `context` adds the dialog's state (settings, folders, region, python, stages, recent messages)."""
    failure = {
        "summary": result.get("summary") or "GPUCoadd could not finish this operation.",
        "next_step": result.get("next_step") or "Copy or export the technical report when asking for help.",
        "error": result.get("error") or "The worker stopped without a final report.",
        "operation": result.get("operation") or operation,
        "source_file": result.get("source_file"),
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        **context,
    }
    return failure


def failure_text(failure):
    return (
        f"GPUCoadd failure report\n{failure.get('summary')}\n{failure.get('next_step')}\n\nTechnical details\n"
        + json.dumps(failure, indent=2, default=str)
        + "\n"
    )


def hardware_args(results_folder):
    args = ["-u", str(HARDWARE_CHECK), "--work", work_dir()]
    if results_folder:
        args += ["--results", results_folder]
    return args


def hardware_result(code, stdout, stderr):
    """The startup GPU check's answer, or a failure explaining why there is none."""
    try:
        result = json.loads(stdout.strip())
        if code != 0 or not isinstance(result.get("supported"), bool) or not isinstance(result.get("summary"), str):
            raise ValueError(f"GPU check returned an invalid result (exit {code}).")
        return result
    except (ValueError, AttributeError) as exc:
        return {
            "supported": False,
            "code": "probe_failed",
            "summary": "GPUCoadd could not check this Python/GPU setup.",
            "next_step": "Choose the CUDA-enabled Python environment under Engine settings, then retry.",
            "error": f"{exc}\n{stderr}",
        }


def hardware_failure(result, python):
    return {
        "operation": "hardware_check",
        "summary": result.get("summary"),
        "next_step": result.get("next_step") or "Check Engine settings and retry the GPU check.",
        "error": result.get("error") or result.get("code"),
        "probe": result,
        "python": python,
        "failed_utc": datetime.now(timezone.utc).isoformat(),
    }


# Choices kept while the host program runs: reopening the window in the same session restores the
# folders; a restart of the host starts empty. Group ticks are never kept: a group is never left out
# without the group table showing it.
def state_file():
    return work_dir() + "/window_state.json"


def save_state(identity, folders, custom_output, output, path=None):
    if not identity:
        return
    try:
        Path(work_dir()).mkdir(parents=True, exist_ok=True)
        Path(path or state_file()).write_text(
            json.dumps(
                {
                    "session": identity,
                    "target": folders.target,
                    "folders": folders.folders,
                    "offered": folders.offered,
                    "custom_output": custom_output,
                    "output": norm_path(output) if custom_output else "",
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # a convenience: never let it stop the dialog


def load_state(identity, path=None):
    """The saved choices of this session with folders that still exist, or None."""
    if not identity:
        return None
    try:
        state = json.loads(Path(path or state_file()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("session") != identity:
        return None  # written in an earlier session
    folders = [f for f in state.get("folders") or [] if is_dir(f)]
    offered = [f for f in state.get("offered") or [] if is_dir(f)]
    if not folders:
        return None
    custom = bool(state.get("custom_output")) and bool(state.get("output"))
    return {
        "folders": folders,
        "offered": offered,
        "target": state.get("target") or parent_path(folders[0]),
        "custom_output": custom,
        "output": state.get("output") if custom else "",
    }


# A host program that runs this window with its own Python passes that environment on: PYTHONPATH
# and VIRTUAL_ENV point at the host's packages and its Scripts folder leads PATH, so a child Python
# would import the host's packages (its torch instead of GPUCoadd's). The worker gets the
# environment without them, and without any variables the host names as its own.
PYTHON_VARIABLES = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONUTF8", "PYTHONUNBUFFERED")


def worker_environment(environ=None, host_variables=()):
    """The environment for GPUCoadd's Python (worker and GPU check)."""
    env = dict(os.environ if environ is None else environ)
    venv = env.get("VIRTUAL_ENV")
    drop_names = {name.upper() for name in PYTHON_VARIABLES + tuple(host_variables)}
    for key in list(env):
        if key.upper() in drop_names:
            del env[key]
    if venv:
        drop = {os.path.normcase(os.path.normpath(os.path.join(venv, d))) for d in ("Scripts", "bin")}
        for key in [k for k in env if k.upper() == "PATH"]:
            env[key] = os.pathsep.join(
                p for p in env[key].split(os.pathsep) if p and os.path.normcase(os.path.normpath(p)) not in drop
            )
    return env
