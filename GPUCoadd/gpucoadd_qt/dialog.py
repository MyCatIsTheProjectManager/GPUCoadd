"""The GPUCoadd window (PyQt6), opened from Siril's Scripts menu (GPUCoadd.py) or on its own
(start_gpucoadd.pyw). The rules behind it are in logic.py and the worker it starts is worker/bridge.py, run by GPUCoadd's own CUDA
Python environment. `host` connects it to the program that opened it: `log(text, error)` for that
program's console, `open(path)` to show a result, `session` (a text naming the program's current
run, so folders are restored within it) and `variables` (environment variables the worker must not
inherit). Without a host the window still works, which the tests use."""

import codecs
import json
import os
from pathlib import Path
import sys
import time

from PyQt6.QtCore import QPointF, QProcess, QProcessEnvironment, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QGuiApplication, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import logic

AMBER = "background-color: #fff3cd; color: #5f4700;"
CHOICE_ITEMS = {
    "debayer": ["RCD (recommended)", "Bilinear (faster)"],
    "interp": ["Bicubic (recommended)", "Bilinear (faster)"],
    "norm": ["Background spread (recommended)", "Star brightness (experimental)", "Overlap matching (experimental)"],
    "weight": ["Equal (recommended)", "Noise-based (experimental)"],
    "rejection": ["Sigma clipping (recommended)", "Winsorized (gentler)", "Noise-aware (experimental)"],
    "transform": ["Similarity (recommended)", "Homography (experimental)"],
}
CHOICE_TITLES = {
    "debayer": "Colour method",
    "interp": "Resampling",
    "norm": "Brightness",
    "weight": "Frame weights",
    "rejection": "Rejection",
    "transform": "Alignment",
}
CHOICE_TIPS = {
    "debayer": "Turns the camera's raw colour pattern into a colour image. RCD keeps more fine detail. Bilinear is "
    "faster but softer.",
    "interp": "Controls how images are moved into place. Bicubic keeps more detail. Bilinear is faster but can blur "
    "small stars.",
    "norm": "Balances brightness between lights. Background spread is the usual choice. Star brightness matches the "
    "stars instead. Overlap matching compares lights only where they show the same sky, so nebula is never mistaken "
    "for background; it handles mosaics and changing haze best in tests, but is still being validated on finished "
    "stacks. Prepared while building the heat map when chosen beforehand (or for two telescopes); otherwise the first "
    "stack with it prepares it once from all lights (about 30 s for 350 lights) and saves it to the heat map.",
    "weight": "Equal gives each light the same influence. Noise-based favours quieter red, green and blue measurements "
    "separately. Experimental: compare faint detail before choosing it.",
    "rejection": "How unusual pixels (satellites, planes, hot pixels) are left out of each stacked pixel. Sigma "
    "clipping is the usual choice. Winsorized is gentler. Noise-aware judges each light against its own "
    "noise, so lights that were brightened (for example by Overlap matching on a hazy night) are not left out more "
    "often.",
    "transform": "Similarity moves, rotates and scales the lights. Homography also changes their shape. Usually leave "
    "this alone. Changing it needs a new heat map.",
}
OUTPUT_CHOICES = [
    ("stack_preview", "Stack preview (PNG)",
     "Saves a small, stretched picture for a quick look. It does not change the full-quality FITS stack."),
    ("coverage_png", "Frame count heat map (PNG)",
     "Saves the coloured frame count map with a legend so you can view or share it later."),
    ("coverage_fits", "Frame count map (FITS)",
     "Saves the frame count at every pixel for further analysis. This file can be large; leave off if you only need "
     "the picture."),
    ("frames_csv", "Per-light report (CSV)",
     "Saves a spreadsheet-friendly list showing which lights aligned, were excluded, or fell outside your crop."),
    ("aligned_lights", "Aligned lights (FITS)",
     "Saves every light that went into the stack as its own FITS in a folder beside the stack, on the stack's pixels "
     "and brightness-matched. This takes a lot of disk: about 12 MB per megapixel of your framing for each light, so "
     "crop first."),
    # Debayered lights: experimental, not shown.
    # ("debayered_lights", "Debayered lights (FITS)",
    #  "Saves every light that went into the stack as its own colour FITS in a folder beside the stack, in its original "
    #  "position and brightness. Use these when another program does the aligning. This takes a lot of disk: about 100 "
    #  "MB for each S50 Pro light and 25 MB for each S50 light."),
    ("activity_log", "Detailed activity log",
     "Keeps the worker's timestamped progress messages on disk. The live log still works when this is off."),
]


def now_ms():
    return int(time.time() * 1000)


def worker_process_environment(host):
    env = QProcessEnvironment()
    for key, value in logic.worker_environment(host_variables=getattr(host, "variables", ())).items():
        env.insert(key, value)
    return env


class HeatMap(QWidget):
    """The heat map with the chosen crop; drag to draw a crop, drag inside it to move it."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        self.drag_from = None
        self.moving = None
        self.setMinimumSize(760, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setToolTip(
            "Drag to draw a crop. Drag inside an existing crop to move it. Drag outside it to draw a new crop, or "
            "check Full Frame to reset."
        )

    def rect_on_screen(self):
        pixmap = self.dialog.pixmap
        if pixmap is None:
            return logic.display_rect(self.width(), self.height(), 0, 0)
        return logic.display_rect(self.width(), self.height(), pixmap.width(), pixmap.height())

    def paintEvent(self, event):
        d = self.dialog
        g = QPainter(self)
        try:
            g.fillRect(0, 0, self.width(), self.height(), QColor(0x15, 0x19, 0x1E))
            if d.pixmap is None:
                g.setPen(QColor(0xB4, 0xBE, 0xC9))
                g.drawText(24, self.height() // 2,
                           "Choose lights and click Build Heat Map to see how many frames will cover each pixel.")
                return
            r = self.rect_on_screen()
            g.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            g.drawPixmap(QRectF(*r), d.pixmap, QRectF(d.pixmap.rect()))
            if d.region:
                c, a = d.analysis["canvas"], d.region
                g.setPen(QPen(QColor(0x52, 0xE8, 0xFF), 2))
                g.drawRect(QRectF(r[0] + a[0] * r[2] / c["width"], r[1] + a[1] * r[3] / c["height"],
                                  a[2] * r[2] / c["width"], a[3] * r[3] / c["height"]))
        except Exception as exc:  # drawing must never take the dialog down
            if not d.paint_error:
                d.paint_error = True
                d.status.setText("Heat-map display error: see activity log.")
                d.append(f"Heat-map drawing failed: {exc}")
        finally:
            g.end()

    def mousePressEvent(self, event):
        d = self.dialog
        if d.busy or not d.analysis or event.button() != Qt.MouseButton.LeftButton:
            return
        r, p = self.rect_on_screen(), event.position()
        if not logic.inside(r, p.x(), p.y()):
            return
        self.drag_from = logic.sky_point(r, d.analysis["canvas"], p.x(), p.y())
        self.moving = logic.drag_start(d.region, d.analysis["canvas"], self.drag_from)

    def mouseMoveEvent(self, event):
        d = self.dialog
        if self.drag_from is None or d.busy:
            return
        p = event.position()
        point = logic.sky_point(self.rect_on_screen(), d.analysis["canvas"], p.x(), p.y())
        d.set_region(logic.drag_region(self.drag_from, self.moving, d.analysis["canvas"], point), True)

    def mouseReleaseEvent(self, event=None):
        self.drag_from = self.moving = None


class StageCircle(QWidget):
    """One stage: a circle that fills with the stage's progress."""

    def __init__(self, parent):
        super().__init__(parent)
        self.stage = None
        self.setFixedSize(92, 92)

    def paintEvent(self, event):
        if not self.stage:
            return
        import math

        g = QPainter(self)
        g.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = min(self.width(), self.height()) / 2 - 4
        cx, cy = self.width() / 2, self.height() / 2
        state = self.stage["state"]
        base = QColor(0xF2, 0xB8, 0xB8) if state == "failed" else QColor(0xF0, 0xE6, 0xB4) if state in (
            "running", "done") else QColor(0xD4, 0xD4, 0xD4)
        g.setPen(QPen(base, 1))
        g.setBrush(QBrush(base))
        g.drawEllipse(QPointF(cx, cy), r, r)
        f = logic.Stages.fill(self.stage)
        if f > 0:
            n = max(2, math.ceil(64 * f))
            points = [QPointF(cx, cy)] + [
                QPointF(cx + r * math.cos(-math.pi / 2 + 2 * math.pi * f * i / n),
                        cy + r * math.sin(-math.pi / 2 + 2 * math.pi * f * i / n))
                for i in range(n + 1)
            ]
            green = QColor(0x9B, 0xE2, 0x2D)
            g.setPen(QPen(green, 1))
            g.setBrush(QBrush(green))
            g.drawPolygon(QPolygonF(points))
        g.setPen(QPen(QColor(0x30, 0x30, 0x30), 2))
        for k in range(8):
            a = 2 * math.pi * k / 8 - math.pi / 2
            g.drawLine(QPointF(cx, cy), QPointF(cx + r * math.cos(a), cy + r * math.sin(a)))
        g.setPen(QPen(QColor(0x20, 0x20, 0x20), 3))
        g.setBrush(Qt.BrushStyle.NoBrush)
        g.drawEllipse(QPointF(cx, cy), r, r)
        g.end()


def bold(text):
    label = QLabel(f"<b>{text}</b>")
    label.setTextFormat(Qt.TextFormat.RichText)
    return label


def wrapped(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


def amber_panel():
    panel = QFrame()
    panel.setStyleSheet(AMBER)
    title = QLabel()
    title.setStyleSheet("font-weight: bold;")
    title.setTextFormat(Qt.TextFormat.PlainText)
    message = wrapped()
    box = QVBoxLayout(panel)
    box.setContentsMargins(8, 8, 8, 8)
    box.setSpacing(4)
    box.addWidget(title)
    box.addWidget(message)
    panel.setVisible(False)
    return panel, title, message


class PathRow:
    def __init__(self, title, text, choose, changed=None):
        self.layout = QHBoxLayout()
        self.layout.setSpacing(6)
        name = QLabel(title)
        name.setFixedWidth(105)
        self.edit = QLineEdit(text)
        self.browse = QPushButton("Browse…")
        self.changed = changed
        if changed:
            self.edit.textEdited.connect(lambda _: changed())

        def pick():
            value = choose(self.edit.text())
            if value:
                self.edit.setText(value)
                if changed:
                    changed()

        self.browse.clicked.connect(pick)
        self.layout.addWidget(name)
        self.layout.addWidget(self.edit, 100)
        self.layout.addWidget(self.browse)


class GPUCoaddDialog(QDialog):
    def __init__(self, host=None, session=None):
        super().__init__()
        self.host = host
        self.session_id = (getattr(host, "session", "") if session is None else session) or ""
        self.setWindowTitle("GPUCoadd: Preview, Frame, Stack")
        self.analysis = None
        self.stale_analysis = None  # analysis.json of a heat map outdated only by the sharpness limit
        self.pixmap = None
        self.region = None
        self.busy = False
        self.request_file = None
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.started_at = 0.0
        self.stage_eta = None
        self.eta_at = 0.0
        self.syncing = False
        self.operation = None
        self.logs = []
        self.folders = logic.Folders()
        self.groups = []
        self.filling_groups = self.filling_folders = False
        self.restoring_state = self.restoring_settings = False
        self.session_dir = None
        self.worker_persistent = False
        self.releasing = False
        self.pending_operation = None
        self.worker_idle_at = 0.0
        self.last_failure = None
        self.hardware_state = "checking"
        self.hardware_probe = None
        self.hardware_closed = False
        self.hardware_result = None
        self.paint_error = False
        self.rgb_auto_off = False
        self.result_files = []
        self.stages = logic.Stages()
        self.notices = logic.StorageNotices()
        self.work = logic.work_dir()
        self.worker = QProcess(self)
        self.worker.readyReadStandardOutput.connect(self.read_worker)
        self.worker.readyReadStandardError.connect(self.read_worker_errors)
        self.worker.finished.connect(self.worker_finished)
        self.worker.errorOccurred.connect(self.worker_error)
        self.build_ui()
        self.append("Ready. Heat-map analysis starts only when you click Build Heat Map.")
        self.restore_session_state()
        self.update_output_plan()
        self.stages.reset("analyze", False, time.time())  # what Build Heat Map will do, shown before it runs
        self.render_stages()
        self.set_busy(False)
        self.fit_layout()
        screen = self.screen_size()
        self.resize(min(1640, screen[0] - 40), min(860, screen[1] - 40))
        self.check_hardware()

    # ---- building the window -------------------------------------------------------------------
    def build_ui(self):
        title = QLabel("<b>GPUCoadd</b>  ·  See how many frames cover each pixel before you stack")
        title.setTextFormat(Qt.TextFormat.RichText)
        instructions = wrapped(
            'Choose the target folder (the one holding "lights"), build the heat map, then drag a rectangle to '
            "frame the result."
        )
        self.hardware_status = wrapped("Checking NVIDIA/CUDA support…")
        self.hardware_retry = QPushButton("Check again")
        self.hardware_retry.clicked.connect(self.check_hardware)
        self.hardware_retry.setToolTip(
            "Rechecks the Python environment, the NVIDIA GPU, free GPU memory, RAM and disk space (for example after "
            "closing other programs). Reads no lights and starts no analysis."
        )
        self.resource_panel, self.resource_title, self.resource_message = amber_panel()
        self.resource_title.setText("Action recommended: free up resources")
        self.resource_panel.setToolTip(
            "GPUCoadd still runs: with too little memory it uses slower RAM or disk storage with the same image "
            "quality, or stops with a clear message. Click Check again after closing programs."
        )
        hardware_row = QHBoxLayout()
        hardware_row.addWidget(self.hardware_status, 100)
        hardware_row.addWidget(self.hardware_retry)

        self.target_row = PathRow("Target folder", "", self.choose_directory)
        self.target_row.edit.setReadOnly(True)
        self.target_row.edit.setToolTip(
            'The folder of one target, holding its "lights" folder. Results are saved here.'
        )
        self.target_row.browse.clicked.disconnect()
        self.target_row.browse.clicked.connect(self.choose_target)
        self.target_row.browse.setToolTip(
            'Choose the target folder (the one holding "lights"). Choosing the lights folder itself also works.'
        )
        self.folder_list = QTreeWidget()
        self.folder_list.setColumnCount(1)
        self.folder_list.setHeaderHidden(True)
        self.folder_list.setRootIsDecorated(False)
        self.folder_list.setFixedHeight(56)  # ~2 folders (a target and one offered); scrolls beyond
        self.folder_list.setToolTip(
            'Light folders to stack together: ticked folders are used. Other "lights" folders found in the target '
            "(for example a second telescope) are listed unticked; tick one to add it. Same filter names can hide "
            "different filters, so check before ticking."
        )
        self.folder_list.itemChanged.connect(self.folder_ticked)
        self.add_folder = QPushButton("Add Folder…")
        self.add_folder.clicked.connect(self.add_folder_clicked)
        self.add_folder.setToolTip(
            "Add another folder of lights, for example another night or telescope of this target. Lights from "
            "different telescopes, filters and exposures are sorted into groups automatically."
        )
        self.remove_folder = QPushButton("Remove")
        self.remove_folder.clicked.connect(self.remove_folder_clicked)
        self.remove_folder.setToolTip("Remove the selected folder from the list (the files are not touched).")
        self.check_folders = QPushButton("Check Folders")
        self.check_folders.clicked.connect(lambda: self.start_job("scan"))
        self.check_folders.setToolTip(
            "Reads only the file headers (a few seconds) and lists the light groups: telescope, filter, exposure and "
            "how many lights. Nothing is stacked."
        )
        folder_buttons = QVBoxLayout()
        folder_buttons.setSpacing(4)
        folder_buttons.addWidget(self.add_folder)
        folder_buttons.addWidget(self.remove_folder)
        folder_buttons.addStretch()
        folder_title = QLabel("Light folders")
        folder_title.setFixedWidth(105)
        folder_row = QHBoxLayout()
        folder_row.addWidget(folder_title, 0, Qt.AlignmentFlag.AlignTop)
        folder_row.addWidget(self.folder_list, 100)
        folder_row.addLayout(folder_buttons)

        self.group_table = QTreeWidget()
        self.group_table.setColumnCount(4)
        self.group_table.setHeaderLabels(["Light group", "Lights", "Exposure", "Note"])
        self.group_table.setRootIsDecorated(False)
        self.group_table.setFixedHeight(88)  # ~3 groups; scrolls beyond
        self.group_table.setVisible(False)
        self.group_table.setToolTip(
            "Light groups found in the folders. Untick a group to leave it out. Groups with another filter than the "
            "main group are never stacked with it (LP and IR-cut lights are different data); to stack them, untick "
            "the main group's filter instead."
        )
        self.group_table.itemChanged.connect(self.group_ticked)
        self.group_hint = wrapped(
            "Check Folders lists the telescopes, filters and exposures found, before anything is stacked."
        )

        self.output = PathRow("Results folder", "", self.choose_directory, self.save_session_state)
        self.output.edit.setToolTip(
            "Where the stack and the extras you choose are saved: the target folder unless you choose another."
        )
        self.custom_output = QCheckBox("Another folder")
        self.custom_output.setToolTip(
            "Off: results go to the target folder, next to its lights folder. On: choose any folder."
        )
        self.custom_output.toggled.connect(self.custom_output_toggled)
        self.output.layout.addWidget(self.custom_output)

        self.reference_check = QCheckBox("Choose an alignment reference (optional)")
        self.reference = PathRow("Reference light", "", lambda p: self.choose_file(p, False), self.invalidate)
        self.reference_check.toggled.connect(self.reference_toggled)
        reference_hint = wrapped("Automatic reference unless selected. Choose output framing in step 2.")

        self.build = QPushButton("Build Heat Map")
        self.build.clicked.connect(lambda: self.start_job("analyze"))
        build_row = QHBoxLayout()
        build_row.addWidget(self.build)
        build_row.addWidget(QLabel("Aligns the lights and counts the frames on every pixel."))
        build_row.addStretch()
        self.keep_ready = QCheckBox("Keep lights ready")
        self.keep_ready.setChecked(True)
        self.keep_ready.setToolTip(
            "Keep prepared lights ready between the heat map and stack. Small sets stay on the GPU; larger sets use "
            "RAM or their original files. Released after five idle minutes, on Close, or with Release GPU. The heat "
            "map stays available."
        )
        self.keep_ready.toggled.connect(lambda checked: None if checked else self.release_gpu())
        self.release_button = QPushButton("Release GPU")
        self.release_button.clicked.connect(self.release_gpu)
        self.release_button.setToolTip(
            "Free the worker and any held GPU or RAM storage now. Your heat map, framing and saved files stay "
            "available; the next operation prepares the lights again."
        )
        self.cache_state = QLabel("GPU lights released")
        cache_row = QHBoxLayout()
        cache_row.addWidget(self.keep_ready)
        cache_row.addWidget(self.release_button)
        cache_row.addWidget(self.cache_state)
        cache_row.addStretch()

        self.map = HeatMap(self)
        self.map_info = wrapped("Expected frames per pixel: black = none · blue = few · red = most")
        frame_row = QHBoxLayout()
        frame_row.setSpacing(7)
        self.region_boxes = []
        for k, name in enumerate(["X", "Y", "Width", "Height"]):
            frame_row.addWidget(QLabel(name))
            spin = QSpinBox()
            spin.setRange(0 if k < 2 else 1, 1000000)
            spin.setValue(0 if k < 2 else 1)
            spin.setFixedWidth(85)
            spin.valueChanged.connect(self.region_typed)
            self.region_boxes.append(spin)
            frame_row.addWidget(spin)
        frame_row.addStretch()
        self.full_frame = QCheckBox("Full Frame")
        self.full_frame.setChecked(True)
        self.full_frame.setToolTip(
            "Use the entire heat-map canvas. Drawing a crop or editing its numbers turns this off. Check it again to "
            "restore the full frame; this does not change the alignment reference."
        )
        self.full_frame.clicked.connect(self.full_frame_clicked)

        self.status = wrapped("Ready: choose the target folder and click Build Heat Map.")
        self.output_warning = wrapped("")
        self.output_warning.setTextFormat(Qt.TextFormat.PlainText)
        self.output_warning.setStyleSheet(AMBER + " padding: 8px;")
        self.output_warning.hide()
        self.timing = wrapped("Started: -    Elapsed: -    Stage remaining: -")
        self.progress_text = wrapped("No operation running.")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)

        self.stage_panel = QWidget()
        self.stage_panel.setMinimumWidth(540)
        grid = QGridLayout(self.stage_panel)
        grid.setSpacing(8)
        self.stage_cells = []
        for k in range(12):
            cell = QWidget()
            box = QVBoxLayout(cell)
            box.setSpacing(3)
            box.setContentsMargins(0, 0, 0, 0)
            caption = QLabel()
            caption.setTextFormat(Qt.TextFormat.RichText)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            timing = QLabel()
            timing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            circle = StageCircle(cell)
            box.addWidget(caption)
            box.addWidget(timing)
            box.addWidget(circle, 0, Qt.AlignmentFlag.AlignHCenter)
            cell.setVisible(False)
            self.stage_cells.append((cell, caption, timing, circle))
            grid.addWidget(cell, k // 4, k % 4)

        self.stack_button = QPushButton("Stack Full Frame")
        self.stack_button.clicked.connect(lambda: self.start_job("stack"))
        self.open_result = QPushButton("Open Selected Image")
        self.open_result.clicked.connect(self.open_result_clicked)
        self.open_result.setToolTip(
            "Opens the saved image selected in the results list. A frame count map gives the number of "
            "lights stacked at each pixel; it is not an image of the sky."
        )
        self.result_choice = QComboBox()
        self.result_choice.setToolTip(
            "Choose which saved image to inspect. The stacked image and frame count map are separate outputs from "
            "one stacking operation."
        )
        self.result_choice.currentIndexChanged.connect(self.result_selected)
        self.result_summary = wrapped()
        self.result_path = wrapped()
        self.results_panel = QWidget()
        results_box = QVBoxLayout(self.results_panel)
        results_box.setContentsMargins(0, 0, 0, 0)
        results_box.setSpacing(4)
        result_row = QHBoxLayout()
        result_row.addWidget(QLabel("Saved images"))
        result_row.addWidget(self.result_choice, 100)
        result_row.addWidget(self.open_result)
        results_box.addWidget(self.result_summary)
        results_box.addLayout(result_row)
        results_box.addWidget(self.result_path)
        self.results_panel.setVisible(False)

        self.save_log = QPushButton("Save Activity Log…")
        self.save_log.clicked.connect(self.save_log_clicked)
        self.failure_summary = wrapped()
        self.error_details = QPushButton("Error Details…")
        self.error_details.setToolTip("Read the full technical report, copy it, or export it for someone helping you.")
        self.error_details.clicked.connect(self.open_error_details)
        self.failure_panel = QWidget()
        failure_box = QVBoxLayout(self.failure_panel)
        failure_box.setContentsMargins(0, 0, 0, 0)
        failure_box.addWidget(self.failure_summary)
        failure_box.addWidget(self.error_details, 0, Qt.AlignmentFlag.AlignLeft)
        self.failure_panel.setVisible(False)
        self.storage_panel, self.storage_title, self.storage_message = amber_panel()
        self.storage_panel.setToolTip(
            "This storage strategy preserves precision, samples and processing settings. Adequate disk space and "
            "working memory are still required; resource failures are reported separately."
        )
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_clicked)
        self.cancel = QPushButton("Cancel Operation")
        self.cancel.clicked.connect(self.cancel_job)

        # Image Quality
        self.quality_check = QCheckBox("Image Quality")
        self.quality_check.setToolTip("Optional processing choices. The recommended defaults are already selected.")
        self.quality_panel = QWidget()
        self.choices = {}
        rows = {}
        for key, items in CHOICE_ITEMS.items():
            row = QHBoxLayout()
            name = QLabel(CHOICE_TITLES[key])
            name.setFixedWidth(100)
            name.setToolTip(CHOICE_TIPS[key])
            combo = QComboBox()
            combo.addItems(items)
            combo.setToolTip(CHOICE_TIPS[key])
            row.addWidget(name)
            row.addWidget(combo, 100)
            self.choices[key] = combo
            rows[key] = row
        self.choices["norm"].currentIndexChanged.connect(lambda _: self.update_rgb_equal())
        self.choices["transform"].currentIndexChanged.connect(
            lambda _: None if self.restoring_settings else self.invalidate()
        )
        self.rgb_equal = QCheckBox("Equalize RGB backgrounds")
        self.rgb_equal.setChecked(True)
        self.rgb_equal.setToolTip(
            "On by default. With Background spread, matches all three channels to the reference light's green "
            "background level and spread. Changes colour balance; "
            "this is not colour calibration. Not available with Star brightness, Overlap matching or mixed light "
            "groups. Stacking only: the heat map stays valid."
        )
        self.match_gradients = QCheckBox("Match gradients (Overlap matching)")
        self.match_gradients.setToolTip(
            "Experimental, off by default. With Overlap matching, adjusts smooth background differences using "
            "overlapping sky. Noisy or small overlaps can change faint structure: compare with this option off. It "
            "does not replace calibration flats. Changes stacking only; the heat map stays valid."
        )
        feather_tip = (
            "Blends each light in gradually over this many pixels from its edge, so edges that do not quite match "
            "(vignetting, gradients, mosaic panels) leave no visible step. 0 turns it off. Try 20-50 px for small "
            "steps; stacks with strong edge differences may need 200 px or more. Use the smallest value that hides the "
            "steps: lights count less near their edges, so wide feathering adds noise where few lights overlap. "
            "Stacking only: the heat map stays valid."
        )
        self.feather = QSpinBox()
        self.feather.setRange(0, 1000)
        self.feather.setToolTip(feather_tip)
        feather_label = QLabel("Feather edges (px)")
        feather_label.setToolTip(feather_tip)
        self.frame_selection = QCheckBox("Leave out unusually blurry lights")
        self.frame_selection.setChecked(False)
        self.frame_selection.setToolTip(
            "Off by default: every light that aligns is used. Turn on to leave out lights that are much blurrier "
            "than the rest. Changing this needs a new heat map."
        )
        self.frame_selection.toggled.connect(self.frame_selection_toggled)
        sharpness_tip = (
            "Lower numbers leave out more blurry lights. Higher numbers keep more. 2.5 is a good starting point. "
            "Changing it needs a new heat map."
        )
        self.sharpness = QDoubleSpinBox()
        self.sharpness.setRange(0.5, 10)
        self.sharpness.setDecimals(1)
        self.sharpness.setSingleStep(0.1)
        self.sharpness.setValue(logic.DEFAULT_SHARPNESS)
        self.sharpness.setEnabled(False)
        self.sharpness.setToolTip(sharpness_tip)
        self.sharpness.valueChanged.connect(
            lambda _: self.mark_stale() if self.frame_selection.isChecked() and not self.restoring_settings else None
        )
        sharpness_label = QLabel("Sharpness limit")
        sharpness_label.setToolTip(sharpness_tip)
        self.defaults = QPushButton("Restore Recommended Defaults")
        self.defaults.setToolTip(
            "Returns all image-quality choices to the tested defaults. It does not start analysis or stacking."
        )
        self.defaults.clicked.connect(self.restore_defaults)
        q = QVBoxLayout(self.quality_panel)
        q.setContentsMargins(0, 0, 0, 0)
        q.setSpacing(6)

        def pair(*items):
            row = QHBoxLayout()
            row.setSpacing(18)
            for item, stretch in items:
                if isinstance(item, QHBoxLayout):
                    row.addLayout(item, stretch)
                else:
                    row.addWidget(item, stretch)
            return row

        feather_row = QHBoxLayout()
        feather_row.addWidget(feather_label)
        feather_row.addWidget(self.feather)
        sharp_row = QHBoxLayout()
        sharp_row.addWidget(self.frame_selection)
        sharp_row.addWidget(sharpness_label)
        sharp_row.addWidget(self.sharpness)
        q.addLayout(pair((rows["debayer"], 50), (rows["interp"], 50)))
        q.addLayout(pair((rows["norm"], 50), (rows["weight"], 50)))
        q.addLayout(pair((rows["rejection"], 50), (feather_row, 50)))
        q.addLayout(pair((rows["transform"], 50), (sharp_row, 50)))
        q.addLayout(pair((self.defaults, 0), (self.rgb_equal, 0), (self.match_gradients, 0), (QWidget(), 100)))
        self.quality_panel.setVisible(False)

        # Engine settings
        self.advanced_check = QCheckBox("Engine settings")
        self.advanced_check.setToolTip(
            "Computer settings for speed and memory use. These do not change your chosen framing."
        )
        self.advanced = QWidget()
        self.python = PathRow("Python", str(logic.DEFAULT_PYTHON), lambda p: self.choose_file(p, True),
                              self.python_changed)
        self.python.edit.setToolTip(
            "The Python program used by GPUCoadd. It needs the project's CUDA-enabled environment. Usually leave this "
            "unchanged."
        )
        self.python.browse.setToolTip(self.python.edit.toolTip())
        self.workers = QSpinBox()
        self.workers.setRange(1, 32)
        self.workers.setValue(8)
        self.workers.setToolTip(
            "How many CPU helpers align the lights. More may be faster, but can make the computer less responsive. "
            "Eight is the current default."
        )
        self.tile_size = QSpinBox()
        self.tile_size.setRange(64, 1024)
        self.tile_size.setValue(512)
        self.tile_size.setToolTip(
            "Manual tile size in pixels, used when Automatic tiles is off. Larger pieces need more GPU memory and are "
            "not always faster. This does not crop your image."
        )
        self.auto_tile = QCheckBox("Automatic tiles")
        self.auto_tile.setChecked(True)
        self.auto_tile.setToolTip(
            "Chooses a tested tile size that leaves room in GPU memory. Turn off to choose the size yourself. Your "
            "selected framing stays the same."
        )
        self.auto_tile.toggled.connect(lambda checked: self.tile_size.setEnabled(not checked and not self.busy))
        settings_row = QHBoxLayout()
        for item in (QLabel("Registration workers"), self.workers, self.auto_tile, QLabel("Tile pixels"),
                     self.tile_size, QLabel("NVIDIA CUDA GPU required")):
            settings_row.addWidget(item)
        settings_row.addStretch()
        a = QVBoxLayout(self.advanced)
        a.setContentsMargins(0, 0, 0, 0)
        a.addLayout(self.python.layout)
        a.addLayout(settings_row)
        self.advanced.setVisible(False)

        # Output files
        self.output_check = QCheckBox("Output Files")
        self.output_check.setToolTip(
            "Choose the extra files to save. Your stacked FITS and small run summary are always saved."
        )
        self.output_panel = QWidget()
        self.output_controls = {}
        o1, o2 = QHBoxLayout(), QHBoxLayout()
        for k, (key, text, tip) in enumerate(OUTPUT_CHOICES):
            box = QCheckBox(text)
            box.setToolTip(tip)
            box.toggled.connect(lambda _: self.update_output_plan())
            self.output_controls[key] = box
            (o1 if k < 3 else o2).addWidget(box)
        o1.addStretch()
        o2.addStretch()
        self.output_plan = wrapped()
        o = QVBoxLayout(self.output_panel)
        o.setContentsMargins(0, 0, 0, 0)
        o.addLayout(o1)
        o.addLayout(o2)
        o.addWidget(self.output_plan)
        o.addWidget(wrapped("Stacked FITS and run summary are always saved. Heat-map analysis also keeps its map and "
                            "alignment cache."))
        self.output_panel.setVisible(False)
        # one options section is open at a time, so the dialog stays within the screen
        self.sections = [(self.quality_check, self.quality_panel), (self.advanced_check, self.advanced),
                         (self.output_check, self.output_panel)]
        for check, _ in self.sections:
            check.toggled.connect(lambda checked, c=check: self.section_toggled(c, checked))

        # layout
        scan_row = QHBoxLayout()
        scan_row.addWidget(self.check_folders)
        scan_hint = wrapped("Optional: identify groups and automatic settings.")
        scan_hint.setToolTip(
            "Build Heat Map also checks folders. Mixed-telescope settings are chosen automatically from the light "
            "groups; single-telescope choices are kept."
        )
        scan_row.addWidget(scan_hint, 100)
        framing_row = QHBoxLayout()
        framing_row.setSpacing(12)
        framing_row.addWidget(self.full_frame)
        framing_row.addWidget(wrapped("Drag to crop; drag inside the crop to move it."), 100)
        activity_actions = QHBoxLayout()
        activity_actions.addWidget(self.save_log)
        activity_actions.addStretch()
        activity_actions.addWidget(self.cancel)
        activity_actions.addWidget(self.close_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.stack_button)
        buttons.addStretch()

        self.left_panel = QWidget()
        left = QVBoxLayout(self.left_panel)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(3)
        for item in (title, instructions, hardware_row, self.target_row.layout, folder_row, bold("0 · Check folders"),
                     scan_row, self.group_table, self.group_hint, bold("1 · Build heat map"), self.reference_check,
                     self.reference.layout, reference_hint, build_row, cache_row, bold("2 · Choose framing"),
                     framing_row):
            left.addLayout(item) if isinstance(item, (QHBoxLayout, QVBoxLayout)) else left.addWidget(item)
        left.addWidget(self.map, 100)
        left.addWidget(self.map_info)
        left.addLayout(frame_row)
        for item in (bold("3 · Stack"), self.output.layout, buttons, self.results_panel):
            left.addLayout(item) if isinstance(item, (QHBoxLayout, QVBoxLayout)) else left.addWidget(item)
        right = self.right_layout = QVBoxLayout()
        right.setSpacing(7)
        # Option sections open on the right and take room from the log, so the left column keeps its
        # height, so the window fits a 1080p screen with a section open.
        for item in (self.stage_panel, self.status, self.output_warning, self.timing, self.progress_text, self.failure_panel,
                     self.resource_panel, self.storage_panel, self.quality_check, self.quality_panel,
                     self.advanced_check, self.advanced, self.output_check, self.output_panel):
            right.addWidget(item)
        right.addWidget(self.log, 100)
        right.addLayout(activity_actions)
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)
        # The left column scrolls only when the screen is too short for it (large fonts, a small
        # screen): the window itself never grows past the screen.
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidget(self.left_panel)
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self.left_scroll, 58)
        root.addLayout(right, 42)

        self.timer = QTimer(self)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.tick)
        self.hardware_delay = QTimer(self)
        self.hardware_delay.setSingleShot(True)
        self.hardware_delay.setInterval(750)
        self.hardware_delay.timeout.connect(self.check_hardware)
        self.hardware_timeout = QTimer(self)
        self.hardware_timeout.setSingleShot(True)
        self.hardware_timeout.setInterval(60000)  # only a limit: a slow first start of the Python environment can take a while
        self.hardware_timeout.timeout.connect(self.hardware_timed_out)

    # ---- small helpers -------------------------------------------------------------------------
    def screen_size(self):
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            r = screen.availableGeometry()
            if r.width() > 300 and r.height() > 300:
                return r.width(), r.height()
        return 1920, 1000  # a 1080p screen (taskbar and title bar off) when Qt cannot tell us

    def fit_layout(self):
        """The dialog must fit the screen (a 1080p laptop has ~1000 px): option sections open in the
        right column and take room from the log, the heat map gives way when the results panel opens,
        the window grows to show everything up to the screen's height, and beyond that the left column
        scrolls."""
        self.map.setMinimumHeight(140 if self.results_panel.isVisible() else 200)
        self.grow_to_fit()
        QTimer.singleShot(0, self.grow_to_fit)  # once more after Qt has laid out what just changed

    def grow_to_fit(self):
        layout = self.left_panel.layout()
        layout.invalidate()  # sizes cached before a panel was shown or hidden are stale
        layout.activate()
        margins = self.layout().contentsMargins()
        # wrapped labels need the height for the column's actual width, not Qt's default guess
        width = self.left_scroll.viewport().width()
        content = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else -1
        need = max(content, self.left_panel.minimumSizeHint().height(), self.right_layout.minimumSize().height())
        need += margins.top() + margins.bottom()
        target = min(self.screen_size()[1] - 40, max(self.height(), need))
        if target != self.height():
            self.resize(self.width(), target)

    def scrolls(self):
        """Whether the left column is taller than the space it has."""
        return self.left_scroll.verticalScrollBar().maximum() > 0

    def section_toggled(self, check, checked):
        for other, panel in self.sections:
            if other is check:
                panel.setVisible(checked)
            elif checked:
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
                panel.setVisible(False)
        self.fit_layout()

    def append(self, text):
        line = f"[{logic.stamp()}] {text}"
        self.logs.append(line)
        self.log.appendPlainText(line)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def tell_host(self, text, error=False):
        if self.host is not None:
            try:
                self.host.log(text, error)
            except Exception:  # the host's console is a convenience, never a reason to fail
                pass

    def choose_directory(self, initial):
        return QFileDialog.getExistingDirectory(self, "Select folder", initial or "") or None

    def choose_file(self, initial, python):
        caption = "Select the CUDA Python environment" if python else "Select an optional alignment reference light"
        pattern = "Python executable (*.exe)" if python else "FITS light (*.fit *.fits *.fts)"
        return QFileDialog.getOpenFileName(self, caption, initial or "", pattern)[0] or None

    def results_folder(self):
        return logic.norm_path(self.output.edit.text()) if self.custom_output.isChecked() else self.folders.target

    # ---- folders and groups --------------------------------------------------------------------
    def choose_target(self):
        value = self.choose_directory(self.folders.target)
        if value:
            self.set_target(value)

    def set_target(self, chosen):
        found = logic.find_light_folders(chosen)
        if "error" in found:
            self.group_hint.setText(found["error"])
            self.status.setText(found["error"])
            return False
        self.set_folders([found["lights"]], [], found["offered"], found["target"])
        if not self.busy:
            self.status.setText("Target folder chosen: check folders or build the heat map.")
        return True

    def set_folders(self, folders, excluded=None, offered=None, target=None):
        hint = self.folders.set(folders, excluded, offered, target)
        self.target_row.edit.setText(self.folders.target)
        if not self.custom_output.isChecked():
            self.output.edit.setText(self.folders.target)
        self.filling_folders = True
        self.folder_list.clear()
        for folder, ticked, text in self.folders.rows():
            item = QTreeWidgetItem([text])
            item.setData(0, Qt.ItemDataRole.UserRole, folder)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if ticked else Qt.CheckState.Unchecked)
            self.folder_list.addTopLevelItem(item)
        self.filling_folders = False
        self.show_groups([])
        self.group_hint.setText(hint)
        self.invalidate()
        self.save_session_state()

    def folder_ticked(self, item, column):
        if self.filling_folders:
            return
        folder = item.data(0, Qt.ItemDataRole.UserRole)
        folders = list(self.folders.folders)
        offered = list(self.folders.offered)
        checked = item.checkState(0) == Qt.CheckState.Checked
        if checked and folder not in folders:
            folders.append(folder)
            if folder in offered:
                offered.remove(folder)
        elif not checked and folder in folders:
            folders.remove(folder)
            offered.append(folder)
        # deferred: the list is rebuilt, which must not happen inside its own signal
        QTimer.singleShot(0, lambda: self.set_folders(folders, None, offered))

    def add_folder_clicked(self):
        start = self.folders.folders[-1] if self.folders.folders else self.folders.target
        value = self.choose_directory(start)
        if not value:
            return
        problem = logic.add_folder_problem(value)
        if problem:
            self.group_hint.setText(problem)
            return
        if not self.folders.target:
            self.set_target(value)
        else:
            self.set_folders(self.folders.folders + [value])

    def remove_folder_clicked(self):
        item = self.folder_list.currentItem()
        if item is None:
            return
        folder = item.data(0, Qt.ItemDataRole.UserRole)
        self.set_folders([f for f in self.folders.folders if f != folder], None,
                         [f for f in self.folders.offered if f != folder])

    def show_groups(self, groups):
        self.filling_groups = True
        self.groups = groups or []
        self.group_table.clear()
        rows, hint = logic.group_rows(self.groups)
        for row in rows:
            item = QTreeWidgetItem(row["texts"])
            item.setData(0, Qt.ItemDataRole.UserRole, row["key"])
            flags = item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable
            if row["checkable"]:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
            item.setFlags(flags)
            item.setCheckState(0, Qt.CheckState.Checked if row["checked"] else Qt.CheckState.Unchecked)
            self.group_table.addTopLevelItem(item)
        for c in range(4):
            self.group_table.resizeColumnToContents(c)  # full group names
        self.group_table.setVisible(bool(self.groups))
        if hint:
            self.group_hint.setText(hint)
        self.filling_groups = False
        self.update_rgb_equal()

    def group_ticked(self, item, column):
        if self.filling_groups or not (item.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        self.folders.exclude(key, item.checkState(0) == Qt.CheckState.Checked)
        self.group_hint.setText("Group choice changed: Check Folders updates the table; Build Heat Map uses the new "
                                "choice.")
        self.invalidate()

    def custom_output_toggled(self, checked):
        if not checked:
            self.output.edit.setText(self.folders.target)
        self.output.edit.setEnabled(checked and not self.busy)
        self.output.browse.setEnabled(checked and not self.busy)
        self.save_session_state()

    def reference_toggled(self, checked):
        self.reference.edit.setEnabled(checked)
        self.reference.browse.setEnabled(checked)
        self.invalidate()

    # ---- choices kept within one session of the host program ----------------------------------
    def save_session_state(self):
        if self.restoring_state:
            return
        logic.save_state(self.session_id, self.folders, self.custom_output.isChecked(), self.output.edit.text())

    def restore_session_state(self):
        state = logic.load_state(self.session_id)
        if not state:
            return False
        self.restoring_state = True
        try:
            self.custom_output.setChecked(state["custom_output"])
            if state["custom_output"]:
                self.output.edit.setText(state["output"])
            self.custom_output_toggled(state["custom_output"])
            self.set_folders(state["folders"], [], state["offered"], state["target"])
            self.append(f"Restored the target folder from earlier in this session: {self.folders.target}.")
            return True
        finally:
            self.restoring_state = False

    # ---- image quality and outputs -------------------------------------------------------------
    def quality_settings(self):
        return logic.quality_settings(
            {key: combo.currentIndex() for key, combo in self.choices.items()},
            self.frame_selection.isChecked(),
            round(self.sharpness.value(), 1),
            self.feather.value(),
            self.rgb_equal.isChecked(),
            self.match_gradients.isChecked(),
        )

    def apply_quality(self, settings):
        self.restoring_settings = True
        try:
            c = logic.quality_controls(settings)
            for key, combo in self.choices.items():
                combo.setCurrentIndex(c["choice"][key])
            self.frame_selection.setChecked(c["selection"])
            self.sharpness.setValue(c["sharpness"])
            self.sharpness.setEnabled(c["selection"] and not self.busy)
            self.feather.setValue(int(c["feather"]))
            self.rgb_equal.setChecked(c["rgb_equal"])
            self.rgb_auto_off = False
            self.match_gradients.setChecked(c["match_gradients"])
            self.update_rgb_equal()
        finally:
            self.restoring_settings = False

    def update_rgb_equal(self):
        state = logic.option_availability(self.choices["norm"].currentIndex(), self.groups, self.rgb_equal.isChecked(),
                                          self.rgb_auto_off, self.match_gradients.isChecked())
        self.rgb_equal.setChecked(state["rgb_equal"])
        self.rgb_auto_off = state["rgb_auto_off"]
        self.match_gradients.setChecked(state["match_gradients"])
        self.rgb_equal.setEnabled(state["rgb_available"] and not self.busy)
        self.match_gradients.setEnabled(state["gradients_available"] and not self.busy)
        if not self.restoring_settings:
            for message in state["messages"]:
                self.append(message)

    def frame_selection_toggled(self, checked):
        self.sharpness.setEnabled(checked and not self.busy)
        if not self.restoring_settings:
            self.mark_stale()

    def restore_defaults(self):
        previous = self.quality_settings()
        self.apply_quality({})
        if previous["transform"] != "similarity":
            self.invalidate()
        elif previous["select"] != 0:
            self.mark_stale()
        self.append("Recommended image-quality defaults restored. No operation started.")

    def output_settings(self):
        return {key: box.isChecked() for key, box in self.output_controls.items()}

    def update_output_plan(self):
        self.output_plan.setText(logic.output_plan_text(self.output_settings()))

    def python_changed(self):
        self.invalidate()
        self.schedule_hardware_check()

    # ---- heat map and framing ------------------------------------------------------------------
    def invalidate(self):
        self.stale_analysis = None
        self.release_gpu()
        self.clear_heat_map(logic.HEAT_MAP_OUTDATED)

    def mark_stale(self):
        """Only the sharpness limit changed: the heat map is out of date as with invalidate(), but
        its star detection and alignment still hold. The next Build Heat Map sends it as
        previous_analysis, and the worker keeps its loaded lights for that."""
        if self.analysis:
            self.stale_analysis = self.analysis.get("analysis_file")
        self.clear_heat_map(logic.HEAT_MAP_LIMIT_CHANGED)

    def clear_heat_map(self, texts):
        if not self.analysis:
            return
        self.analysis = self.pixmap = self.region = None
        self.full_frame.setChecked(True)
        self.map.mouseReleaseEvent()
        self.update_stack_caption()
        status, log = texts
        self.status.setText(status)
        self.append(log)
        self.set_busy(False)
        self.map.update()

    def update_stack_caption(self):
        self.stack_button.setText(logic.stack_caption(self.full_frame.isChecked(), self.region))

    def set_region(self, region, edited=False):
        self.region = list(region)
        self.syncing = True
        for box, value in zip(self.region_boxes, region):
            box.setValue(int(value))
        self.syncing = False
        self.full_frame.setChecked(logic.is_full_frame(region, self.analysis["canvas"], edited))
        self.update_stack_caption()
        self.map.update()

    def region_typed(self, _value):
        if self.syncing or self.busy or not self.analysis:
            return
        values = [box.value() for box in self.region_boxes]
        self.set_region(logic.clamp_region(values, self.analysis["canvas"]), True)

    def full_frame_clicked(self, checked):
        if self.busy or not self.analysis:
            self.full_frame.setChecked(True)
            return
        self.map.mouseReleaseEvent()
        if checked:
            self.set_region(logic.full_frame(self.analysis["canvas"]))
        else:
            self.full_frame.setChecked(False)
        self.update_stack_caption()

    def prepare_analysis(self, result):
        path = logic.check_analysis(result)
        pixmap = QPixmap(path)
        if pixmap.isNull() or pixmap.width() < 1 or pixmap.height() < 1:
            raise ValueError("Heat-map image is empty.")
        return pixmap

    def show_analysis(self, result, pixmap=None):
        pixmap = pixmap or self.prepare_analysis(result)
        self.analysis, self.pixmap, self.paint_error = result, pixmap, False
        self.stale_analysis = None
        self.set_region(logic.full_frame(result["canvas"]))
        self.map_info.setText(logic.map_info(result))
        self.map.update()

    # ---- stages, notices, results, failures ----------------------------------------------------
    def render_stages(self):
        now = time.time()
        stages = self.stages.stages
        rows = -(-len(stages) // 4)
        for i, (cell, caption, timing, circle) in enumerate(self.stage_cells):
            # whole rows: an empty cell in a used row keeps its place, so circles stay in columns
            cell.setVisible(i // 4 < rows)
            stage = stages[i] if i < len(stages) else None
            circle.stage = stage
            if stage is None:
                caption.setText("")
                timing.setText("")
            else:
                mark = (" <b><span style='color:#3c9a1c'>&#10004;</span></b>" if stage["state"] == "done" else
                        " <b><span style='color:#c0392b'>&#10008;</span></b>" if stage["state"] == "failed" else "")
                caption.setText(stage["name"] + mark)
                timing.setText(logic.Stages.time_text(stage, now))
            circle.update()

    def show_notice(self, text):
        self.storage_panel.setVisible(text is not None)
        if text:
            self.storage_title.setText(text[0])
            self.storage_message.setText(text[1])
        self.fit_layout()

    def show_results(self, result):
        self.result_files = logic.result_images(result)
        self.result_choice.blockSignals(True)
        self.result_choice.clear()
        for item in self.result_files:
            self.result_choice.addItem(f"{item['label']} ({item['format']})")
        self.result_choice.blockSignals(False)
        self.result_summary.setText(logic.result_summary(result, self.result_files))
        self.results_panel.setVisible(bool(self.result_files))
        if self.result_files:
            self.result_choice.setCurrentIndex(0)
            self.result_selected(0)
        self.open_result.setEnabled(bool(self.result_files) and not self.busy)
        self.append(self.result_summary.text())
        for item in self.result_files:
            self.append(f"Saved {item['label']}: {item['path']}")
        self.fit_layout()

    def result_selected(self, index):
        if 0 <= index < len(self.result_files):
            path = self.result_files[index]["path"]
            self.result_path.setText(logic.base_name(path))
            self.result_path.setToolTip(path)

    def open_result_clicked(self):
        index = self.result_choice.currentIndex()
        if not 0 <= index < len(self.result_files):
            return
        item = self.result_files[index]
        try:
            if not os.path.isfile(item["path"]):
                raise ValueError(f"Saved image not found: {item['path']}")
            if self.host is None:
                raise ValueError("No program is connected to show images: open the file yourself.")
            self.host.open(item["path"])
            self.append(f"Opened {item['label']}: {item['path']}")
        except Exception as exc:
            self.show_failure({"operation": "open_result", "source_file": item["path"],
                               "summary": "The saved image could not be opened.",
                               "next_step": "Check that the file is still available, or create the stack again.",
                               "error": str(exc)})

    def save_log_clicked(self):
        start = self.output.edit.text() + "/GPUCoadd-activity.txt"
        path = QFileDialog.getSaveFileName(self, "Save activity log", start, "Text (*.txt)")[0]
        if not path:
            return
        try:
            Path(path).write_text("\n".join(self.logs) + "\n", encoding="utf-8")
        except OSError as exc:
            self.show_failure({"operation": "save_activity_log", "source_file": path,
                               "summary": "The activity log could not be saved.",
                               "next_step": "Choose another writable location and try again.", "error": str(exc)})

    def show_failure(self, result):
        failure = result.get("failure") or logic.fallback_failure(result, self.operation, {
            "settings": self.quality_settings(), "folders": list(self.folders.folders), "region": self.region,
            "python": self.python.edit.text(), "results_folder": self.results_folder(),
            "stages": self.stages.as_dicts(), "recent_messages": self.logs[-80:]})
        failure["error_report"] = result.get("error_report") or failure.get("error_report")
        failure["error_report_save_errors"] = result.get("error_report_save_errors") or []
        self.last_failure = failure
        self.failure_summary.setText(f"Last failure: {failure['summary']} {failure['next_step']}")
        self.failure_panel.setVisible(True)
        self.status.setText(failure["summary"])
        self.progress_text.setText(f"Failed at {logic.stamp()}")
        self.append(f"ERROR: {failure['summary']} {failure['next_step']}")
        self.tell_host(f"GPUCoadd: {failure['summary']} {failure['next_step']}", error=True)
        if failure["error_report"]:
            self.append(f"Failure report saved: {failure['error_report']}")
        else:
            # also covers startup errors and worker crashes that cannot save a Python report
            try:
                Path(self.work, "errors").mkdir(parents=True, exist_ok=True)
                failure["error_report"] = f"{self.work}/errors/GPUCoadd-error-ui-{now_ms()}.txt"
                Path(failure["error_report"]).write_text(logic.failure_text(failure), encoding="utf-8")
                self.append(f"Failure report saved: {failure['error_report']}")
            except OSError as exc:
                failure["error_report"] = None
                self.append(f"Could not save failure report: {exc}. Use Error Details to copy/export it.")
        self.fit_layout()

    def open_error_details(self):
        if self.last_failure:
            ErrorReport(self, self.last_failure).exec()

    # ---- jobs ----------------------------------------------------------------------------------
    def set_busy(self, busy):
        if busy:
            self.map.mouseReleaseEvent()
            self.output_warning.clear()
            self.output_warning.hide()
        self.busy = busy
        ready = self.hardware_state == "ready"
        self.keep_ready.setEnabled(not busy and not self.releasing)
        self.release_button.setEnabled(not busy and self.session_dir is not None and not self.releasing)
        self.build.setEnabled(not busy and ready)
        self.hardware_retry.setEnabled(not busy and self.hardware_state != "checking")
        self.cancel.setEnabled(busy)
        self.close_button.setEnabled(not busy)
        for check, _ in self.sections:
            check.setEnabled(not busy)
        for box in self.output_controls.values():
            box.setEnabled(not busy)
        for widget in list(self.choices.values()) + [self.defaults, self.frame_selection, self.feather]:
            widget.setEnabled(not busy)
        self.sharpness.setEnabled(not busy and self.frame_selection.isChecked())
        self.update_rgb_equal()
        self.workers.setEnabled(not busy)
        self.auto_tile.setEnabled(not busy)
        self.tile_size.setEnabled(not busy and not self.auto_tile.isChecked())
        for widget in (self.folder_list, self.add_folder, self.remove_folder, self.group_table,
                       self.target_row.browse, self.custom_output, self.python.edit, self.python.browse,
                       self.reference_check):
            widget.setEnabled(not busy)
        self.check_folders.setEnabled(not busy and ready)
        self.output.edit.setEnabled(not busy and self.custom_output.isChecked())
        self.output.browse.setEnabled(not busy and self.custom_output.isChecked())
        self.reference.edit.setEnabled(not busy and self.reference_check.isChecked())
        self.reference.browse.setEnabled(not busy and self.reference_check.isChecked())
        self.stack_button.setEnabled(not busy and self.analysis is not None and ready)
        self.full_frame.setEnabled(not busy and self.analysis is not None)
        self.open_result.setEnabled(not busy and bool(self.result_files))
        self.result_choice.setEnabled(not busy and bool(self.result_files))
        for box in self.region_boxes:
            box.setEnabled(not busy and self.analysis is not None)

    def build_request(self, operation):
        return logic.build_request(
            operation, self.folders.folders, self.folders.excluded, self.results_folder(), self.work,
            self.reference.edit.text() if self.reference_check.isChecked() else "", self.quality_settings(),
            self.output_settings(), self.workers.value(),
            0 if self.auto_tile.isChecked() else self.tile_size.value(),
            self.analysis.get("analysis_file") if operation == "stack" and self.analysis else None,
            self.region if operation == "stack" else None,
            self.stale_analysis if operation == "analyze" else None,
        )

    def worker_running(self):
        return self.worker.state() != QProcess.ProcessState.NotRunning

    def start_job(self, operation):
        if self.busy:
            return
        if self.hardware_state != "ready":
            self.status.setText(self.hardware_status.text())
            return
        if self.session_dir and (self.releasing or (self.worker_idle_at and time.time() - self.worker_idle_at >
                                                     logic.IDLE_RELEASE_S)):
            self.pending_operation = operation
            self.release_gpu()
            return
        try:
            self.operation = operation
            problem = logic.job_problem(operation, self.folders.folders, self.results_folder(),
                                        self.custom_output.isChecked(), self.python.edit.text(),
                                        self.reference_check.isChecked(), self.reference.edit.text(), self.analysis,
                                        self.region)
            if problem:
                raise ValueError(problem)
            Path(self.work, "requests").mkdir(parents=True, exist_ok=True)
            # unique even for two jobs within one millisecond: results are matched to their request by name
            self.request_count = getattr(self, "request_count", 0) + 1
            self.request_file = f"{self.work}/requests/request_{now_ms()}_{self.request_count}.json"
            request = self.build_request(operation)
            Path(self.request_file).write_text(json.dumps(request, indent=2), encoding="utf-8")
            if operation != "scan":
                self.notices.reset()
                self.storage_panel.setVisible(False)
            self.started_at = time.time()
            self.stage_eta = None
            self.set_busy(True)
            self.stages.reset(operation, True, self.started_at)
            self.render_stages()
            self.status.setText(logic.starting_status(operation))
            self.progress_text.setText("Starting worker; time estimate will appear as work completes.")
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_at))
            self.append(f"{logic.operation_title(operation)} started at {started}.")
            self.tell_host(f"GPUCoadd: {logic.operation_title(operation)} started.")
            self.append("Chosen settings: " + json.dumps(request["settings"], separators=(",", ":")))
            self.append("Extra output files: " + json.dumps(request["outputs"], separators=(",", ":")))
            if operation == "stack":
                r = request["region"]
                self.append(f"Selected framing: X {r[0]}, Y {r[1]}, {r[2]} × {r[3]} pixels.")
            self.timer.start()
            self.worker.setWorkingDirectory(str(logic.PROJECT_DIR))
            self.worker.setProcessEnvironment(worker_process_environment(self.host))
            python = self.python.edit.text()
            if self.keep_ready.isChecked():
                if not self.worker_running():
                    self.session_dir = f"{self.work}/sessions/worker_{now_ms()}"
                    Path(self.session_dir).mkdir(parents=True, exist_ok=True)
                    self.worker_persistent = True
                    self.buffer = ""
                    self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
                    self.hand_over(self.request_file)
                    self.worker.start(python, ["-u", str(logic.BRIDGE), "--session", self.session_dir])
                else:
                    if not self.session_dir or self.releasing:
                        raise ValueError("Worker is releasing; wait a moment and try again.")
                    self.hand_over(self.request_file)
            else:
                self.buffer = ""
                self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
                self.worker_persistent = False
                self.worker.start(python, ["-u", str(logic.BRIDGE), "--request", self.request_file])
        except Exception as exc:
            self.timer.stop()
            self.set_busy(False)
            self.stages.finish("error", time.time())
            self.render_stages()
            self.show_failure({"summary": str(exc), "next_step": "Check the selected paths and settings, then try "
                               "again.", "error": str(exc)})

    def hand_over(self, request_file):
        """The session worker reads command.txt once `ready` exists: write both, in that order."""
        Path(self.session_dir, "command.txt").write_text(request_file, encoding="utf-8")
        Path(self.session_dir, "ready").write_text("ready\n", encoding="utf-8")

    def read_worker(self):
        # incremental: a character split between two reads is decoded whole
        chunk = self.decoder.decode(bytes(self.worker.readAllStandardOutput()))
        events, self.buffer = logic.split_events(self.buffer, chunk)
        for event in events:
            self.worker_event(event)

    def worker_event(self, event):
        kind = event.get("type")
        if kind == "progress":
            self.stage_eta, self.eta_at = event.get("stage_eta_s"), time.time()
            self.stages.event(event["stage"], event.get("done") or 0, event.get("total") or 0, time.time())
            self.render_stages()
            done, total = event.get("done") or 0, event.get("total") or 0
            self.progress_text.setText(f"{event['stage']}: {done} / {total}" +
                                       (f" ({round(100 * done / total)}%)" if total else ""))
        elif kind == "stage_started":
            self.stage_eta = None
            self.stages.event(event["stage"], 0, 0, time.time())
            self.render_stages()
            self.status.setText(logic.status_prefix(self.operation) + event["stage"])
            self.append(event.get("message", ""))
        elif kind == "stage_finished":
            self.append(f"{event['stage']} took {logic.duration(event.get('duration_s') or 0)}.")
        elif kind == "finished" and self.worker_persistent:
            # only the first answer for the job being waited on counts; a late or repeated one is logged, not acted on
            if self.busy and logic.same_request(event, self.request_file):
                self.finish_job(event, 0 if event.get("status") == "complete" else 1)
                self.remove_request_files()
            else:
                self.append(f"Ignored a late worker result ({event.get('status') or 'unknown'}) for a job that "
                            "already finished.")
        elif kind == "warning":
            previous = self.output_warning.text()
            self.output_warning.setText((previous + "\n\n" if previous else "") + event.get("message", ""))
            self.output_warning.show()
            self.append("WARNING: " + event.get("message", "") + (f" ({event['error']})" if event.get("error") else ""))
        elif kind == "session_closed":
            self.cache_state.setText(f"GPU lights released ({event.get('reason')})")
            self.append(f"GPU worker released: {event.get('reason')}. The heat map and framing stay available.")
        elif kind == "error":
            self.append("ERROR: " + event.get("message", ""))
        elif kind == "storage":
            self.show_notice(self.notices.add(event))
        elif kind in ("log", "started", "text"):
            self.append(event.get("message", ""))

    def read_worker_errors(self):
        text = bytes(self.worker.readAllStandardError()).decode("utf-8", errors="replace").strip()
        if text:
            self.append(text)

    def finish_job(self, result, exit_code):
        if result.get("operation") != "scan":
            self.show_notice(self.notices.restore(result))
        self.timer.stop()
        self.stage_eta = None
        ok = result.get("status") == "complete" and exit_code == 0
        self.stages.finish("complete" if ok else "cancelled" if result.get("status") == "cancelled" else "error",
                           time.time())
        self.render_stages()
        elapsed = time.time() - self.started_at
        self.timing.setText(f"Started: {time.strftime('%H:%M:%S', time.localtime(self.started_at))}    Took: "
                            f"{logic.duration(elapsed)}")
        try:
            if ok:
                operation = result.get("operation")
                if operation == "scan":
                    self.show_groups(result.get("groups") or [])
                    self.status.setText("Folders checked: " + self.group_hint.text())
                    self.append(f"Folders checked: {result.get('used_frames')} of {result.get('input_frames')} lights "
                                "in the selected groups.")
                elif operation == "analyze":
                    self.show_groups(result.get("groups") or [])
                    self.show_analysis(result)
                    failed = result.get("alignment_failed") or 0
                    self.status.setText(f"Heat map ready: {failed} lights failed alignment; review the frame report."
                                        if failed > 0 else "Heat map ready: review framing, then stack in step 3.")
                    chosen = "Chosen" if result.get("reference_override") else "Automatic"
                    self.append(f"Heat map ready in {logic.duration(elapsed)}. {chosen} alignment reference: "
                                f"{result.get('reference')}")
                    self.tell_host(f"GPUCoadd: heat map ready, {result.get('retained')} of "
                                    f"{result.get('input_frames')} lights retained.")
                else:
                    self.show_results(result)
                    self.status.setText("Stack complete: choose a saved image below, then Open Selected Image.")
                    self.append(f"Stack completed in {logic.duration(elapsed)}. Saved: {result.get('stack_file')}")
                    self.tell_host(f"GPUCoadd: stack saved: {result.get('stack_file')}")
                    if result.get("contributing") is not None:
                        self.append(f"{result['contributing']} lights contribute to this crop, from "
                                    f"{result.get('retained')} retained across the full field.")
                    for line in logic.group_summary(result.get("groups") or []):
                        self.append(line)
                self.progress_text.setText(f"Completed at {logic.stamp()} · report saved in {result.get('folder')}")
                self.append(logic.run_summary_line(result))
                if result.get("warnings"):
                    title = ("Stack saved, but requested outputs are missing or incomplete:"
                             if operation == "stack" else "Completed with output warnings:")
                    message = "\n\n".join([title] + [w["message"] for w in result["warnings"]])
                    self.output_warning.setText(message)
                    self.output_warning.show()
                    self.status.setText("Stack saved with warnings: check missing outputs below."
                                        if operation == "stack" else "Completed with warnings: see below.")
                    self.append(message)
            elif result.get("status") == "cancelled":
                self.status.setText("Operation cancelled.")
                self.progress_text.setText(f"Cancelled at {logic.stamp()}")
                self.append(f"Cancelled after {logic.duration(elapsed)}.")
            else:
                raise RuntimeError(result.get("error") or f"Worker exited with code {exit_code}")
        except Exception as exc:
            self.progress_text.setText(f"Failed at {logic.stamp()}")
            self.append(f"ERROR: {exc}")
            if not result.get("error"):
                result["error"] = str(exc)
            self.show_failure(result)
        self.set_busy(False)
        self.map.update()
        cache = result.get("worker_cache")
        if cache:
            self.cache_state.setText(logic.cache_text(cache))
            self.worker_idle_at = time.time()
            if result.get("operation") != "scan" and result.get("status") == "complete":
                reused = result.get("reused_prepared_lights") or result.get("reused_gpu_lights")
                self.append(f"Reused prepared lights ({cache.get('storage_mode') or 'gpu'})." if reused else
                            "Prepared lights for this operation.")
            if result.get("status") != "complete":
                self.releasing = True
                for button in (self.build, self.stack_button, self.check_folders):
                    button.setEnabled(False)

    def remove_request_files(self):
        if not self.request_file:
            return
        for path in logic.request_files(self.request_file):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass

    def worker_finished(self, exit_code, _status=None):
        self.read_worker()
        self.read_worker_errors()
        if self.busy:
            try:
                result = json.loads(Path(logic.result_file(self.request_file)).read_text(encoding="utf-8-sig"))
                self.finish_job(result, exit_code)
            except Exception as exc:
                self.timer.stop()
                self.set_busy(False)
                self.stages.finish("error", time.time())
                self.render_stages()
                self.show_failure({"summary": "The processing worker stopped unexpectedly.",
                                   "error": f"Worker stopped with exit code {exit_code} and no readable final report. "
                                            f"{exc}"})
            self.remove_request_files()
        self.session_dir = None
        self.worker_persistent = False
        self.releasing = False
        self.cache_state.setText("GPU lights released")
        self.set_busy(False)
        if self.pending_operation:
            pending, self.pending_operation = self.pending_operation, None
            self.start_job(pending)

    def worker_error(self, error):
        if error != QProcess.ProcessError.FailedToStart:
            return  # crashes end in worker_finished, which reads the result or reports the stop
        self.append(f"Worker error {error.name}. Check Python and CUDA under Engine settings.")
        self.session_dir = None
        self.worker_persistent = False
        self.releasing = False
        self.timer.stop()
        self.set_busy(False)
        self.cache_state.setText("GPU lights released")
        self.stages.finish("error", time.time())
        self.render_stages()
        self.show_failure({"summary": "GPUCoadd could not start the processing worker.",
                           "next_step": "Check the Python path under Engine settings, then retry.",
                           "error": f"Could not start the worker ({error.name}). Check Python and CUDA under Engine "
                                    "settings."})

    def release_gpu(self):
        if self.busy or not self.session_dir:
            return
        try:
            Path(self.session_dir, "stop").write_text("stop\n", encoding="utf-8")
        except OSError:
            pass
        self.releasing = True
        self.cache_state.setText("Releasing GPU worker…")
        for widget in (self.release_button, self.keep_ready, self.build, self.stack_button, self.check_folders):
            widget.setEnabled(False)

    def cancel_job(self):
        if not self.busy or not self.request_file:
            return
        Path(logic.cancel_file(self.request_file)).write_text("cancel\n", encoding="utf-8")
        self.cancel.setEnabled(False)
        self.status.setText("Cancelling: waiting for the current work unit and registration workers to finish…")
        self.append(f"Cancellation requested at {logic.stamp()}.")

    def tick(self):
        elapsed = time.time() - self.started_at
        remaining = ("estimating…" if self.stage_eta is None else
                     "~" + logic.duration(max(0, self.stage_eta - (time.time() - self.eta_at))))
        self.timing.setText(f"Started: {time.strftime('%H:%M:%S', time.localtime(self.started_at))}    Elapsed: "
                            f"{logic.duration(elapsed)}    Current stage remaining: {remaining} (estimate)")
        self.render_stages()

    # ---- startup GPU check ---------------------------------------------------------------------
    def stop_hardware_check(self):
        self.hardware_delay.stop()
        self.hardware_timeout.stop()
        probe, self.hardware_probe = self.hardware_probe, None
        if probe is not None and probe.state() != QProcess.ProcessState.NotRunning:
            probe.kill()
            probe.waitForFinished(2000)

    def apply_hardware_result(self, result):
        self.hardware_result = result
        self.hardware_state = "ready" if result.get("supported") is True else "blocked"
        # the first sentence (GPU and its memory) in the header; RAM, drives and set sizes are in the
        # tooltip and the log, which keeps the left column short
        summary = result.get("summary", "")
        cut = summary.find(". ")
        self.hardware_status.setText(summary[: cut + 1] if result.get("supported") is True and cut > 0 else summary)
        self.hardware_status.setToolTip(summary)
        warnings = result.get("warnings") or [] if result.get("supported") is True else []
        self.resource_message.setText("\n".join(warnings))
        self.resource_panel.setVisible(bool(warnings))
        for warning in warnings:
            self.append("WARNING: " + warning)
        self.fit_layout()
        self.set_busy(self.busy)
        if result.get("supported") is True:
            self.append(result["summary"] + " No lights were read.")
            if self.last_failure and self.last_failure.get("operation") == "hardware_check":
                self.last_failure = None
                self.failure_panel.setVisible(False)
                self.fit_layout()
            self.status.setText("Ready: choose the target folder and build the heat map.")
        else:
            self.show_failure({"failure": logic.hardware_failure(result, self.python.edit.text())})

    def schedule_hardware_check(self):
        self.stop_hardware_check()
        self.hardware_state = "checking"
        self.hardware_status.setText("Checking NVIDIA/CUDA support…")
        self.set_busy(self.busy)
        self.hardware_delay.start()

    def check_hardware(self):
        if self.busy or self.hardware_closed:
            return
        self.stop_hardware_check()
        self.hardware_state = "checking"
        self.hardware_status.setText("Checking NVIDIA/CUDA support…")
        self.set_busy(False)
        python = self.python.edit.text().strip()
        probe = QProcess(self)
        self.hardware_probe = probe
        settled = []

        def finish(result):
            if settled or self.hardware_probe is not probe or self.hardware_closed or \
                    python != self.python.edit.text().strip():
                return
            settled.append(True)
            self.hardware_timeout.stop()
            self.hardware_probe = None
            self.apply_hardware_result(result)

        def finished(code, _status=None):
            stdout = bytes(probe.readAllStandardOutput()).decode("utf-8", errors="replace")
            stderr = bytes(probe.readAllStandardError()).decode("utf-8", errors="replace")
            finish(logic.hardware_result(code, stdout, stderr))

        def failed(error):
            if error == QProcess.ProcessError.FailedToStart:
                finish({"supported": False, "code": "python_start_failed",
                        "summary": "The selected Python environment could not be started.",
                        "next_step": "Choose a valid Python executable under Engine settings, then retry.",
                        "error": error.name})

        probe.finished.connect(finished)
        probe.errorOccurred.connect(failed)
        if not os.path.isfile(python):
            finish({"supported": False, "code": "python_start_failed",
                    "summary": "The selected Python environment could not be started.",
                    "next_step": "Choose a valid Python executable under Engine settings, then retry.",
                    "error": f"Python executable not found: {python}"})
            return
        self.hardware_timeout.start()
        probe.setWorkingDirectory(str(logic.PROJECT_DIR))
        probe.setProcessEnvironment(worker_process_environment(self.host))
        probe.start(python, logic.hardware_args(self.results_folder()))

    def hardware_timed_out(self):
        self.stop_hardware_check()
        if not self.hardware_closed:
            self.apply_hardware_result({"supported": False, "code": "probe_timeout",
                                        "summary": "The NVIDIA/CUDA check did not finish.",
                                        "next_step": "Check the Python environment and NVIDIA driver, then retry."})

    # ---- closing -------------------------------------------------------------------------------
    def close_clicked(self):
        self.pending_operation = None
        self.release_gpu()
        self.accept()

    def done(self, code):
        if self.busy:
            return  # finish or cancel the running operation first
        self.save_session_state()
        self.hardware_closed = True
        self.stop_hardware_check()
        self.pending_operation = None
        self.release_gpu()
        super().done(code)

    def reject(self):  # Escape and the window's close button
        self.done(QDialog.DialogCode.Rejected)

    def shutdown(self):
        """After the window closed: stop the worker cleanly so the GPU memory is free when the host
        gets control back (the stop file first, then a kill if it does not stop)."""
        self.hardware_closed = True
        self.stop_hardware_check()
        if self.busy and self.request_file:
            self.cancel_job()
        if self.session_dir:
            try:
                Path(self.session_dir, "stop").write_text("stop\n", encoding="utf-8")
            except OSError:
                pass
        if self.worker_running() and not self.worker.waitForFinished(30000):
            self.worker.kill()
            self.worker.waitForFinished(5000)


class ErrorReport(QDialog):
    def __init__(self, parent, failure):
        super().__init__(parent)
        self.failure = failure
        self.setWindowTitle("GPUCoadd: Error Report")
        summary = wrapped(f"{failure.get('summary')}\n{failure.get('next_step')}")
        privacy = wrapped("Includes local file paths, settings and recent messages. Review before sharing.")
        self.details = QPlainTextEdit(logic.failure_text(failure))
        self.details.setReadOnly(True)
        self.details.setMinimumSize(640, 300)
        self.details.moveCursor(self.details.textCursor().MoveOperation.Start)  # the cause first
        self.notice = wrapped(f"Saved report: {failure['error_report']}" if failure.get("error_report") else
                              "Automatic report saving was unavailable. Copy or export the details below.")
        copy = QPushButton("Copy Error Details")
        copy.setToolTip("Copies the complete report to the clipboard so you can paste it into a message.")
        copy.clicked.connect(self.copy)
        save = QPushButton("Export Error Log…")
        save.setToolTip("Saves the full report as a text file you can attach when reporting a problem.")
        save.clicked.connect(self.export)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(copy)
        row.addWidget(save)
        row.addStretch()
        row.addWidget(close)
        box = QVBoxLayout(self)
        box.setContentsMargins(12, 12, 12, 12)
        for widget in (summary, privacy):
            box.addWidget(widget)
        box.addWidget(self.details, 100)
        box.addWidget(self.notice)
        box.addLayout(row)

    def copy(self):
        QApplication.clipboard().setText(logic.failure_text(self.failure))
        self.notice.setText("Error report copied to the clipboard.")

    def export(self):
        parent = self.parent()
        start = parent.output.edit.text() if parent is not None else ""
        path = QFileDialog.getSaveFileName(self, "Export GPUCoadd error report",
                                           f"{start}/GPUCoadd-error-{now_ms()}.txt", "Text (*.txt)")[0]
        if not path:
            return
        try:
            Path(path).write_text(logic.failure_text(self.failure), encoding="utf-8")
            self.notice.setText(f"Error report exported: {path}")
            if parent is not None:
                parent.append(f"Error report exported: {path}")
        except OSError as exc:
            self.notice.setText(f"Could not export error report: {exc}")


def run(host=None):
    """Shows the window until it is closed; returns after the worker has stopped."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    dialog = GPUCoaddDialog(host)
    icon = Path(__file__).resolve().parents[1] / "gpucoadd.ico"
    if icon.is_file():
        dialog.setWindowIcon(QIcon(str(icon)))
    dialog.show()
    try:
        app.exec()
    finally:
        dialog.shutdown()
    return dialog
