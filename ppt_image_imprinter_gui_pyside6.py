"""
PowerPoint Image Imprinter - PySide6 Edition

Features
--------
- Pretty PySide6 GUI
- Capture size, crop, and location from currently selected image shapes in an open PowerPoint
- Imprint captured values onto other selected images
- Toggle boxes for:
    - Imprint location
    - Imprint size
    - Imprint crop
- Capture button updates to show how many images were captured, e.g. "Capture (4)"

Requirements
------------
- Windows
- Desktop Microsoft PowerPoint
- pywin32: pip install pywin32
- PySide6: pip install PySide6
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List

try:
    import pythoncom
    import win32com.client
except ImportError as exc:
    raise SystemExit(
        "This script requires pywin32.\n"
        "Install it with:\n"
        "    pip install pywin32"
    ) from exc

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit(
        "This script requires PySide6.\n"
        "Install it with:\n"
        "    pip install PySide6"
    ) from exc


PP_SELECTION_SHAPES = 2
MSO_PICTURE = 13
MSO_LINKED_PICTURE = 11
MSO_PLACEHOLDER = 14
MSO_GROUP = 6


@dataclass
class ShapeProfile:
    slide_index: int
    name: str
    left: float
    top: float
    width: float
    height: float
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0


class PowerPointBridge:
    def get_ppt_app(self):
        pythoncom.CoInitialize()
        try:
            try:
                app = win32com.client.GetActiveObject("PowerPoint.Application")
            except Exception:
                app = win32com.client.Dispatch("PowerPoint.Application")
            app.Visible = True
            return app
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to PowerPoint. Make sure desktop PowerPoint is open."
            ) from exc

    def get_selected_shapes(self):
        app = self.get_ppt_app()
        if app.Presentations.Count == 0:
            raise RuntimeError("No PowerPoint presentation is open.")

        window = app.ActiveWindow
        if window is None:
            raise RuntimeError("No active PowerPoint window found.")

        selection = window.Selection
        if selection is None or selection.Type != PP_SELECTION_SHAPES:
            raise RuntimeError("Select one or more shapes in PowerPoint first.")

        return selection.ShapeRange

    @staticmethod
    def shape_is_picture_like(shp) -> bool:
        try:
            shp_type = int(shp.Type)
        except Exception:
            return False

        if shp_type in (MSO_PICTURE, MSO_LINKED_PICTURE):
            return True

        if shp_type == MSO_PLACEHOLDER:
            try:
                return bool(getattr(shp, "PictureFormat", None))
            except Exception:
                return False

        return False

    def iter_selected_picture_shapes(self):
        shape_range = self.get_selected_shapes()
        pictures = []
        skipped = 0

        for i in range(1, shape_range.Count + 1):
            shp = shape_range.Item(i)

            if int(shp.Type) == MSO_GROUP:
                try:
                    group_items = shp.GroupItems
                    for gi in range(1, group_items.Count + 1):
                        gshp = group_items.Item(gi)
                        if self.shape_is_picture_like(gshp):
                            pictures.append(gshp)
                        else:
                            skipped += 1
                except Exception:
                    skipped += 1
                continue

            if self.shape_is_picture_like(shp):
                pictures.append(shp)
            else:
                skipped += 1

        if not pictures:
            raise RuntimeError("No picture shapes found in the current selection.")

        return pictures, skipped

    @staticmethod
    def capture_profile_from_shape(shp) -> ShapeProfile:
        crop_left = crop_top = crop_right = crop_bottom = 0.0

        try:
            picfmt = shp.PictureFormat
            crop_left = float(picfmt.CropLeft)
            crop_top = float(picfmt.CropTop)
            crop_right = float(picfmt.CropRight)
            crop_bottom = float(picfmt.CropBottom)
        except Exception:
            pass

        slide_index = -1
        try:
            slide_index = int(shp.Parent.SlideIndex)
        except Exception:
            pass

        return ShapeProfile(
            slide_index=slide_index,
            name=str(shp.Name),
            left=float(shp.Left),
            top=float(shp.Top),
            width=float(shp.Width),
            height=float(shp.Height),
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bridge = PowerPointBridge()
        self.captured_profiles: List[ShapeProfile] = []

        self.setWindowTitle("PowerPoint Image Imprinter")
        self.resize(760, 520)
        self.setMinimumSize(680, 440)

        self._build_ui()
        self._apply_styles()
        self.set_status("Ready. Open PowerPoint, select image(s), then click Capture.")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        title = QLabel(
            "Capture image geometry from selected PowerPoint shapes, then imprint it onto other selected images."
        )
        title.setWordWrap(True)
        title.setObjectName("TitleLabel")
        root.addWidget(title)

        options_box = QGroupBox("Imprint options")
        options_layout = QVBoxLayout(options_box)
        options_layout.setSpacing(10)

        self.chk_location = QCheckBox("Imprint location (Left / Top)")
        self.chk_size = QCheckBox("Imprint size (Width / Height)")
        self.chk_crop = QCheckBox("Imprint crop (Left / Top / Right / Bottom)")

        self.chk_location.setChecked(True)
        self.chk_size.setChecked(True)
        self.chk_crop.setChecked(True)

        options_layout.addWidget(self.chk_location)
        options_layout.addWidget(self.chk_size)
        options_layout.addWidget(self.chk_crop)
        root.addWidget(options_box)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self.capture_btn = QPushButton("Capture")
        self.imprint_btn = QPushButton("Imprint")
        self.clear_btn = QPushButton("Clear Capture")

        self.capture_btn.clicked.connect(self.capture)
        self.imprint_btn.clicked.connect(self.imprint)
        self.clear_btn.clicked.connect(self.clear_capture)

        self.capture_btn.setMinimumHeight(40)
        self.imprint_btn.setMinimumHeight(40)
        self.clear_btn.setMinimumHeight(40)

        button_row.addWidget(self.capture_btn)
        button_row.addWidget(self.imprint_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch(1)

        root.addLayout(button_row)

        summary_box = QGroupBox("Captured summary")
        summary_layout = QVBoxLayout(summary_box)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText("No capture yet.")
        summary_layout.addWidget(self.summary)
        root.addWidget(summary_box, 1)

        self.status_card = QFrame()
        self.status_card.setObjectName("StatusCard")
        status_layout = QHBoxLayout(self.status_card)
        status_layout.setContentsMargins(12, 10, 12, 10)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        root.addWidget(self.status_card)

    def _apply_styles(self):
        font = QFont()
        font.setPointSize(10)
        QApplication.instance().setFont(font)

        self.setStyleSheet("""
        QWidget {
            background: #0f172a;
            color: #e5e7eb;
        }

        QMainWindow {
            background: #0f172a;
        }

        QLabel#TitleLabel {
            font-size: 16px;
            font-weight: 600;
            color: #f8fafc;
            padding-bottom: 4px;
        }

        QGroupBox {
            border: 1px solid #334155;
            border-radius: 12px;
            margin-top: 10px;
            padding-top: 12px;
            background: #111827;
            font-weight: 600;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px 0 6px;
            color: #cbd5e1;
        }

        QTextEdit {
            border: 1px solid #334155;
            border-radius: 10px;
            background: #020617;
            color: #dbeafe;
            padding: 8px;
            selection-background-color: #2563eb;
        }

        QCheckBox {
            spacing: 10px;
            color: #e5e7eb;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }

        QCheckBox::indicator:unchecked {
            border: 1px solid #64748b;
            border-radius: 5px;
            background: #0f172a;
        }

        QCheckBox::indicator:checked {
            border: 1px solid #60a5fa;
            border-radius: 5px;
            background: #2563eb;
        }

        QPushButton {
            background: #1d4ed8;
            border: none;
            border-radius: 10px;
            padding: 10px 18px;
            color: white;
            font-weight: 600;
        }

        QPushButton:hover {
            background: #2563eb;
        }

        QPushButton:pressed {
            background: #1e40af;
        }

        QPushButton:disabled {
            background: #475569;
            color: #cbd5e1;
        }

        QFrame#StatusCard {
            border: 1px solid #334155;
            border-radius: 10px;
            background: #111827;
        }
        """)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_summary(self, text: str):
        self.summary.setPlainText(text)

    def update_capture_button_text(self):
        if self.captured_profiles:
            self.capture_btn.setText(f"Capture ({len(self.captured_profiles)})")
        else:
            self.capture_btn.setText("Capture")

    def show_error(self, title: str, text: str):
        QMessageBox.critical(self, title, text)

    def show_info(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def show_warning(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    def capture(self):
        try:
            pics, skipped = self.bridge.iter_selected_picture_shapes()
            self.captured_profiles = [
                self.bridge.capture_profile_from_shape(shp) for shp in pics
            ]
            self.update_capture_button_text()

            lines = [f"Captured {len(self.captured_profiles)} image profile(s):", ""]
            for idx, prof in enumerate(self.captured_profiles, start=1):
                lines.append(
                    f"{idx}. Slide {prof.slide_index} | {prof.name}\n"
                    f"   Left={prof.left:.2f}, Top={prof.top:.2f}, "
                    f"Width={prof.width:.2f}, Height={prof.height:.2f}\n"
                    f"   Crop L/T/R/B = "
                    f"{prof.crop_left:.2f} / {prof.crop_top:.2f} / "
                    f"{prof.crop_right:.2f} / {prof.crop_bottom:.2f}"
                )

            if skipped:
                lines.append("")
                lines.append(f"Skipped non-picture shapes: {skipped}")

            self.set_summary("\n".join(lines))
            if skipped:
                self.set_status(
                    f"Captured {len(self.captured_profiles)} image(s). Skipped {skipped} non-picture shape(s)."
                )
            else:
                self.set_status(f"Captured {len(self.captured_profiles)} image(s).")
        except Exception as exc:
            self.show_error("Capture failed", str(exc))
            self.set_status("Capture failed.")

    def apply_profile_to_shape(self, shp, prof: ShapeProfile):
        if self.chk_crop.isChecked():
            try:
                picfmt = shp.PictureFormat
                picfmt.CropLeft = prof.crop_left
                picfmt.CropTop = prof.crop_top
                picfmt.CropRight = prof.crop_right
                picfmt.CropBottom = prof.crop_bottom
            except Exception as exc:
                raise RuntimeError(
                    f"Could not apply crop to '{shp.Name}'. PowerPoint did not expose PictureFormat."
                ) from exc

        if self.chk_size.isChecked():
            try:
                shp.LockAspectRatio = 0
            except Exception:
                pass
            shp.Width = prof.width
            shp.Height = prof.height

        if self.chk_location.isChecked():
            shp.Left = prof.left
            shp.Top = prof.top

    def imprint(self):
        if not self.captured_profiles:
            self.show_warning("Nothing captured", "Capture one or more images first.")
            return

        if not (
            self.chk_location.isChecked()
            or self.chk_size.isChecked()
            or self.chk_crop.isChecked()
        ):
            self.show_warning("No options selected", "Turn on at least one imprint option.")
            return

        try:
            dest_shapes, skipped = self.bridge.iter_selected_picture_shapes()

            if len(dest_shapes) == len(self.captured_profiles):
                mapping = list(zip(dest_shapes, self.captured_profiles))
                mode = "1-to-1"
            else:
                mapping = [(shp, self.captured_profiles[0]) for shp in dest_shapes]
                mode = (
                    f"count mismatch ({len(self.captured_profiles)} captured vs "
                    f"{len(dest_shapes)} selected), so the first captured profile was reused"
                )

            for shp, prof in mapping:
                self.apply_profile_to_shape(shp, prof)

            status = f"Imprint complete: {len(dest_shapes)} image(s) updated using {mode}."
            if skipped:
                status += f" Skipped {skipped} non-picture shape(s)."

            self.set_status(status)
            self.show_info(
                "Imprint complete",
                f"Updated {len(dest_shapes)} image(s).\n\nMode: {mode}"
                + (f"\nSkipped non-picture shapes: {skipped}" if skipped else "")
            )
        except Exception as exc:
            self.show_error("Imprint failed", str(exc))
            self.set_status("Imprint failed.")

    def clear_capture(self):
        self.captured_profiles = []
        self.update_capture_button_text()
        self.set_summary("No capture yet.")
        self.set_status("Captured profiles cleared.")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
