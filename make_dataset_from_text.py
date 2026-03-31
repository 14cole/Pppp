#!/usr/bin/env python3
"""Build a .grim dataset from a text file of RCS samples.

Expected text columns:
azimuth_deg elevation_deg frequency polarization rcs_real rcs_imag

Example input line:
0.0 0.0 9.6 HH 1.23 -0.04
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grim_dataset import RcsGrid


def parse_rows(path: Path) -> list[tuple[float, float, float, str, float, float]]:
    rows: list[tuple[float, float, float, str, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.replace(",", " ").split()
            if len(tokens) < 6:
                raise ValueError(
                    f"{path}:{line_number} needs 6 columns, found {len(tokens)}"
                )

            try:
                azimuth = float(tokens[0])
                elevation = float(tokens[1])
                frequency = float(tokens[2])
                polarization = str(tokens[3])
                rcs_real = float(tokens[4])
                rcs_imag = float(tokens[5])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} contains invalid values") from exc

            rows.append((azimuth, elevation, frequency, polarization, rcs_real, rcs_imag))

    if not rows:
        raise ValueError(f"{path} had no parseable rows")
    return rows


def unique_preserve_order(values: list[str]) -> np.ndarray:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return np.asarray(ordered)


def build_grid(rows: list[tuple[float, float, float, str, float, float]]) -> RcsGrid:
    azimuths = np.asarray(sorted({row[0] for row in rows}), dtype=float)
    elevations = np.asarray(sorted({row[1] for row in rows}), dtype=float)
    frequencies = np.asarray(sorted({row[2] for row in rows}), dtype=float)
    polarizations = unique_preserve_order([row[3] for row in rows])

    shape = (len(azimuths), len(elevations), len(frequencies), len(polarizations))
    rcs = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)

    az_index = {value: i for i, value in enumerate(azimuths)}
    el_index = {value: i for i, value in enumerate(elevations)}
    f_index = {value: i for i, value in enumerate(frequencies)}
    p_index = {value: i for i, value in enumerate(polarizations)}

    for azimuth, elevation, frequency, polarization, rcs_real, rcs_imag in rows:
        key = (
            az_index[azimuth],
            el_index[elevation],
            f_index[frequency],
            p_index[polarization],
        )
        if not np.isnan(rcs[key].real):
            raise ValueError(
                "duplicate coordinate detected for "
                f"(az={azimuth}, el={elevation}, f={frequency}, pol={polarization})"
            )
        rcs[key] = rcs_real + 1j * rcs_imag

    missing_points = int(np.isnan(rcs.real).sum())
    if missing_points:
        raise ValueError(
            f"grid is incomplete: {missing_points} points are missing in the text file"
        )

    return RcsGrid(
        azimuths=azimuths,
        elevations=elevations,
        frequencies=frequencies,
        polarizations=polarizations,
        rcs=rcs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_text", type=Path, help="Path to the text file")
    parser.add_argument("output_grim", type=Path, help="Output dataset path (.grim optional)")
    args = parser.parse_args()

    rows = parse_rows(args.input_text)
    grid = build_grid(rows)
    saved_path = grid.save(str(args.output_grim))

    print(f"Wrote dataset: {saved_path}")
    print(
        f"Shape: az={len(grid.azimuths)} el={len(grid.elevations)} "
        f"f={len(grid.frequencies)} pol={len(grid.polarizations)}"
    )


if __name__ == "__main__":
    main()
