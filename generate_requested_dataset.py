#!/usr/bin/env python3
"""Generate a .grim dataset with fixed dimensions requested by the user.

Axis layout:
- Frequency: 11 samples (GHz)
- Polarization: 4 samples (HH, HV, VH, VV)
- Elevation: 9 samples (deg)
- Azimuth: -180 to 180 by 0.1 deg (inclusive)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grim_dataset import RcsGrid


def build_dataset(
    freq_start_ghz: float = 8.0,
    freq_stop_ghz: float = 18.0,
) -> RcsGrid:
    azimuths = np.round(np.arange(-180.0, 180.0 + 0.05, 0.1), 1)
    elevations = np.linspace(-20.0, 20.0, 9, endpoint=True)
    frequencies = np.linspace(freq_start_ghz, freq_stop_ghz, 11, endpoint=True)
    polarizations = np.asarray(["HH", "HV", "VH", "VV"])

    az_rad = np.deg2rad(azimuths)[:, None, None, None]
    el_rad = np.deg2rad(elevations)[None, :, None, None]
    p_idx = np.arange(len(polarizations), dtype=float)[None, None, None, :]
    f_norm = (frequencies - frequencies.min()) / max(np.ptp(frequencies), 1e-12)
    f_norm = f_norm[None, None, :, None]

    amplitude = 1.0 + 0.25 * np.cos(2.0 * az_rad) + 0.10 * np.sin(el_rad) + 0.20 * f_norm + 0.05 * p_idx
    phase = 0.8 * az_rad + 0.5 * el_rad + (2.0 * np.pi) * f_norm + 0.3 * p_idx
    rcs = amplitude * np.exp(1j * phase)

    return RcsGrid(
        azimuths=azimuths,
        elevations=elevations,
        frequencies=frequencies,
        polarizations=polarizations,
        rcs=rcs.astype(np.complex128),
        units={
            "azimuth": "deg",
            "elevation": "deg",
            "frequency": "GHz",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("dataset_11f_4pol_9el_az0p1.grim"),
        help="Output .grim path (default: %(default)s)",
    )
    parser.add_argument(
        "--freq-start-ghz",
        type=float,
        default=8.0,
        help="Start frequency in GHz (default: %(default)s)",
    )
    parser.add_argument(
        "--freq-stop-ghz",
        type=float,
        default=18.0,
        help="Stop frequency in GHz (default: %(default)s)",
    )
    args = parser.parse_args()

    dataset = build_dataset(freq_start_ghz=args.freq_start_ghz, freq_stop_ghz=args.freq_stop_ghz)
    saved_path = dataset.save(str(args.output))

    print(f"Wrote dataset: {saved_path}")
    print(
        "Shape: "
        f"az={len(dataset.azimuths)} "
        f"el={len(dataset.elevations)} "
        f"f={len(dataset.frequencies)} "
        f"pol={len(dataset.polarizations)}"
    )


if __name__ == "__main__":
    main()
