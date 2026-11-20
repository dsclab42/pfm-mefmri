#!/usr/bin/env python3
"""Remove motor stripe artifacts (face dependence) from a CIFTI dtseries.

Python port of Regress_face_dependence_fromcifti.m (Gordon et al.):
  - Per cortical vertex, correlate its FC to nearby vertices with the
    anterior-posterior component of the pial surface normal ("face dependence")
  - PCA the timeseries of stripy vertices near the central sulcus
  - Regress components whose FC tracks the pial normal out of every grayordinate

Only cortex x cortex FC is computed, in blocks, so the full grayordinate FC
matrix the MATLAB code builds is never held in memory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import scipy.sparse as sp
from nibabel.cifti2.cifti2_axes import BrainModelAxis, ScalarAxis, SeriesAxis
from scipy.spatial.distance import cdist

SURFACE_NORMAL_COMPONENT = 1  # y (anterior-posterior) component of the pial normal (MATLAB column 2)
DISTANCE_LIMIT_MM = 20.0  # Euclidean, on the pial surface
EXCLUDE_CLOSEST_HOPS = 2  # vertices within this many surface edges of the seed are skipped
STRIPY_THRESHOLD = 0.5  # face dependence above which central sulcus ROI vertices count as stripy
DEFAULT_ROI = Path(__file__).resolve().parents[1] / "res0urces" / "Face_dependence_regression_ROIs.dscalar.nii"


def cortex_layout(bm: BrainModelAxis) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per hemisphere (left, right): cortical grayordinate indices and their surface vertex ids."""
    out = []
    for name in ("CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"):
        sel = bm.name == name
        out.append((np.where(sel)[0], bm.vertex[sel]))
    return out


def pial_normals(
    bm: BrainModelAxis, left_pial: Path, right_pial: Path
) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Pial coordinates, mean unit face normal and surface adjacency per cortical grayordinate.

    Normals are the mean of the unit normals of the faces a vertex touches (not renormalized),
    as compute_pialnormvec. Edges through the medial wall are dropped from the adjacency.
    """
    layout = cortex_layout(bm)
    n_cort = sum(idx.size for idx, _ in layout)
    if n_cort == 0 or layout[0][0].size + layout[1][0].size != n_cort or max(i.max() for i, _ in layout) != n_cort - 1:
        raise ValueError("Expected cortical grayordinates first (left then right) in the CIFTI")
    coords = np.zeros((n_cort, 3))
    normals = np.zeros((n_cort, 3))
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for (idx, verts), surf_path in zip(layout, (left_pial, right_pial)):
        surf = nib.load(str(surf_path))
        points = surf.agg_data("NIFTI_INTENT_POINTSET").astype(np.float64)
        faces = surf.agg_data("NIFTI_INTENT_TRIANGLE")

        face_normals = np.cross(points[faces[:, 0]] - points[faces[:, 1]], points[faces[:, 0]] - points[faces[:, 2]])
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
        normal_sum = np.zeros_like(points)
        face_count = np.zeros(len(points))
        for corner in range(3):
            np.add.at(normal_sum, faces[:, corner], face_normals)
            np.add.at(face_count, faces[:, corner], 1)

        coords[idx] = points[verts]
        normals[idx] = normal_sum[verts] / face_count[verts, None]

        position = np.full(len(points), -1)  # surface vertex -> grayordinate (-1 = medial wall)
        position[verts] = idx
        for a, b in ((0, 1), (1, 2), (2, 0)):
            r, c = position[faces[:, a]], position[faces[:, b]]
            keep = (r >= 0) & (c >= 0)
            rows += [r[keep], c[keep]]
            cols += [c[keep], r[keep]]

    r_all, c_all = np.concatenate(rows), np.concatenate(cols)
    adjacency = sp.csr_matrix((np.ones(len(r_all), dtype=bool), (r_all, c_all)), shape=(n_cort, n_cort))
    return coords, normals, adjacency


def zscore_columns(data: np.ndarray) -> np.ndarray:
    """Columns demeaned and scaled to unit norm, so z.T @ z gives correlations; constant columns become 0."""
    z = data - data.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(z, axis=0, keepdims=True)
    return np.divide(z, norm, out=np.zeros_like(z), where=norm > 0)


def masked_row_correlation(Y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row i, the correlation of x[mask[i]] with Y[i, mask[i]]."""
    M = mask.astype(np.float64)
    MY = M * Y
    n, sx, sxx = M.sum(axis=1), M @ x, M @ x**2
    sy, syy, sxy = MY.sum(axis=1), (MY * Y).sum(axis=1), MY @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx**2) * (n * syy - sy**2))


def face_dependence(
    cortex_data: np.ndarray,
    coords: np.ndarray,
    normals: np.ndarray,
    adjacency: sp.csr_matrix,
    block_size: int = 1000,
    label: str = "",
) -> np.ndarray:
    """Per cortical vertex: |r| between its FC map and the pial normal's y component.

    Taken over vertices within DISTANCE_LIMIT_MM, skipping those within EXCLUDE_CLOSEST_HOPS
    surface edges. cortex_data is time x cortex.
    """
    z = zscore_columns(cortex_data.astype(np.float64)).astype(np.float32)
    hops = sp.identity(adjacency.shape[0], dtype=bool, format="csr")
    for _ in range(EXCLUDE_CLOSEST_HOPS):
        hops = hops + hops @ adjacency
    normal = normals[:, SURFACE_NORMAL_COMPONENT]

    n_cort = z.shape[1]
    dependence = np.zeros(n_cort)
    for start in range(0, n_cort, int(block_size)):
        seeds = np.arange(start, min(start + int(block_size), n_cort))
        fc = (z[:, seeds].T @ z).astype(np.float64)
        mask = (cdist(coords[seeds], coords) < DISTANCE_LIMIT_MM) & ~hops[seeds].toarray()
        dependence[seeds] = np.abs(masked_row_correlation(fc, normal, mask))
        if label and (start // int(block_size) + 1) % 20 == 0:
            print(f"[stripe] face dependence ({label}): {seeds[-1] + 1}/{n_cort}", flush=True)
    return dependence


def stripe_components(
    cortex_data: np.ndarray,
    stripy: np.ndarray,
    normal: np.ndarray,
    component_thresh: float,
    variance_explained: float,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """PCA of the stripy vertices; returns the component scores to regress and a per-component table.

    A component is regressed when its FC with the stripy vertices tracks the pial normal
    (|r| > component_thresh) and it explains more than variance_explained percent.
    """
    X = cortex_data[:, stripy].astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    n_comp = min(Xc.shape[0] - 1, Xc.shape[1])
    scores = (U * S)[:, :n_comp]
    explained = (S**2 / np.sum(S**2) * 100)[:n_comp]

    with np.errstate(divide="ignore", invalid="ignore"):
        comp_fc = np.arctanh(zscore_columns(X).T @ zscore_columns(scores))  # stripy vertices x components
        comp_dep = np.abs(zscore_columns(comp_fc).T @ zscore_columns(normal[stripy, None]))[:, 0]
        use = (comp_dep > component_thresh) & (explained > variance_explained)

    table = [
        {
            "component": i + 1,
            "variance_explained": float(explained[i]),
            "face_dependence": float(comp_dep[i]),
            "regressed": bool(use[i]),
        }
        for i in range(n_comp)
    ]
    return scores[:, use], table


def regress_out(data: np.ndarray, regressors: np.ndarray, chunk: int = 10000) -> np.ndarray:
    """Residuals of every column of data (time x grayordinates) on [regressors, 1], as MATLAB's regress."""
    X = np.column_stack([regressors, np.ones(len(data))])
    pinv = np.linalg.pinv(X)
    out = np.empty(data.shape, dtype=data.dtype)
    for start in range(0, data.shape[1], chunk):
        Y = data[:, start : start + chunk].astype(np.float64)
        out[:, start : start + chunk] = Y - X @ (pinv @ Y)
    return out


def remove_stripes(
    data: np.ndarray,
    coords: np.ndarray,
    normals: np.ndarray,
    adjacency: sp.csr_matrix,
    cs_roi: np.ndarray,
    dorsal_roi: np.ndarray,
    component_thresh: float = 0.4,
    variance_explained: float = 1.0,
    block_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    """Stripe regression on time x grayordinate data with cortex first.

    Returns the regressed data, face dependence before and after (per cortical vertex),
    a summary and the per-component table. Stripiness is the mean face dependence over
    the dorsal central sulcus ROI, the value the MATLAB function returns.
    """
    n_cort = coords.shape[0]
    cortex = data[:, :n_cort]
    before = face_dependence(cortex, coords, normals, adjacency, block_size, "before")
    with np.errstate(invalid="ignore"):
        stripy = (cs_roi & (before > STRIPY_THRESHOLD)) | dorsal_roi

    components, table = stripe_components(
        cortex, stripy, normals[:, SURFACE_NORMAL_COMPONENT], component_thresh, variance_explained
    )
    if components.shape[1] == 0:
        print("[stripe] no stripy components found; the data is only demeaned")
    regressed = regress_out(data, components)
    after = face_dependence(regressed[:, :n_cort], coords, normals, adjacency, block_size, "after")

    summary = {
        "stripy_vertices": int(stripy.sum()),
        "components": len(table),
        "components_regressed": int(components.shape[1]),
        "stripiness_before": float(np.mean(before[dorsal_roi])),
        "stripiness_after": float(np.mean(after[dorsal_roi])),
    }
    return regressed, before, after, summary, table


def save_dtseries(data_time_by_gray: np.ndarray, ref_img: nib.Cifti2Image, out_path: Path) -> None:
    axes = [ref_img.header.get_axis(i) for i in range(ref_img.ndim)]
    if len(axes) != 2 or not isinstance(axes[0], SeriesAxis):
        raise ValueError("Expected dtseries with SeriesAxis first")
    new_series = SeriesAxis(axes[0].start, axes[0].step, data_time_by_gray.shape[0], unit=axes[0].unit)
    new_header = nib.Cifti2Header.from_axes((new_series, axes[1]))
    out_img = nib.Cifti2Image(data_time_by_gray.astype(np.float32), new_header, nifti_header=ref_img.nifti_header)
    nib.save(out_img, str(out_path))


def save_dscalar(cortex_values: np.ndarray, bm: BrainModelAxis, name: str, out_path: Path) -> None:
    values = np.zeros((1, bm.size), dtype=np.float32)
    values[0, : cortex_values.size] = np.nan_to_num(cortex_values)
    header = nib.Cifti2Header.from_axes((ScalarAxis([name]), bm))
    nib.save(nib.Cifti2Image(values, header), str(out_path))


def write_summary(path: Path, summary: Dict[str, object], table: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        for key, value in summary.items():
            f.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(f, fieldnames=["component", "variance_explained", "face_dependence", "regressed"])
        writer.writeheader()
        writer.writerows(table)


def main() -> int:
    ap = argparse.ArgumentParser(description="Motor stripe (face dependence) regression for CIFTI dtseries")
    ap.add_argument("--in-cifti", required=True, help="Concatenated (and censored) dtseries")
    ap.add_argument("--out-cifti", required=True)
    ap.add_argument("--left-pial", required=True, help="Left pial surface on the dtseries' cortical mesh")
    ap.add_argument("--right-pial", required=True)
    ap.add_argument("--roi-cifti", default=str(DEFAULT_ROI), help="dscalar with [central sulcus, dorsal central sulcus] ROIs")
    ap.add_argument("--component-thresh", type=float, default=0.4, help="Minimum |r| between a component's FC and the pial normal")
    ap.add_argument("--variance-explained", type=float, default=1.0, help="Minimum percent variance a regressed component explains")
    ap.add_argument("--block-size", type=int, default=1000)
    ap.add_argument("--out-prefix", default="", help="Prefix for the face-dependence maps and summary; default: --out-cifti without .dtseries.nii")
    args = ap.parse_args()

    out_cifti = Path(args.out_cifti)
    out_cifti.parent.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or str(out_cifti).replace(".dtseries.nii", "")

    img = nib.load(args.in_cifti)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    bm = img.header.get_axis(1)
    if data.ndim != 2 or not isinstance(bm, BrainModelAxis):
        raise ValueError("Expected a 2D dtseries with a BrainModelAxis")

    coords, normals, adjacency = pial_normals(bm, Path(args.left_pial), Path(args.right_pial))
    n_cort = coords.shape[0]
    rois = np.asarray(nib.load(args.roi_cifti).dataobj).astype(bool)
    if rois.shape[0] < 2 or rois.shape[1] < n_cort:
        raise ValueError(f"ROI file must have 2 maps covering {n_cort} cortical grayordinates; got {rois.shape}")

    regressed, before, after, summary, table = remove_stripes(
        data,
        coords,
        normals,
        adjacency,
        rois[0, :n_cort],
        rois[1, :n_cort],
        float(args.component_thresh),
        float(args.variance_explained),
        int(args.block_size),
    )
    print(
        f"[stripe] stripiness {summary['stripiness_before']:.4f} -> {summary['stripiness_after']:.4f} "
        f"({summary['components_regressed']} of {summary['components']} components regressed)"
    )

    save_dscalar(before, bm, "face_dependence", Path(f"{prefix}_FaceDependence.dscalar.nii"))
    save_dscalar(after, bm, "face_dependence", Path(f"{prefix}_FaceDependence_PostRegression.dscalar.nii"))
    write_summary(Path(f"{prefix}_StripeRegression.csv"), summary, table)
    save_dtseries(regressed, img, out_cifti)
    print(f"[stripe] wrote {out_cifti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
