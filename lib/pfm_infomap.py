#!/usr/bin/env python3
"""PFM InfoMap community detection.

Implements the core workflow from PFM-InfoMap `pfm_wrapper.m`:
1) build density-thresholded weighted graphs from FC with local-edge removal
   (block-wise by default: only each node's strongest edges are kept, so the full
   node x node FC matrix is never held in memory; --graph-builder dense keeps the
   original full-matrix construction)
2) run InfoMap per graph density
3) collect memberships into Bipartite_PhysicalCommunities.dtseries.nii
4) remove communities smaller than 10 vertices per density
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np
import scipy.io as sio


def parse_num_list_or_colon(expr: str, as_int: bool = False) -> List[float]:
    expr = expr.strip()
    if not expr:
        return []
    if ":" in expr and "," not in expr:
        parts = [p.strip() for p in expr.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid colon expression: {expr}")
        start, step, stop = [float(x) for x in parts]
        if step == 0:
            raise ValueError("step cannot be zero")
        out = []
        x = start
        eps = abs(step) * 1e-9
        if step > 0:
            while x <= stop + eps:
                out.append(x)
                x += step
        else:
            while x >= stop - eps:
                out.append(x)
                x += step
    else:
        out = [float(x.strip()) for x in expr.split(",") if x.strip()]
    if as_int:
        return [int(round(x)) for x in out]
    return out


def format_density(d: float) -> str:
    return f"{d:.15g}"


def graph_density_dir_name(d: float) -> str:
    return "GraphDensity_" + format_density(float(d)).replace(".", "p")


def normalize_structure_name(name: str) -> str:
    n = str(name).strip().upper()
    if n.startswith("CIFTI_STRUCTURE_"):
        n = n[len("CIFTI_STRUCTURE_") :]
    return n


def _subset_from_h5_dataset(dset, good: np.ndarray, chunk_rows: int = 512) -> np.ndarray:
    # h5py datasets from MATLAB v7.3 can be indexed efficiently by rows.
    n = good.size
    out = np.empty((n, n), dtype=np.float32)
    # Try direct orientation first.
    try:
        for i0 in range(0, n, chunk_rows):
            i1 = min(i0 + chunk_rows, n)
            rows = good[i0:i1]
            out[i0:i1, :] = np.asarray(dset[rows, :][:, good], dtype=np.float32)
        return out
    except Exception:
        # Transposed storage fallback.
        for i0 in range(0, n, chunk_rows):
            i1 = min(i0 + chunk_rows, n)
            rows = good[i0:i1]
            out[i0:i1, :] = np.asarray(dset[:, rows][good, :].T, dtype=np.float32)
        return out


def load_distance_subset(path: Path, good: np.ndarray, n_gray: int, keep_integer_dtype: bool = False) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        d = np.load(path, mmap_mode="r")
        if d.shape[0] < n_gray or d.shape[1] < n_gray:
            raise RuntimeError(f"Distance matrix shape {d.shape} smaller than grayordinates {n_gray}")
        # A uint8 matrix kept as uint8 is a quarter of the float32 copy; thresholding gives the same result.
        dtype = d.dtype if keep_integer_dtype and np.issubdtype(d.dtype, np.integer) else np.float32
        if good.size == d.shape[0] == d.shape[1] and np.array_equal(good, np.arange(good.size)):
            return np.array(d, dtype=dtype)
        return np.asarray(d[np.ix_(good, good)], dtype=dtype)

    if path.suffix.lower() == ".mat":
        # Try scipy loadmat first (v7 and earlier).
        try:
            m = sio.loadmat(path, variable_names=["D"])
            if "D" in m:
                d = np.asarray(m["D"])
                if d.shape[0] < n_gray or d.shape[1] < n_gray:
                    raise RuntimeError(f"Distance matrix shape {d.shape} smaller than grayordinates {n_gray}")
                return np.asarray(d[np.ix_(good, good)], dtype=np.float32)
        except Exception:
            pass

        # Try HDF5 v7.3 without loading full matrix.
        try:
            import h5py  # type: ignore

            with h5py.File(path, "r") as f:
                if "D" not in f:
                    raise KeyError("Variable D not found")
                dset = f["D"]
                if dset.shape[0] < n_gray or dset.shape[1] < n_gray:
                    raise RuntimeError(f"Distance matrix shape {dset.shape} smaller than grayordinates {n_gray}")
                return _subset_from_h5_dataset(dset, good)
        except Exception as e:
            raise RuntimeError(f"Unable to load distance matrix from {path}: {e}") from e

    raise ValueError(f"Unsupported distance matrix format: {path}")


def write_pajek(path: Path, w: np.ndarray, conn: np.ndarray) -> int:
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    keep = conn[iu]
    x = iu[0][keep] + 1
    y = iu[1][keep] + 1
    z = w[iu][keep]

    with path.open("w", encoding="utf-8") as f:
        f.write(f"*Vertices {n}\n")
        for i in range(1, n + 1):
            f.write(f'{i} "{i}"\n')
        f.write(f"*Edges {len(x)}\n")
        for a, b, c in zip(x, y, z):
            f.write(f"{a} {b} {float(c):.9g}\n")
    return len(x)


def parse_clu(path: Path, n_nodes: int) -> np.ndarray:
    out = np.zeros((n_nodes,), dtype=np.int32)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                node = int(float(parts[0]))
                ci = int(float(parts[1]))
            except ValueError:
                continue
            if 1 <= node <= n_nodes:
                out[node - 1] = ci
    return out


def build_connection_matrix(fc: np.ndarray, density: float) -> np.ndarray:
    n = fc.shape[0]
    k = int(np.ceil(n * density))
    k = max(1, min(k, n))
    conn = np.zeros((n, n), dtype=bool)
    for i in range(n):
        col = fc[:, i]
        if not np.any(col):
            continue
        idx = np.argsort(col)[::-1][:k]
        conn[idx, i] = True
        conn[i, idx] = True
    np.fill_diagonal(conn, False)
    return conn


def top_edges_blockwise(
    X: np.ndarray,
    Dg: np.ndarray,
    min_distance: float,
    n_time: int,
    k_max: int,
    block_cols: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each node's k_max strongest FC edges, strongest first, without the full FC matrix.

    Equivalent to sorting every column of the masked dense FC matrix (as
    build_connection_matrix does) and keeping the first k_max entries: the FC of
    block_cols nodes against all nodes is computed, masked and ranked at a time.
    Returns (top_idx, top_val, active); active is False for nodes whose masked FC
    is all zero, which pick no edges.
    """
    n = X.shape[0]
    k_max = max(1, min(int(k_max), n))
    denom = max(n_time - 1, 1)
    top_idx = np.zeros((n, k_max), dtype=np.int32)
    top_val = np.zeros((n, k_max), dtype=np.float32)
    active = np.zeros((n,), dtype=bool)
    XT = np.ascontiguousarray(X.T)

    for c0 in range(0, n, int(block_cols)):
        c1 = min(c0 + int(block_cols), n)
        # blk[j, r] = fc[r, c0 + j]
        blk = (X[c0:c1] @ XT) / denom
        j = np.arange(c1 - c0)
        blk[j, c0 + j] = 0.0
        blk[Dg[:, c0:c1].T <= float(min_distance)] = 0.0
        np.nan_to_num(blk, copy=False)
        active[c0:c1] = blk.any(axis=1)

        if k_max < n:
            idx = np.argpartition(blk, n - k_max, axis=1)[:, n - k_max :]
        else:
            idx = np.tile(np.arange(n), (c1 - c0, 1))
        val = np.take_along_axis(blk, idx, axis=1)
        order = np.argsort(-val, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        val = np.take_along_axis(val, order, axis=1)

        # Fewer than k_max positive edges: zero-valued (masked) entries are picked too.
        # Rank those columns exactly as the dense builder does so the same ties are chosen.
        for jj in np.where(val[:, -1] <= 0)[0]:
            full = np.argsort(blk[jj])[::-1][:k_max]
            idx[jj] = full
            val[jj] = blk[jj, full]

        top_idx[c0:c1] = idx
        top_val[c0:c1] = val
    return top_idx, top_val, active


def edges_from_top(
    top_idx: np.ndarray,
    top_val: np.ndarray,
    active: np.ndarray,
    density: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper-triangle edge list (1-based x < y, weight) for one density.

    An edge is kept if either node picked the other among its top k, as in
    build_connection_matrix. Edges are ordered as write_pajek orders them.
    """
    n = top_idx.shape[0]
    k = int(np.ceil(n * density))
    k = max(1, min(k, n, top_idx.shape[1]))
    cols = np.where(active)[0]
    r = top_idx[cols, :k].ravel().astype(np.int64)
    c = np.repeat(cols.astype(np.int64), k)
    v = top_val[cols, :k].ravel()

    # Picks already in upper-triangle orientation (row < col) first, so their weight is the one kept.
    upper = r < c
    lower = r > c
    key = np.concatenate([r[upper] * n + c[upper], c[lower] * n + r[lower]])
    v = np.concatenate([v[upper], v[lower]])
    key, first = np.unique(key, return_index=True)
    return key // n + 1, key % n + 1, v[first]


def write_pajek_edges(
    path: Path,
    n: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    chunk: int = 1_000_000,
) -> int:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"*Vertices {n}\n")
        f.writelines(f'{i} "{i}"\n' for i in range(1, n + 1))
        f.write(f"*Edges {len(x)}\n")
        for s in range(0, len(x), chunk):
            e = s + chunk
            f.writelines("%d %d %.9g\n" % abc for abc in zip(x[s:e].tolist(), y[s:e].tolist(), z[s:e].tolist()))
    return len(x)


def run_infomap(
    infomap_bin: str,
    net_path: Path,
    outdir: Path,
    reps: int,
    logfile: Path,
) -> None:
    cmd = [
        infomap_bin,
        str(net_path),
        str(outdir),
        "--clu",
        "-2",
        "-s",
        "42",
        "-N",
        str(int(reps)),
        "--no-self-links",
    ]
    with logfile.open("a", encoding="utf-8") as lf:
        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT, text=True)


def brain_structure_vectors(bm_axis) -> Tuple[np.ndarray, List[str]]:
    n = bm_axis.size
    bs = np.zeros((n,), dtype=np.int32)
    labels: List[str] = []
    for name, slc, _sub in bm_axis.iter_structures():
        nm = normalize_structure_name(name)
        if nm not in labels:
            labels.append(nm)
        code = labels.index(nm) + 1
        stop = slc.stop if slc.stop is not None else n
        bs[slc.start:stop] = code
    return bs, labels


def main() -> int:
    ap = argparse.ArgumentParser(description="PFM InfoMap community detection")
    ap.add_argument("--in-cifti", required=True)
    ap.add_argument("--distance", required=True, help="Distance matrix (.mat or .npy)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--graph-densities", default="0.001:0.001:0.009")
    ap.add_argument("--num-reps", default="20", help="Single value or comma list matching densities")
    ap.add_argument("--min-distance", type=float, default=20.0)
    ap.add_argument("--structures-csv", default="", help="Comma list like CORTEX_LEFT,CORTEX_RIGHT; empty=all")
    ap.add_argument("--bad-verts-csv", default="", help="1-based vertex indices to remove")
    ap.add_argument("--num-cores", type=int, default=1)
    ap.add_argument("--infomap-binary", default="infomap")
    ap.add_argument("--min-community-size", type=int, default=10)
    ap.add_argument("--density-subdirs", type=int, choices=(0, 1), default=1)
    ap.add_argument(
        "--graph-builder",
        choices=("blockwise", "dense"),
        default="blockwise",
        help=(
            "blockwise keeps only each node's strongest edges while computing FC in blocks; "
            "dense builds the full node x node FC matrix (original behavior, much more memory and time)."
        ),
    )
    ap.add_argument("--fc-block-cols", type=int, default=1000, help="Nodes per FC block for --graph-builder blockwise")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_cifti = Path(args.in_cifti)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    d_path = Path(args.distance)

    densities = parse_num_list_or_colon(args.graph_densities, as_int=False)
    if not densities:
        raise ValueError("No graph densities provided")
    reps = parse_num_list_or_colon(args.num_reps, as_int=True)
    if len(reps) == 1:
        reps = reps * len(densities)
    if len(reps) != len(densities):
        raise ValueError("num-reps must be scalar or same length as graph-densities")

    if args.dry_run:
        print(
            f"[pfm_infomap] dry-run in={in_cifti} distance={d_path} outdir={outdir} "
            f"densities={densities} reps={reps} min_distance={args.min_distance} "
            f"structures='{args.structures_csv}' num_cores={args.num_cores} infomap='{args.infomap_binary}'"
        )
        return 0

    img = nib.load(str(in_cifti))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"Expected 2D CIFTI data, got {data.shape}")
    # dtseries is time x grayordinates
    t, n_gray = data.shape
    bm = img.header.get_axis(1)
    bs, bs_labels = brain_structure_vectors(bm)
    n_cort = int(np.sum(bs == 1) + np.sum(bs == 2))

    if args.structures_csv.strip():
        keep_struct = {normalize_structure_name(x) for x in args.structures_csv.split(",") if x.strip()}
    else:
        keep_struct = set(bs_labels)
    keep_codes = {i + 1 for i, nm in enumerate(bs_labels) if nm in keep_struct}
    good = np.where(np.isin(bs, list(keep_codes)))[0]

    if args.bad_verts_csv.strip():
        bad_1based = [int(x.strip()) for x in args.bad_verts_csv.split(",") if x.strip()]
        bad_0 = np.array(bad_1based, dtype=np.int64) - 1
        bad_0 = bad_0[(bad_0 >= 0) & (bad_0 < n_gray)]
        if bad_0.size > 0:
            good = good[~np.isin(good, bad_0)]

    if good.size < 2:
        raise RuntimeError("Too few good vertices after structure/bad-vertex filtering.")

    # Subset distance to good vertices (avoid loading full .mat where possible).
    blockwise = args.graph_builder == "blockwise"
    Dg = load_distance_subset(d_path, good, n_gray, keep_integer_dtype=blockwise)
    good_sub = good >= n_cort
    if np.any(good_sub):
        Dg[np.ix_(good_sub, good_sub)] = 0.0

    X = data[:, good].T  # nodes x time
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    X = X / sd
    n_nodes = X.shape[0]
    if blockwise:
        k_max = int(np.ceil(n_nodes * max(densities)))
        top_idx, top_val, active = top_edges_blockwise(
            X, Dg, float(args.min_distance), t, k_max, int(args.fc_block_cols)
        )
        del Dg
    else:
        fc = (X @ X.T) / max(t - 1, 1)
        np.fill_diagonal(fc, 0.0)
        fc[Dg <= float(args.min_distance)] = 0.0
        np.nan_to_num(fc, copy=False)

    out_membership = np.zeros((n_gray, len(densities)), dtype=np.float32)

    net_paths: List[Path] = []
    clu_paths: List[Path] = []
    tasks: List[Tuple[str, Path, Path, int, Path]] = []
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    for di, dens in enumerate(densities):
        dstr = format_density(float(dens))
        density_dir = outdir / graph_density_dir_name(float(dens)) if int(args.density_subdirs) == 1 else outdir
        density_dir.mkdir(parents=True, exist_ok=True)
        net_path = density_dir / f"Bipartite_Density{dstr}.net"
        clu_path = density_dir / f"Bipartite_Density{dstr}.clu"
        log_path = density_dir / f"Bipartite_Density{dstr}_LogFile_{ts}.txt"

        if blockwise:
            n_edges = write_pajek_edges(net_path, n_nodes, *edges_from_top(top_idx, top_val, active, float(dens)))
        else:
            conn = build_connection_matrix(fc, float(dens))
            n_edges = write_pajek(net_path, fc, conn)
        print(f"[pfm_infomap] density={dstr} nodes={n_nodes} edges={n_edges}")

        net_paths.append(net_path)
        clu_paths.append(clu_path)
        tasks.append((args.infomap_binary, net_path, density_dir, int(reps[di]), log_path))

    if args.num_cores > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=int(args.num_cores)) as ex:
            futs = [ex.submit(run_infomap, *task) for task in tasks]
            for fut in as_completed(futs):
                fut.result()
    else:
        for task in tasks:
            run_infomap(*task)

    for di, clu_path in enumerate(clu_paths):
        if not clu_path.exists():
            raise RuntimeError(f"Missing InfoMap output: {clu_path}")
        ci = parse_clu(clu_path, n_nodes)
        out_membership[good, di] = ci.astype(np.float32)

    # Remove small communities per density.
    for di in range(out_membership.shape[1]):
        col = out_membership[:, di]
        u = np.unique(col[col > 0]).astype(np.int32)
        for cid in u:
            idx = np.where(col == cid)[0]
            if idx.size < int(args.min_community_size):
                col[idx] = 0
        out_membership[:, di] = col

    series_axis = nib.cifti2.SeriesAxis(start=0, step=1, size=out_membership.shape[1], unit="SECOND")
    out_img = nib.Cifti2Image(out_membership.T, header=(series_axis, bm))
    out_path = outdir / "Bipartite_PhysicalCommunities.dtseries.nii"
    nib.save(out_img, str(out_path))
    print(f"[pfm_infomap] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
