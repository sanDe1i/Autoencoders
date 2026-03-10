"""
Utilities for flank-segment MUSTANG alignment and transferring the induced rigid-body transform
to the full PDB structure.

Core idea (ground truth):
- Align ONLY two flanking segments, excluding the activation loop between DFG and APE:
  Segment N: DFG-40 → DFG  (includes DFG motif)
  Segment C: APE → APE+40  (includes APE motif)
- Treat the two segments as ONE point set (concatenate CA coords) when estimating a rigid transform (R, t).
- Apply (R, t) to ALL atoms of the original full PDB to generate a "full aligned PDB".

This module is designed to be called from MUSTANG_Alignment.ipynb, but can also be used as a script-like API.
"""

from __future__ import annotations

import csv
import os
import pickle
import subprocess
from dataclasses import dataclass
from glob import glob
from typing import Iterable, Literal

import numpy as np
from Bio.PDB import PDBIO, PDBParser, PPBuilder, Select


ChainPolicy = Literal["first_chain_first_model"]


@dataclass(frozen=True)
class SegmentInfo:
    pdb_path: str
    full_sequence: str
    dfg_idx: int
    ape_idx: int
    # inclusive python slices (start, end) on sequence indices
    seg1_range: tuple[int, int]  # DFG-flank ... DFG+3
    seg2_range: tuple[int, int]  # APE ... APE+3+flank
    n_selected_residues: int


def _first_model_first_chain(structure, pdb_path: str):
    try:
        model = next(structure.get_models())
    except StopIteration as e:
        raise RuntimeError(f"No models found in {pdb_path}") from e
    try:
        chain = next(model.get_chains())
    except StopIteration as e:
        raise RuntimeError(f"No chains found in {pdb_path}") from e
    return model, chain


def extract_chain_sequence_from_pdb(
    pdb_path: str,
    *,
    chain_policy: ChainPolicy = "first_chain_first_model",
) -> str:
    """
    Extract the amino-acid sequence for the chain used by our pipeline (first model + first chain).
    """
    if chain_policy != "first_chain_first_model":
        raise ValueError(f"Unsupported chain_policy: {chain_policy}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    _, chain = _first_model_first_chain(structure, pdb_path)

    ppb = PPBuilder()
    peptides = list(ppb.build_peptides(chain))
    if not peptides:
        raise RuntimeError(f"No polypeptide built from the first chain in {pdb_path}")

    seq_chars: list[str] = []
    for pp in peptides:
        seq_chars.extend(list(str(pp.get_sequence())))
    return "".join(seq_chars)


def extract_flanking_segments_excluding_aloop(
    input_pdb: str,
    output_pdb: str,
    flank: int = 40,
    *,
    dfg_motif: str = "DFG",
    ape_motif: str = "APE",
    chain_policy: ChainPolicy = "first_chain_first_model",
) -> SegmentInfo:
    """
    Extract two flanking segments (as ONE PDB) while excluding the activation loop between DFG and APE.

    Segment N: [DFG-flank : DFG+3)
    Segment C: [APE : APE+3+flank)

    Selection is done on the first model and first chain (consistent with existing notebooks).
    """
    if chain_policy != "first_chain_first_model":
        raise ValueError(f"Unsupported chain_policy: {chain_policy}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", input_pdb)
    model, chain = _first_model_first_chain(structure, input_pdb)

    ppb = PPBuilder()
    peptides = list(ppb.build_peptides(chain))
    if not peptides:
        raise RuntimeError(f"No polypeptide built from first chain in {input_pdb}")

    # Concatenate full sequence and residue objects in sequence order
    seq_chars: list[str] = []
    residue_list = []  # residue per AA char, in order
    for pp in peptides:
        seq_str = str(pp.get_sequence())
        residues = list(pp)  # residues aligning to seq_str
        for aa, res in zip(seq_str, residues):
            seq_chars.append(aa)
            residue_list.append(res)
    seq = "".join(seq_chars)

    dfg_idx = seq.find(dfg_motif)
    ape_idx = seq.find(ape_motif)
    if dfg_idx == -1 or ape_idx == -1 or ape_idx <= dfg_idx:
        raise ValueError(
            f"Cannot find ordered '{dfg_motif}' ... '{ape_motif}' in sequence from {input_pdb}."
        )

    seg1_start = max(0, dfg_idx - flank)
    seg1_end = min(len(seq), dfg_idx + len(dfg_motif))  # include DFG (3)

    seg2_start = ape_idx
    seg2_end = min(len(seq), ape_idx + len(ape_motif) + flank)  # include APE (3)

    seg_residues = set()
    # Segment 1: DFG-flank → DFG
    for i in range(seg1_start, seg1_end):
        seg_residues.add(residue_list[i])
    # Segment 2: APE → APE+flank
    for i in range(seg2_start, seg2_end):
        seg_residues.add(residue_list[i])

    class SegmentSelect(Select):
        def accept_residue(self, residue):
            return residue in seg_residues

    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, SegmentSelect())

    return SegmentInfo(
        pdb_path=output_pdb,
        full_sequence=seq,
        dfg_idx=dfg_idx,
        ape_idx=ape_idx,
        seg1_range=(seg1_start, seg1_end),
        seg2_range=(seg2_start, seg2_end),
        n_selected_residues=len(seg_residues),
    )


def run_mustang_pairwise(
    template_seg_pdb: str,
    target_seg_pdb: str,
    output_prefix: str,
    *,
    mustang_exe: str = "mustang",
) -> dict:
    """
    Run MUSTANG on (template_seg_pdb, target_seg_pdb).

    MUSTANG will write multiple files with base name output_prefix, including:
    - {output_prefix}.pdb      (superposed structures, commonly with chain A=template, B=target)
    - {output_prefix}.rms_rot  (RMSD table)
    - {output_prefix}.afasta   (sequence alignment in FASTA-like format)
    """
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    cmd = [
        mustang_exe,
        "-i",
        template_seg_pdb,
        target_seg_pdb,
        "-o",
        output_prefix,
        "-F",
        "fasta",
        "-s",
        "ON",
        "-r",
        "ON",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"MUSTANG failed (exit={result.returncode}): {result.stderr}")

    rms_file = output_prefix + ".rms_rot"
    rmsd_value = None
    if os.path.exists(rms_file):
        with open(rms_file, "r") as f:
            for line in f:
                if line.strip().startswith("2|"):
                    parts = line.strip().split("|")[-1].split()
                    for p in parts:
                        try:
                            rmsd_value = float(p)
                            break
                        except ValueError:
                            continue
                    break

    aligned_pdb = output_prefix + ".pdb"
    produced = []
    for ext in (".pdb", ".rms_rot", ".afasta", ".html", ".msa"):
        fp = output_prefix + ext
        if os.path.exists(fp):
            produced.append(fp)

    return {
        "cmd": cmd,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "rmsd_value": rmsd_value,
        "rms_file": rms_file if os.path.exists(rms_file) else None,
        "aligned_pdb": aligned_pdb if os.path.exists(aligned_pdb) else None,
        "produced_files": produced,
    }


def _ca_coords_from_pdb(
    pdb_path: str,
    *,
    chain_id: str | None = None,
) -> np.ndarray:
    """
    Extract CA coordinates in residue/atom iteration order.
    If chain_id is provided, only that chain is used (first model).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    model, chain0 = _first_model_first_chain(structure, pdb_path)
    chain = chain0 if chain_id is None else model[chain_id]

    coords: list[list[float]] = []
    for residue in chain:
        if "CA" in residue:
            atom = residue["CA"]
            coords.append([float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2])])
    return np.asarray(coords, dtype=float)


def kabsch_rigid_transform(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transform (R, t) that maps P -> Q in least squares sense:
        Q ≈ (R @ P.T).T + t
    Returns:
        R: (3,3)
        t: (3,)
    """
    if P.shape != Q.shape:
        raise ValueError(f"P and Q must have same shape, got {P.shape} vs {Q.shape}")
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got {P.shape}")
    if P.shape[0] < 3:
        raise ValueError(f"Need at least 3 points for stable rigid transform, got N={P.shape[0]}")

    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ

    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cQ - (R @ cP)
    return R, t


def apply_rigid_transform_to_pdb(
    input_pdb: str,
    output_pdb: str,
    R: np.ndarray,
    t: np.ndarray,
    *,
    chain_id: str | None = None,
) -> None:
    """
    Apply (R, t) to ALL atoms in the PDB (optionally restricted to one chain) and write output_pdb.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", input_pdb)
    model, _ = _first_model_first_chain(structure, input_pdb)

    chains: Iterable = model.get_chains() if chain_id is None else [model[chain_id]]
    for chain in chains:
        for residue in chain:
            for atom in residue.get_atoms():
                x = np.asarray(atom.coord, dtype=float)
                atom.coord = (R @ x) + t

    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Root mean square deviation between two point sets (N,3)."""
    if P.shape != Q.shape or P.shape[0] == 0:
        return float("nan")
    d = P - Q
    return float(np.sqrt((d * d).sum(axis=1).mean()))


def align_full_pdb_via_flanks(
    template_full_pdb: str,
    target_full_pdb: str,
    work_dir: str,
    *,
    flank: int = 40,
    mustang_exe: str = "mustang",
    mustang_target_chain_id: str = "B",
) -> dict:
    """
    End-to-end for a single structure:
      1) extract flanks from template and target (excluding A-loop)
      2) MUSTANG align flanks
      3) compute rigid transform from (target_flanks_before) -> (target_flanks_after)
      4) apply transform to full target PDB (all atoms)
      5) return paths + (R,t) + sanity RMSD
    """
    os.makedirs(work_dir, exist_ok=True)
    template_seg = os.path.join(work_dir, "template_flanks.pdb")
    target_seg = os.path.join(work_dir, "target_flanks.pdb")

    extract_flanking_segments_excluding_aloop(template_full_pdb, template_seg, flank=flank)
    extract_flanking_segments_excluding_aloop(target_full_pdb, target_seg, flank=flank)

    output_prefix = os.path.join(work_dir, "mustang_flanks")
    mres = run_mustang_pairwise(template_seg, target_seg, output_prefix, mustang_exe=mustang_exe)
    aligned_combined_pdb = mres["aligned_pdb"]
    if not aligned_combined_pdb or not os.path.exists(aligned_combined_pdb):
        raise RuntimeError("MUSTANG did not produce an aligned .pdb output as expected.")

    P = _ca_coords_from_pdb(target_seg, chain_id=None)
    Q = _ca_coords_from_pdb(aligned_combined_pdb, chain_id=mustang_target_chain_id)
    if P.shape != Q.shape:
        raise RuntimeError(f"CA count mismatch for transform: before={P.shape} after={Q.shape}")

    R, t = kabsch_rigid_transform(P, Q)
    P_mapped = (R @ P.T).T + t
    seg_rmsd = rmsd(P_mapped, Q)

    full_aligned_pdb = os.path.join(work_dir, "full_aligned.pdb")
    apply_rigid_transform_to_pdb(target_full_pdb, full_aligned_pdb, R, t)

    return {
        "template_seg_pdb": template_seg,
        "target_seg_pdb": target_seg,
        "mustang_output_prefix": output_prefix,
        "mustang_aligned_pdb": aligned_combined_pdb,
        "mustang_rmsd": mres.get("rmsd_value"),
        "mustang_target_chain_id": mustang_target_chain_id,
        "n_points": int(P.shape[0]),
        "R": R,
        "t": t,
        "segment_rmsd_after_transform": seg_rmsd,
        "full_aligned_pdb": full_aligned_pdb,
        "produced_files": mres.get("produced_files", []),
    }


def batch_align_full_pdbs_via_flanks(
    *,
    template_full_pdb: str,
    targets_dir: str,
    out_full_aligned_dir: str,
    transforms_pkl: str,
    transforms_csv: str | None = None,
    flank: int = 40,
    mustang_exe: str = "mustang",
    mustang_target_chain_id: str = "B",
    glob_pattern: str = "*.pdb",
) -> dict:
    """
    Batch process:
      - For each target PDB in targets_dir, compute (R,t) from flanks and write full aligned PDB to out_full_aligned_dir.
      - Save transforms log to transforms_pkl and (optional) transforms_csv.
      - Skip + log failures for unstable/missing cases.

    Returns a summary dict.
    """
    os.makedirs(out_full_aligned_dir, exist_ok=True)
    os.makedirs(os.path.dirname(transforms_pkl), exist_ok=True)
    if transforms_csv:
        os.makedirs(os.path.dirname(transforms_csv), exist_ok=True)

    targets = sorted(glob(os.path.join(targets_dir, glob_pattern)))
    transforms: dict[str, dict] = {}
    failures: list[dict] = []

    for target_full_pdb in targets:
        name = os.path.splitext(os.path.basename(target_full_pdb))[0]
        work_dir = os.path.join(out_full_aligned_dir, "_work", name)
        try:
            res = align_full_pdb_via_flanks(
                template_full_pdb=template_full_pdb,
                target_full_pdb=target_full_pdb,
                work_dir=work_dir,
                flank=flank,
                mustang_exe=mustang_exe,
                mustang_target_chain_id=mustang_target_chain_id,
            )
            out_pdb = os.path.join(out_full_aligned_dir, f"{name}.pdb")
            # Move/copy produced full aligned pdb into the canonical output folder
            apply_rigid_transform_to_pdb(target_full_pdb, out_pdb, res["R"], res["t"])

            transforms[name] = {
                "name": name,
                "input_full_pdb": target_full_pdb,
                "output_full_aligned_pdb": out_pdb,
                "R": res["R"],
                "t": res["t"],
                "n_points": int(_ca_coords_from_pdb(res["target_seg_pdb"]).shape[0]),
                "segment_rmsd_after_transform": float(res["segment_rmsd_after_transform"]),
                "mustang_rmsd": res.get("mustang_rmsd"),
                "mustang_output_prefix": res["mustang_output_prefix"],
            }
        except Exception as e:
            failures.append({"name": name, "input_full_pdb": target_full_pdb, "reason": str(e)})
            continue

    with open(transforms_pkl, "wb") as f:
        pickle.dump({"transforms": transforms, "failures": failures}, f)

    if transforms_csv:
        with open(transforms_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "name",
                    "input_full_pdb",
                    "output_full_aligned_pdb",
                    "n_points",
                    "segment_rmsd_after_transform",
                    "mustang_rmsd",
                    "t_x",
                    "t_y",
                    "t_z",
                    "R_00",
                    "R_01",
                    "R_02",
                    "R_10",
                    "R_11",
                    "R_12",
                    "R_20",
                    "R_21",
                    "R_22",
                ]
            )
            for name, rec in transforms.items():
                R = rec["R"].reshape(3, 3)
                t = rec["t"].reshape(3)
                w.writerow(
                    [
                        name,
                        rec["input_full_pdb"],
                        rec["output_full_aligned_pdb"],
                        rec["n_points"],
                        rec["segment_rmsd_after_transform"],
                        rec.get("mustang_rmsd"),
                        float(t[0]),
                        float(t[1]),
                        float(t[2]),
                        float(R[0, 0]),
                        float(R[0, 1]),
                        float(R[0, 2]),
                        float(R[1, 0]),
                        float(R[1, 1]),
                        float(R[1, 2]),
                        float(R[2, 0]),
                        float(R[2, 1]),
                        float(R[2, 2]),
                    ]
                )

        # also dump failures alongside
        fail_csv = os.path.splitext(transforms_csv)[0] + "_failures.csv"
        with open(fail_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "input_full_pdb", "reason"])
            w.writeheader()
            w.writerows(failures)

    return {
        "n_targets": len(targets),
        "n_success": len(transforms),
        "n_fail": len(failures),
        "out_full_aligned_dir": out_full_aligned_dir,
        "transforms_pkl": transforms_pkl,
        "transforms_csv": transforms_csv,
    }

