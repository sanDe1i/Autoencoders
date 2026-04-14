from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:  # pragma: no cover
    PchipInterpolator = None


@dataclass
class AlignmentLike:
    name: str
    seq1: str
    seq2: str


@dataclass
class AtomRecord:
    serial: int
    name: str
    altloc: str
    resn: str
    chain: str
    resi: int
    icode: str
    coord: np.ndarray
    occupancy: float | None
    line: str


@dataclass
class GeometryMetrics:
    pdb_id: str
    n_ca: int
    min_adjacent_distance: float
    max_adjacent_distance: float
    duplicate_count: int
    jump_count: int
    geometry_ok: bool
    failure_reasons: str

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return {
            "pdb_id": self.pdb_id,
            "n_ca": self.n_ca,
            "min_adjacent_distance": self.min_adjacent_distance,
            "max_adjacent_distance": self.max_adjacent_distance,
            "duplicate_count": self.duplicate_count,
            "jump_count": self.jump_count,
            "geometry_ok": self.geometry_ok,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class MotifSelection:
    pdb_id: str
    seq1_dfg_positions: list[int]
    seq1_ape_positions: list[int]
    seq2_dfg_positions: list[int]
    seq2_ape_positions: list[int]
    legacy_dfg_index: int
    legacy_ape_index: int
    selected_dfg_index: int | None
    selected_ape_index: int | None
    selected_residue_count: int | None
    selected_alignment_span: int | None
    status: str
    failure_reason: str
    warning_reason: str
    length_in_range: bool | None
    candidate_summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "pdb_id": self.pdb_id,
            "seq1_dfg_positions": _list_to_string(self.seq1_dfg_positions),
            "seq1_ape_positions": _list_to_string(self.seq1_ape_positions),
            "seq2_dfg_positions": _list_to_string(self.seq2_dfg_positions),
            "seq2_ape_positions": _list_to_string(self.seq2_ape_positions),
            "legacy_dfg_index": self.legacy_dfg_index,
            "legacy_ape_index": self.legacy_ape_index,
            "selected_dfg_index": self.selected_dfg_index if self.selected_dfg_index is not None else "",
            "selected_ape_index": self.selected_ape_index if self.selected_ape_index is not None else "",
            "selected_residue_count": self.selected_residue_count if self.selected_residue_count is not None else "",
            "selected_alignment_span": self.selected_alignment_span if self.selected_alignment_span is not None else "",
            "status": self.status,
            "failure_reason": self.failure_reason,
            "warning_reason": self.warning_reason,
            "length_in_range": self.length_in_range if self.length_in_range is not None else "",
            "candidate_summary": self.candidate_summary,
        }


def _safe_float(text: str) -> float | None:
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _list_to_string(values: list[int]) -> str:
    return "|".join(str(v) for v in values)


def find_motif_positions(seq: str, motif: str) -> list[int]:
    return [match.start() for match in re.finditer(motif, seq)]


def ungapped_length(seq: str) -> int:
    return sum(1 for ch in seq if ch != "-")


def choose_dfg_ape_pair(
    align_obj: AlignmentLike,
    expected_residue_count: int = 27,
    min_residue_count: int = 14,
    max_residue_count: int = 40,
) -> MotifSelection:
    seq1 = align_obj.seq1
    seq2 = align_obj.seq2
    seq1_dfg_positions = find_motif_positions(seq1, "DFG")
    seq1_ape_positions = find_motif_positions(seq1, "APE")
    seq2_dfg_positions = find_motif_positions(seq2, "DFG")
    seq2_ape_positions = find_motif_positions(seq2, "APE")

    legacy_dfg_index = seq1.find("DFG")
    legacy_ape_index = seq1.find("APE")

    candidates: list[dict[str, int | bool]] = []
    for dfg_index in seq2_dfg_positions:
        for ape_index in seq2_ape_positions:
            if ape_index <= dfg_index:
                continue
            residue_count = ungapped_length(seq2[dfg_index : ape_index + 3])
            candidates.append(
                {
                    "dfg_index": dfg_index,
                    "ape_index": ape_index,
                    "residue_count": residue_count,
                    "alignment_span": ape_index - dfg_index + 3,
                    "in_range": min_residue_count <= residue_count <= max_residue_count,
                }
            )

    candidate_summary = "; ".join(
        f"{cand['dfg_index']}->{cand['ape_index']}|len={cand['residue_count']}|span={cand['alignment_span']}|ok={int(bool(cand['in_range']))}"
        for cand in candidates
    )

    if not seq2_dfg_positions:
        return MotifSelection(
            pdb_id=align_obj.name,
            seq1_dfg_positions=seq1_dfg_positions,
            seq1_ape_positions=seq1_ape_positions,
            seq2_dfg_positions=seq2_dfg_positions,
            seq2_ape_positions=seq2_ape_positions,
            legacy_dfg_index=legacy_dfg_index,
            legacy_ape_index=legacy_ape_index,
            selected_dfg_index=None,
            selected_ape_index=None,
            selected_residue_count=None,
            selected_alignment_span=None,
            status="failed",
            failure_reason="no_exact_seq2_dfg",
            warning_reason="",
            length_in_range=None,
            candidate_summary=candidate_summary,
        )

    if not seq2_ape_positions:
        return MotifSelection(
            pdb_id=align_obj.name,
            seq1_dfg_positions=seq1_dfg_positions,
            seq1_ape_positions=seq1_ape_positions,
            seq2_dfg_positions=seq2_dfg_positions,
            seq2_ape_positions=seq2_ape_positions,
            legacy_dfg_index=legacy_dfg_index,
            legacy_ape_index=legacy_ape_index,
            selected_dfg_index=None,
            selected_ape_index=None,
            selected_residue_count=None,
            selected_alignment_span=None,
            status="failed",
            failure_reason="no_exact_seq2_ape",
            warning_reason="",
            length_in_range=None,
            candidate_summary=candidate_summary,
        )

    if not candidates:
        return MotifSelection(
            pdb_id=align_obj.name,
            seq1_dfg_positions=seq1_dfg_positions,
            seq1_ape_positions=seq1_ape_positions,
            seq2_dfg_positions=seq2_dfg_positions,
            seq2_ape_positions=seq2_ape_positions,
            legacy_dfg_index=legacy_dfg_index,
            legacy_ape_index=legacy_ape_index,
            selected_dfg_index=None,
            selected_ape_index=None,
            selected_residue_count=None,
            selected_alignment_span=None,
            status="failed",
            failure_reason="no_ape_after_dfg",
            warning_reason="",
            length_in_range=None,
            candidate_summary=candidate_summary,
        )

    selected = min(
        candidates,
        key=lambda cand: (
            abs(int(cand["residue_count"]) - expected_residue_count),
            abs(int(cand["residue_count"]) - expected_residue_count) > 0,
            int(cand["alignment_span"]),
            int(cand["dfg_index"]),
            int(cand["ape_index"]),
        ),
    )

    length_in_range = bool(selected["in_range"])
    warning_reason = "" if length_in_range else "segment_length_out_of_range"
    return MotifSelection(
        pdb_id=align_obj.name,
        seq1_dfg_positions=seq1_dfg_positions,
        seq1_ape_positions=seq1_ape_positions,
        seq2_dfg_positions=seq2_dfg_positions,
        seq2_ape_positions=seq2_ape_positions,
        legacy_dfg_index=legacy_dfg_index,
        legacy_ape_index=legacy_ape_index,
        selected_dfg_index=int(selected["dfg_index"]),
        selected_ape_index=int(selected["ape_index"]),
        selected_residue_count=int(selected["residue_count"]),
        selected_alignment_span=int(selected["alignment_span"]),
        status="selected",
        failure_reason="",
        warning_reason=warning_reason,
        length_in_range=length_in_range,
        candidate_summary=candidate_summary,
    )


def filter_alignments_by_motif_safe(
    alignments: Iterable[AlignmentLike],
    motif_log_path: str | Path | None = None,
    expected_residue_count: int = 27,
    min_residue_count: int = 14,
    max_residue_count: int = 40,
) -> tuple[list[AlignmentLike], list[MotifSelection]]:
    selected_alignments: list[AlignmentLike] = []
    motif_rows: list[MotifSelection] = []

    for align_obj in alignments:
        selection = choose_dfg_ape_pair(
            align_obj,
            expected_residue_count=expected_residue_count,
            min_residue_count=min_residue_count,
            max_residue_count=max_residue_count,
        )
        motif_rows.append(selection)
        if selection.status == "selected":
            selected_alignments.append(align_obj)

    if motif_log_path is not None:
        write_dict_rows(
            motif_log_path,
            [row.to_dict() for row in motif_rows],
            [
                "pdb_id",
                "seq1_dfg_positions",
                "seq1_ape_positions",
                "seq2_dfg_positions",
                "seq2_ape_positions",
                "legacy_dfg_index",
                "legacy_ape_index",
                "selected_dfg_index",
                "selected_ape_index",
                "selected_residue_count",
                "selected_alignment_span",
                "status",
                "failure_reason",
                "warning_reason",
                "length_in_range",
                "candidate_summary",
            ],
        )

    return selected_alignments, motif_rows


def parse_pdb_atoms(pdb_path: str | Path) -> list[AtomRecord]:
    atoms: list[AtomRecord] = []
    with Path(pdb_path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atoms.append(
                AtomRecord(
                    serial=int(line[6:11]),
                    name=line[12:16].strip(),
                    altloc=line[16].strip(),
                    resn=line[17:20].strip(),
                    chain=(line[21] or "A").strip() or "A",
                    resi=int(line[22:26]),
                    icode=line[26].strip(),
                    coord=np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    ),
                    occupancy=_safe_float(line[54:60]),
                    line=line.rstrip("\n"),
                )
            )
    return atoms


def _residue_key(atom: AtomRecord) -> tuple[str, int, str, str]:
    return (atom.chain, atom.resi, atom.icode, atom.resn)


def ordered_residue_groups(atoms: Iterable[AtomRecord]) -> list[tuple[tuple[str, int, str, str], list[AtomRecord]]]:
    groups: list[tuple[tuple[str, int, str, str], list[AtomRecord]]] = []
    index: dict[tuple[str, int, str, str], int] = {}
    for atom in atoms:
        key = _residue_key(atom)
        if key not in index:
            index[key] = len(groups)
            groups.append((key, []))
        groups[index[key]][1].append(atom)
    return groups


def pick_altloc_atom(atoms: list[AtomRecord], prefer_altloc: str = "A") -> AtomRecord:
    preferred = [atom for atom in atoms if atom.altloc == prefer_altloc]
    if preferred:
        return preferred[0]

    with_occ = [atom for atom in atoms if atom.occupancy is not None]
    if with_occ:
        return max(with_occ, key=lambda atom: (atom.occupancy, -atom.serial))

    return atoms[0]


def extract_ca_segment_safe(
    pdb_path: str | Path,
    start_residue_index: int,
    end_residue_index: int,
    prefer_altloc: str = "A",
) -> list[AtomRecord]:
    atoms = parse_pdb_atoms(pdb_path)
    residue_groups = ordered_residue_groups(atoms)
    selected_groups = residue_groups[start_residue_index:end_residue_index]
    picked_ca: list[AtomRecord] = []
    for _, residue_atoms in selected_groups:
        ca_atoms = [atom for atom in residue_atoms if atom.name == "CA"]
        if not ca_atoms:
            continue
        picked_ca.append(pick_altloc_atom(ca_atoms, prefer_altloc=prefer_altloc))
    return picked_ca


def save_ca_records(records: list[AtomRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["MODEL        0"]
    for atom in records:
        x, y, z = atom.coord
        occupancy = 1.00 if atom.occupancy is None else atom.occupancy
        lines.append(
            f"ATOM  {atom.serial:5d}  CA  {atom.resn:>3s} {atom.chain}{atom.resi:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  {occupancy:4.2f}  0.00      {atom.chain:>1s}    C  "
        )
    if records:
        last = records[-1]
        lines.append(f"TER   {last.serial + 1:5d}      {last.resn:>3s} {last.chain}{last.resi:4d}")
    lines.extend(["ENDMDL", "END"])
    out_path.write_text("\n".join(lines) + "\n")


def strip_to_ca_safe(
    pdb_path: str | Path,
    start_residue: int,
    end_residue: int,
    prefer_altloc: str = "A",
) -> list[AtomRecord]:
    return extract_ca_segment_safe(
        pdb_path,
        start_residue_index=start_residue,
        end_residue_index=end_residue,
        prefer_altloc=prefer_altloc,
    )


def process_alignments_safe(
    pdb_dir: str | Path,
    target_dir: str | Path,
    alignments: Iterable[AlignmentLike],
    prefer_altloc: str = "A",
    extraction_log_path: str | Path | None = None,
    expected_residue_count: int = 27,
    min_residue_count: int = 14,
    max_residue_count: int = 40,
) -> list[dict[str, str | int]]:
    pdb_dir = Path(pdb_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    extraction_log: list[dict[str, str | int]] = []

    for align_obj in alignments:
        matching_fp = None
        for fp in pdb_files:
            if align_obj.name in fp.name:
                matching_fp = fp
                break

        if matching_fp is None:
            extraction_log.append({"pdb_id": align_obj.name, "status": "missing_source", "start_index": -1, "end_index": -1, "ca_count": 0})
            continue

        motif_selection = choose_dfg_ape_pair(
            align_obj,
            expected_residue_count=expected_residue_count,
            min_residue_count=min_residue_count,
            max_residue_count=max_residue_count,
        )
        if motif_selection.status != "selected":
            extraction_log.append(
                {
                    "pdb_id": align_obj.name,
                    "status": motif_selection.status,
                    "start_index": -1,
                    "end_index": -1,
                    "ca_count": 0,
                    "failure_reason": motif_selection.failure_reason,
                    "warning_reason": motif_selection.warning_reason,
                    "length_in_range": motif_selection.length_in_range if motif_selection.length_in_range is not None else "",
                    "selected_dfg_index": motif_selection.selected_dfg_index if motif_selection.selected_dfg_index is not None else "",
                    "selected_ape_index": motif_selection.selected_ape_index if motif_selection.selected_ape_index is not None else "",
                    "selected_residue_count": motif_selection.selected_residue_count if motif_selection.selected_residue_count is not None else "",
                    "candidate_summary": motif_selection.candidate_summary,
                }
            )
            continue

        dfg_index = int(motif_selection.selected_dfg_index)
        ape_index = int(motif_selection.selected_ape_index)
        dfg_index_adjusted = dfg_index - sum(1 for ch in align_obj.seq2[:dfg_index] if ch == "-")
        ape_index_adjusted = ape_index - sum(1 for ch in align_obj.seq2[:ape_index] if ch == "-")

        records = strip_to_ca_safe(matching_fp, dfg_index_adjusted, ape_index_adjusted + 3, prefer_altloc=prefer_altloc)
        save_ca_records(records, target_dir / matching_fp.name)
        extraction_log.append(
            {
                "pdb_id": align_obj.name,
                "status": "saved",
                "start_index": dfg_index_adjusted,
                "end_index": ape_index_adjusted + 3,
                "ca_count": len(records),
                "failure_reason": "",
                "warning_reason": motif_selection.warning_reason,
                "length_in_range": motif_selection.length_in_range,
                "selected_dfg_index": dfg_index,
                "selected_ape_index": ape_index,
                "selected_residue_count": motif_selection.selected_residue_count,
                "candidate_summary": motif_selection.candidate_summary,
            }
        )

    if extraction_log_path is not None:
        write_dict_rows(
            extraction_log_path,
            extraction_log,
            [
                "pdb_id",
                "status",
                "start_index",
                "end_index",
                "ca_count",
                "failure_reason",
                "warning_reason",
                "length_in_range",
                "selected_dfg_index",
                "selected_ape_index",
                "selected_residue_count",
                "candidate_summary",
            ],
        )
    return extraction_log


def consecutive_distances(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.zeros(0, dtype=float)
    return np.linalg.norm(np.diff(coords, axis=0), axis=1)


def evaluate_ca_geometry(
    coords: np.ndarray,
    pdb_id: str,
    max_adjacent_threshold: float = 8.0,
    duplicate_threshold: float = 1e-3,
    jump_threshold: float = 8.0,
) -> GeometryMetrics:
    coords = np.asarray(coords, dtype=float)
    dists = consecutive_distances(coords)
    duplicate_count = int(np.sum(dists <= duplicate_threshold))
    jump_count = 0
    reasons: list[str] = []

    if len(dists) and float(np.max(dists)) > max_adjacent_threshold:
        reasons.append(f"adjacent_distance_gt_{max_adjacent_threshold:g}")
    if duplicate_count > 0:
        reasons.append("duplicate_points")

    for i in range(1, len(coords) - 1):
        d_prev = float(np.linalg.norm(coords[i] - coords[i - 1]))
        d_next = float(np.linalg.norm(coords[i + 1] - coords[i]))
        d_skip = float(np.linalg.norm(coords[i + 1] - coords[i - 1]))
        if d_prev > jump_threshold and d_next > jump_threshold and d_skip <= jump_threshold and d_skip <= 0.5 * min(d_prev, d_next):
            jump_count += 1

    if jump_count > 0:
        reasons.append("single_point_jump")

    return GeometryMetrics(
        pdb_id=pdb_id,
        n_ca=len(coords),
        min_adjacent_distance=float(np.min(dists)) if len(dists) else 0.0,
        max_adjacent_distance=float(np.max(dists)) if len(dists) else 0.0,
        duplicate_count=duplicate_count,
        jump_count=jump_count,
        geometry_ok=len(reasons) == 0,
        failure_reasons=";".join(reasons),
    )


def template_ca_lines(template_path: str | Path) -> list[str]:
    lines = []
    with Path(template_path).open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                lines.append(line.rstrip("\n"))
    return lines


def _build_interpolator(t: np.ndarray, values: np.ndarray, interpolation: str):
    method = interpolation.lower()
    if method == "pchip" and PchipInterpolator is not None:
        return PchipInterpolator(t, values, axis=0)

    def linear_interp(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        out = np.zeros((len(query), values.shape[1]), dtype=float)
        for dim in range(values.shape[1]):
            out[:, dim] = np.interp(query, t, values[:, dim])
        return out

    return linear_interp


def resample_ca_trace(
    coords: np.ndarray,
    n_output: int,
    interpolation: str = "pchip",
    dense_points: int = 2048,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 2:
        raise ValueError("Need at least 2 points to resample a CA trace")

    chord = consecutive_distances(coords)
    t = np.concatenate([[0.0], np.cumsum(np.maximum(chord, 1e-6))])
    interpolator = _build_interpolator(t, coords, interpolation=interpolation)
    dense_t = np.linspace(t[0], t[-1], max(dense_points, n_output * 32))
    dense_coords = interpolator(dense_t)
    dense_step = consecutive_distances(dense_coords)
    arc = np.concatenate([[0.0], np.cumsum(dense_step)])
    target_arc = np.linspace(0.0, arc[-1], n_output)

    resampled = np.zeros((n_output, 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_arc, arc, dense_coords[:, dim])
    return resampled


def save_template_with_new_ca_coords(template_path: str | Path, new_coords: np.ndarray, save_path: str | Path) -> None:
    ca_lines = template_ca_lines(template_path)
    if len(ca_lines) != len(new_coords):
        raise ValueError(f"Template CA count ({len(ca_lines)}) does not match fitted coordinate count ({len(new_coords)})")

    out_lines = []
    for line, coord in zip(ca_lines, new_coords):
        x, y, z = coord
        out_lines.append(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
    out_lines.append(f"TER   {len(ca_lines) + 1:5d}      {ca_lines[-1][17:20].strip():>3s} {ca_lines[-1][21]}{int(ca_lines[-1][22:26]):4d}")
    out_lines.append("END")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(out_lines) + "\n")


def write_dict_rows(csv_path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_dict_row(csv_path: str | Path, row: dict[str, object], fieldnames: list[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fitting_code_safe(
    pdb_path: str | Path,
    save_path: str | Path,
    template_path: str | Path = "template.pdb",
    interpolation: str = "pchip",
    qc_log_path: str | Path | None = None,
    max_adjacent_threshold: float = 8.0,
    duplicate_threshold: float = 1e-3,
    jump_threshold: float = 8.0,
) -> tuple[bool, GeometryMetrics]:
    atoms = parse_pdb_atoms(pdb_path)
    coords = np.array([atom.coord for atom in atoms if atom.name == "CA"], dtype=float)
    pdb_id = Path(pdb_path).stem
    metrics = evaluate_ca_geometry(coords, pdb_id, max_adjacent_threshold, duplicate_threshold, jump_threshold)

    if qc_log_path is not None:
        append_dict_row(
            qc_log_path,
            metrics.to_dict(),
            ["pdb_id", "n_ca", "min_adjacent_distance", "max_adjacent_distance", "duplicate_count", "jump_count", "geometry_ok", "failure_reasons"],
        )

    if not metrics.geometry_ok:
        return False, metrics

    n_output = len(template_ca_lines(template_path))
    new_coords = resample_ca_trace(coords, n_output=n_output, interpolation=interpolation)
    save_template_with_new_ca_coords(template_path, new_coords, save_path)
    return True, metrics


def load_geometry_qc(qc_csv_path: str | Path) -> dict[str, dict[str, str]]:
    qc_csv_path = Path(qc_csv_path)
    if not qc_csv_path.exists():
        return {}
    with qc_csv_path.open() as handle:
        return {row["pdb_id"]: row for row in csv.DictReader(handle)}
