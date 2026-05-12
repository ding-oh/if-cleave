"""Paper PROPKA encoding (6D per-residue categorical features).

PROPKA3 produces a `.pka` file with one detailed block per ionizable
residue. IF-Cleave's published model uses a deliberately *categorical*
encoding of that block:

  * non-ionizable residues          -> [7, 0, 0, 0, 0, 0]
  * ionizable, "coupled" (pKa with `*` suffix)
                                    -> [7, 0, 0, 0, 0, 0]   (rejected)
  * ionizable, simple (one .pka line, no `*`)
                                    -> populated columns from the primary
                                       PROPKA line, with desolv_reg /
                                       desolv_eff sign-clamped to >= 0
  * ionizable, complex (>= 2 .pka lines, hbond network)
                                    -> populated from the *last*
                                       continuation line via fixed token
                                       offsets; buried fraction and the
                                       primary desolv columns are cleared

This 3-category encoding is what the paper checkpoints were trained on
and what the CD4 out-of-distribution validation depends on. A "corrected"
parser that captures every numeric field on every line reproduces the
in-distribution test MCC but breaks the OOD generalization, so we keep
the encoding here exactly.

Column layout (cols 512-517 of the 518D feature):
  0: pKa (primary) | sidechain hbond strength (continuation)
  1: buried fraction (primary only)
  2: sidechain hbond partner number (continuation only)
  3: desolv_regular (primary, clamped) | sidechain chain id as float
     (continuation, clamped)
  4: num_volume (primary) | backbone hbond strength (continuation)
  5: desolv_effective (primary, clamped) | partner-name-shifted token
     (continuation; only non-zero when the 3-digit residue-number
     field overflows into the partner-name token)
"""
from __future__ import annotations

import re

import numpy as np


IONIZABLE = {"ASP", "GLU", "HIS", "TYR", "LYS", "ARG", "CYS"}

PROPKA_DEFAULT = np.array([7.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
N_PROPKA_COLS = 6


# Primary line: full PROPKA header with BURIED %.
# - pKa must be plain numeric — values flagged as "coupled" (suffixed `*`)
#   are intentionally rejected so the residue falls back to PROPKA_DEFAULT.
# - desolv_reg / desolv_eff are captured permissively and clamped to
#   non-negative below.
_PRIMARY_RE = re.compile(
    r"^([A-Z][A-Z0-9+\-]{1,2})\s+(\d+)\s+([A-Za-z0-9])\s+"
    r"(-?\d+\.\d+)\s+(\d+)\s*%\s+"
    r"(\S+)\s+(\d+)\s+"
    r"(\S+)\s+(\d+)"
)

# Continuation lines have NO BURIED % field. We only need to recognise the
# residue header — the column values are read by fixed token offsets.
_CONTINUATION_HEADER_RE = re.compile(
    r"^([A-Z][A-Z0-9+\-]{1,2})\s+(\d+)\s+([A-Za-z0-9])\s"
)


def _safe_float(token: str) -> float:
    try:
        return float(token.rstrip("*"))
    except ValueError:
        return 0.0


def _safe_float_nonneg(token: str) -> float:
    """Parse `token` and clamp to non-negative.

    Strict `> 0` (not `>= 0`) so a parsed `-0.0` collapses to positive
    zero — otherwise the float32 byte representation would still carry
    the sign bit.
    """
    try:
        v = float(token.rstrip("*"))
    except ValueError:
        return 0.0
    return v if v > 0.0 else 0.0


def parse_propka_file(pka_path: str, chain_id: str) -> dict[tuple[str, int], np.ndarray]:
    """Return {(res_name, res_num) -> 6D float32 array} for an ionizable chain."""
    rows: dict[tuple[str, int], np.ndarray] = {}
    in_block = False
    with open(pka_path) as fh:
        for line in fh:
            if "RESIDUE    pKa    BURIED" in line:
                in_block = True
                continue
            if not in_block:
                continue
            if "SUMMARY" in line.upper():
                break

            m = _PRIMARY_RE.match(line)
            if m:
                res, num, ch = m.group(1), int(m.group(2)), m.group(3)
                if ch != chain_id or res not in IONIZABLE:
                    continue
                pka = _safe_float(m.group(4))
                buried = float(m.group(5)) / 100.0
                desolv_reg = _safe_float_nonneg(m.group(6))
                num_vol = float(m.group(7))
                desolv_eff = _safe_float_nonneg(m.group(8))
                rows[(res, num)] = np.array(
                    [pka, buried, 0.0, desolv_reg, num_vol, desolv_eff],
                    dtype=np.float32,
                )
                continue

            # A header line with a BURIED% column that *didn't* match
            # _PRIMARY_RE was rejected by the `*`-suffix filter. It must
            # not fall through to the continuation parser.
            if "%" in line[:32]:
                continue

            m = _CONTINUATION_HEADER_RE.match(line)
            if not m:
                continue
            res, num, ch = m.group(1), int(m.group(2)), m.group(3)
            if ch != chain_id or res not in IONIZABLE:
                continue
            parts = line.split()
            if len(parts) <= 8:
                continue
            rows[(res, num)] = np.array(
                [
                    _safe_float(parts[3]),       # col 0: sc hbond val
                    0.0,                         # col 1: buried (cleared)
                    _safe_float(parts[5]),       # col 2: sc partner num
                    _safe_float_nonneg(parts[6]),  # col 3: sc chain digit
                    _safe_float(parts[7]),       # col 4: bb hbond val
                    _safe_float(parts[8]),       # col 5: shifted partner
                ],
                dtype=np.float32,
            )
    return rows


def chain_residue_order(pdb_path: str, chain_id: str) -> list[tuple[str, int]]:
    """Ordered (resname, resid) for every unique residue on `chain_id`.

    All residues are kept including non-standard / UNK; they will receive
    PROPKA_DEFAULT during alignment, matching the paper feature length.
    """
    seen: list[tuple[str, int]] = []
    seen_set: set[tuple[str, int]] = set()
    last: tuple[str, int] | None = None
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21] != chain_id:
                continue
            resname = line[17:20].strip()
            try:
                resid = int(line[22:26])
            except ValueError:
                continue
            key = (resname, resid)
            if key != last and key not in seen_set:
                seen.append(key)
                seen_set.add(key)
                last = key
    return seen


def build_propka_features(pdb_path: str, pka_path: str, chain_id: str) -> np.ndarray:
    """Build the (N, 6) PROPKA slice aligned to the chain's residue order."""
    residues = chain_residue_order(pdb_path, chain_id)
    rows = parse_propka_file(pka_path, chain_id)
    out = np.tile(PROPKA_DEFAULT, (len(residues), 1)).astype(np.float32)
    for i, key in enumerate(residues):
        if key in rows:
            out[i] = rows[key]
    return out
