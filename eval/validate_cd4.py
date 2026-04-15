#!/usr/bin/env python
"""
CD4+ Epitope Validation Pipeline for GeoCleav

Validates that GeoCleav-predicted cleavage sites are enriched near
experimentally validated CD4+ T cell epitope boundaries.

CD4+ epitopes require MHC-II presentation, which depends on antigen processing
(cleavage). If GeoCleav captures biologically relevant cleavage patterns,
predicted cleavage probabilities should be elevated near epitope N/C-terminal boundaries.

Epitope sources:
  - IEDB MHC-II ligand elution (mass spectrometry) — naturally processed epitopes
  - Boundaries reflect actual protease cleavage sites (not synthetic peptide libraries)
  - Loaded from cd4_validation/iedb_epitopes/*_elution_epitopes.json
  - Use --epitope-type=all for all IEDB epitopes, or hardcoded for original curated set

Targets:
  1. SARS-CoV-2 Spike (P0DTC2, 1273 aa, 1621 elution epitopes)
  2. Influenza H1N1 HA (ABF21272.1, 566 aa, 102 elution epitopes)
  3. RSV Fusion Protein F (P03420, 574 aa, 19 elution epitopes)
  4. Vatreptacog alfa (P08709, 466 aa, 3 elution epitopes)

Usage:
  python validate_cd4_epitopes.py --target all --device cuda
  python validate_cd4_epitopes.py --target spike influenza_ha --device cpu
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import torch

from model.model import BiLSTMGNN

# ===========================================================================
# TARGET DEFINITIONS: sequence + experimentally validated CD4+ epitopes
# ===========================================================================

TARGETS = {}

# ---------------------------------------------------------------------------
# 1. SARS-CoV-2 Spike protein (UniProt P0DTC2, 1273 aa)
# ---------------------------------------------------------------------------
TARGETS["spike"] = {
    "name": "SARS-CoV-2 Spike",
    "accession": "P0DTC2",
    "sequence": (
        "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFS"
        "NVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIV"
        "NNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLE"
        "GKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQT"
        "LLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETK"
        "CTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISN"
        "CVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD"
        "YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPC"
        "NGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVN"
        "FNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITP"
        "GTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSY"
        "ECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTI"
        "SVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQE"
        "VFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC"
        "LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAM"
        "QMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALN"
        "TLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRA"
        "SANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPA"
        "ICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDP"
        "LQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDL"
        "QELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD"
        "SEPVLKGVKLHYT"
    ),
    "epitopes": [
        # Peng et al. 2020 Nature Immunology (Table 1)
        {"seq": "CTFEYVSQPFLMDLE", "start": 166, "source": "Peng2020"},
        {"seq": "YAWNRKRISNCVADY", "start": 351, "source": "Peng2020"},
        {"seq": "GVSPTKLNDLCFTNV", "start": 381, "source": "Peng2020"},
        {"seq": "VVLSFELLHAPATVC", "start": 511, "source": "Peng2020"},
        {"seq": "NLLLQYGSFCTQLNR", "start": 751, "source": "Peng2020"},
        {"seq": "NFSQILPDPSKPSKR", "start": 801, "source": "Peng2020"},
        {"seq": "TDEMIAQYTSALLAG", "start": 866, "source": "Peng2020"},
        # Heide et al. 2022 (PMC9363231, Table 2)
        {"seq": "CEFQFCNDPFLGVYY", "start": 131, "source": "Heide2022"},
        {"seq": "FKIYSKHTPINLVRD", "start": 201, "source": "Heide2022"},
        {"seq": "TRFQTLLALHRSYLT", "start": 236, "source": "Heide2022"},
        {"seq": "GIYQTSNFRVQPTES", "start": 311, "source": "Heide2022"},
        {"seq": "VFNATRFASVYAWNR", "start": 341, "source": "Heide2022"},
        {"seq": "RFASVYAWNRKRISN", "start": 346, "source": "Heide2022"},
        {"seq": "SASFSTFKCYGVSPT", "start": 371, "source": "Heide2022"},
        {"seq": "YLYRLFRKSNLKPFE", "start": 451, "source": "Heide2022"},
        {"seq": "KPSKRSFIEDLLFNK", "start": 811, "source": "Heide2022"},
        {"seq": "SFIEDLLFNKVTLAD", "start": 816, "source": "Heide2022"},
        {"seq": "AGFIKQYGDCLGDIA", "start": 831, "source": "Heide2022"},
        {"seq": "IPFAMQMAYRFNGIG", "start": 896, "source": "Heide2022"},
        # Tarke et al. 2022
        {"seq": "YNYLYRLFRKSNLKP", "start": 449, "source": "Tarke2022"},
    ],
}

TARGETS["spike"]["pdb"] = {
    "id": "6ZGE", "chains": ["A"],
    "path": "cd4_validation/pdb/6ZGE.pdb",
}

# ---------------------------------------------------------------------------
# 1b. Spike NTD domain (X-ray, 7B62 chain A, 1.82 Å)
# ---------------------------------------------------------------------------
TARGETS["spike_ntd"] = {
    "name": "SARS-CoV-2 Spike NTD (X-ray)",
    "accession": "P0DTC2",
    "sequence": TARGETS["spike"]["sequence"],
    "epitopes": list(TARGETS["spike"]["epitopes"]),
    "pdb": {
        "id": "7B62", "chains": ["A"],
        "path": "cd4_validation/pdb/7B62.pdb",
    },
}

# ---------------------------------------------------------------------------
# 1c. Spike RBD domain (X-ray, 6M0J chain E, 2.45 Å)
# ---------------------------------------------------------------------------
TARGETS["spike_rbd"] = {
    "name": "SARS-CoV-2 Spike RBD (X-ray)",
    "accession": "P0DTC2",
    "sequence": TARGETS["spike"]["sequence"],
    "epitopes": list(TARGETS["spike"]["epitopes"]),
    "pdb": {
        "id": "6M0J", "chains": ["E"],
        "path": "cd4_validation/pdb/6M0J.pdb",
    },
}

# ---------------------------------------------------------------------------
# 2. Influenza H1N1 Hemagglutinin (A/New Caledonia/20/1999, GenBank ABF21272)
# ---------------------------------------------------------------------------
TARGETS["influenza_ha"] = {
    "name": "Influenza H1N1 HA",
    "accession": "ABF21272.1",
    "sequence": (
        "MKAKLLVLLCTFTATYADTICIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDSHNGKLCL"
        "LKGIAPLQLGNCSVAGWILGNPECELLISKESWSYIVETPNPENGTCYPGYFADYEELR"
        "EQLSSVSSFERFEIFPKESSWNPHTVTGVSASCSHNGKSSFYRNLLWLTGKNGLYPNLS"
        "KSYVNNKEKEVLVLWGVHHPPNIGNQRALYHTENAYVSVVSSHYSRRFTPEIAKRPKVR"
        "DQEGRINYYWTLLEPGDTIIFEANGNLIAPWYAFALSRGFGSGIIITSNAPMDECDAKCQ"
        "TPQGAINSSLPFQNVHPVTIGECPKYVRSAKLRMVTGLRNIPSIQSRGLFGAIAGFIEGG"
        "WTGMVDGWYGYHHQNEQGSGYAADQKSTQNAINGITNKVNSVIEKMNTQFTAVGKEFNK"
        "LERRMENLNKKVDDGFLDIWTYNAELLVLLENERTLDFHDSNVKNLYEKVKSQLKNNAK"
        "EIGNGCFEFYHKCNNECMESVKNGTYDYPKYSEESKLNREKIDGVKLESMGVYQILAIYS"
        "TVASSLVLLVSLGAISFWMCSNGSLQCRICI"
    ),
    # Richards et al. 2020 J Exp Med (PMC6963931, Table 1)
    # 17-mer peptides, IL-2 ELISpot validated, multiple mouse strains + HLA-DR1/DQ8
    "epitopes": [
        {"seq": "EQLSSVSSFERFEIFPK", "start": 120, "source": "Richards2020"},
        {"seq": "SSFERFEIFPKESSWNP", "start": 126, "source": "Richards2020"},
        {"seq": "TVTGVSASCSHNGKSSF", "start": 144, "source": "Richards2020"},
        {"seq": "GKSSFYRNLLWLTGKNG", "start": 156, "source": "Richards2020"},
        {"seq": "RNLLWLTGKNGLYPNLS", "start": 162, "source": "Richards2020"},
        {"seq": "YPNLSKSYVNNKEKEVL", "start": 174, "source": "Richards2020"},
        {"seq": "NQRALYHTENAYVSVVS", "start": 203, "source": "Richards2020"},
        {"seq": "VSVVSSHYSRRFTPEIA", "start": 215, "source": "Richards2020"},
        {"seq": "GIIITSNAPMDECDAKC", "start": 280, "source": "Richards2020"},
        {"seq": "IGECPKYVRSAKLRMVT", "start": 316, "source": "Richards2020"},
        {"seq": "YVRSAKLRMVTGLRNIP", "start": 322, "source": "Richards2020"},
        {"seq": "LRMVTGLRNIPSIQSRG", "start": 328, "source": "Richards2020"},
        {"seq": "LRNIPSIQSRGLFGAIA", "start": 334, "source": "Richards2020"},
        {"seq": "TGMVDGWYGYHHQNEQG", "start": 358, "source": "Richards2020"},
        {"seq": "SGYAADQKSTQNAINGI", "start": 375, "source": "Richards2020"},
        {"seq": "NAINGITNKVNSVIEKM", "start": 386, "source": "Richards2020"},
        {"seq": "VIEKMNTQFTAVGKEFN", "start": 398, "source": "Richards2020"},
        {"seq": "IWTYNAELLVLLENERT", "start": 434, "source": "Richards2020"},
        {"seq": "ELLVLLENERTLDFHDS", "start": 440, "source": "Richards2020"},
    ],
}

TARGETS["influenza_ha"]["pdb"] = {
    "id": "1RU7", "chains": ["A", "B"],
    "path": "cd4_validation/pdb/1RU7.pdb",
}

# ---------------------------------------------------------------------------
# 3. RSV Fusion protein F0 (UniProt P03420, strain A2, 574 aa)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 4. Vatreptacog alfa (Factor VIIa variant, P08709 mutant, 466 aa)
# ---------------------------------------------------------------------------
TARGETS["vatreptacog"] = {
    "name": "Vatreptacog alfa (Factor VIIa)",
    "accession": "P08709_mut",
    "sequence": (
        "MVSQALRLLCLLLGLQGCLAAGGVAKASGGETRDMPWKPGPHRVFVTQEEAHGVLHRRRR"
        "ANAFLEELRPGSLERECKEEQCSFEEAREIFKDAERTKLFWISYSDGDQCASSPCQNGGS"
        "CKDQLQSYICFCLPAFEGRNCETHKDDQLICVNENGGCEQYCSDHTGTKRSCRCHEGYSL"
        "LADGDSCTPTVEYPCGKIPILEKRNASKPQGRIVGGKVCPKGECPWQVLLLVNGAQLCGG"
        "TLINTIWVVSAAHCFDKIKNWRNLIAVLGEHDLSEHDGDEQSRRVAQVIIPSTYVPGTTN"
        "HDIALLRLHQPVVLTDHVVPLCLPVRTFSERTLAFVRFSLVSGWGQLLDRGATALELMVL"
        "NVPRLMTQDCLQQSRKVGDSPNITEYMFCAGYSDGSKDSCKGDSGGPHATHYRGTWYLTG"
        "IVSWGQGCATVGHFGVYTRVSQYIEWLQKLMRSEPRPGVLLRAPFP"
    ),
    # Known MHC-II epitopes from vatreptacog immunogenicity studies
    # Cleavage boundaries at positions 284, 294, 310
    "epitopes": [
        {"seq": "VAQVIIPSTYVPGTTNHDIALLRLH", "start": 285, "source": "Vatreptacog_25mer"},
        {"seq": "VPGTTNHDIALLRLH", "start": 295, "source": "Vatreptacog_15mer"},
    ],
    "pdb": {
        "id": "1DAN", "chains": ["L", "H"],
        "path": "cd4_validation/pdb/1DAN_clean.pdb",
    },
}

TARGETS["rsv_f"] = {
    "name": "RSV Fusion Protein F",
    "accession": "P03420",
    "sequence": (
        "MELLILKANAITTILTAVTFCFASGQNITEEFYQSTCSAVSKGYLSALRTGWYTSVI"
        "TIELSIKENKCNGTDAKVKLIKQELDKYKNAVTELQLLMQSTPPTNNRARRELPRFM"
        "NYTLNNAKKTNVTLSKKRKRRFLGFLLGVGSAIASGVAVSKVLHLEGEVNKIKSALL"
        "STNKAVVSLSNGVSVLTSKVLDLKNYIDKQLLPIVNKQSCSISNIETVIEFQQKNNR"
        "LLEITREFSVNAGVTTPVSTYMLTNSELLSLINDMPITNDQKKLMSNNVQIVRQQSYS"
        "IMSIIKEEVLAYVVQLPLYGVIDTPCWKLHTSPLCTTNTKEGSNICLTRTDRGWYCDN"
        "AGSVSFFPQAETCKVQSNRVFCDTMNSLTLPSEINLCNVDIFNPKYDCKIMTSKTDV"
        "SSSVITSLGAIVSCYGKTKCTASNKNRGIIKTFSNGCDYVSNKGMDTVSVGNTLYYVN"
        "KQEGKSLYVKGEPIINFYDPLVFPSDEFDASISQVNEKINQSLAFIRKSDELLHNVNA"
        "GKSTTNIMITTIIIIVIIVILLS"
        "LIAVGLLLYCKARSTPVTLSKDQLSGINNIAFSN"
    ),
    # Varga et al. 2003 J Virol (PMC140824, Table 3)
    # 18-mer, 12-aa overlap, 31 immunodominant peptides, human CD4+, HLA-DR/DQ
    "epitopes": [
        {"seq": "KANAITTILTAVTFCFAS", "start": 7,   "source": "Varga2003"},
        {"seq": "TILTAVTFCFASGQNITE", "start": 13,  "source": "Varga2003"},
        {"seq": "GQNITEEFYQSTCSAVSK", "start": 25,  "source": "Varga2003"},
        {"seq": "EFYQSTCSAVSKGYLSAL", "start": 31,  "source": "Varga2003"},
        {"seq": "GYLSALRTGWYTSVITIE", "start": 43,  "source": "Varga2003"},
        {"seq": "RTGWYTSVITIELSIKEN", "start": 49,  "source": "Varga2003"},
        {"seq": "SVITIELSIKENKCNGTD", "start": 55,  "source": "Varga2003"},
        {"seq": "DAKVKLIKQELDKYKNAV", "start": 73,  "source": "Varga2003"},
        {"seq": "IKQELDKYKNAVTELQLL", "start": 79,  "source": "Varga2003"},
        {"seq": "KYKNAVTELQLLMQSTPP", "start": 85,  "source": "Varga2003"},
        {"seq": "RELPRFMNYTLNNAKKTN", "start": 109, "source": "Varga2003"},
        {"seq": "MNYTLNNAKKTNVTLSKK", "start": 115, "source": "Varga2003"},
        {"seq": "NKAVVSLSNGVSVLTSKV", "start": 175, "source": "Varga2003"},
        {"seq": "LDLKNYIDKQLLPIVNKQ", "start": 193, "source": "Varga2003"},
        {"seq": "RLLEITREFSVNAGVTTP", "start": 229, "source": "Varga2003"},
        {"seq": "REFSVNAGVTTPVSTYML", "start": 235, "source": "Varga2003"},
        {"seq": "PITNDQKKLMSNNVQIVR", "start": 265, "source": "Varga2003"},
        {"seq": "KKLMSNNVQIVRQQSYSI", "start": 271, "source": "Varga2003"},
        {"seq": "EVLAYVVQLPLYGVIDTP", "start": 295, "source": "Varga2003"},
        {"seq": "VQLPLYGVIDTPCWKLHT", "start": 301, "source": "Varga2003"},
        {"seq": "TDRGWYCDNAGSVSFFPQ", "start": 337, "source": "Varga2003"},
        {"seq": "CDNAGSVSFFPQAETCKV", "start": 343, "source": "Varga2003"},
        {"seq": "YDCKIMTSKTDVSSSVIT", "start": 391, "source": "Varga2003"},
        {"seq": "SLGAIVSCYGKTKCTASN", "start": 409, "source": "Varga2003"},
        {"seq": "KNRGIIKTFSNGCDYVSN", "start": 427, "source": "Varga2003"},
        {"seq": "YYVNKQEGKSLYVKGEPI", "start": 457, "source": "Varga2003"},
        {"seq": "VKGEPIINFYDPLVFPSD", "start": 469, "source": "Varga2003"},
        {"seq": "SQVNEKINQSLAFIRKSD", "start": 493, "source": "Varga2003"},
        {"seq": "INQSLAFIRKSDELLHNV", "start": 499, "source": "Varga2003"},
        {"seq": "NAGKSTTNIMITTIIIIV", "start": 517, "source": "Varga2003"},
        {"seq": "LIAVGLLLYCKARSTPVT", "start": 541, "source": "Varga2003"},
    ],
    "pdb": {
        "id": "8W3K", "chains": ["F"],
        "path": "cd4_validation/pdb/8W3K.pdb",
    },
}


# ── Load IEDB epitopes (if available) and auto-fix positions ──────────────
IEDB_EPITOPE_DIR = Path("cd4_validation/iedb_epitopes")

def _load_iedb_epitopes(epitope_type="all"):
    """Load epitopes from IEDB JSON files if available, replacing hardcoded ones.

    Args:
        epitope_type: "all" for all IEDB epitopes, "elution" for naturally processed only
    """
    suffix = "_all_epitopes.json" if epitope_type == "all" else "_elution_epitopes.json"
    for tname, tdata in TARGETS.items():
        json_path = IEDB_EPITOPE_DIR / f"{tname}{suffix}"
        if json_path.exists():
            with open(json_path) as f:
                iedb_eps = json.load(f)
            if len(iedb_eps) > 0:
                tdata["epitopes"] = [
                    {"seq": ep["seq"], "start": ep["start"], "source": ep.get("source", "IEDB")}
                    for ep in iedb_eps
                ]
    # Sync spike_ntd / spike_rbd with spike
    for variant in ("spike_ntd", "spike_rbd"):
        if variant in TARGETS and "spike" in TARGETS:
            TARGETS[variant]["epitopes"] = list(TARGETS["spike"]["epitopes"])

def _autofix_epitope_positions():
    """Recalculate all epitope start positions from actual sequence."""
    for tname, tdata in TARGETS.items():
        seq = tdata["sequence"]
        for ep in tdata["epitopes"]:
            pos = seq.find(ep["seq"])
            if pos >= 0:
                ep["start"] = pos + 1  # 1-indexed
            else:
                print(f"  WARNING: epitope '{ep['seq']}' not found in {tname}")

_load_iedb_epitopes(epitope_type="elution")
_autofix_epitope_positions()


# ── Step 1 ─────────────────────────────────────────────────────────────────
def get_target(target_name):
    """Return target dict (sequence + epitopes)."""
    if target_name not in TARGETS:
        raise ValueError(f"Unknown target '{target_name}'. Available: {list(TARGETS.keys())}")
    return TARGETS[target_name]


def load_epitopes(target, csv_path=None):
    """Load epitopes: hardcoded + optional CSV."""
    epitopes = list(target["epitopes"])

    if csv_path and Path(csv_path).exists():
        n_before = len(epitopes)
        print(f"  Loading additional epitopes from {csv_path} ...")
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                epitopes.append({
                    "seq": row["sequence"].strip().upper(),
                    "start": int(row.get("start", 0)),
                    "source": row.get("source", "csv"),
                })
        print(f"  +{len(epitopes) - n_before} epitopes from CSV")

    print(f"  Total epitopes: {len(epitopes)}")
    return epitopes


# ── Step 2: Feature Extraction (IF1 + PROPKA) ────────────────────────────

def extract_if1_features(pdb_path, chain_id, model_cache=None, device="cpu"):
    """Extract IF1 (inverse folding) embeddings from a PDB structure.

    Returns (embeddings, pdb_seq): ((N_pdb, 512), str)
    """
    from data.build_if1_propka_dataset import load_if1_model, extract_if1_embeddings

    if model_cache and "if1_model" in model_cache:
        model, alphabet = model_cache["if1_model"]
    else:
        print("  Loading ESM-IF1 model ...")
        model, alphabet = load_if1_model(device)
        if model_cache is not None:
            model_cache["if1_model"] = (model, alphabet)

    print(f"  Extracting IF1 embeddings from {pdb_path} chain {chain_id} ...")
    if1_emb, pdb_seq = extract_if1_embeddings(pdb_path, chain_id, model, alphabet, device)
    print(f"  IF1 features: {if1_emb.shape}, PDB seq len: {len(pdb_seq)}")
    return if1_emb, pdb_seq


def extract_propka_features(sequence):
    """Sequence-based PROPKA features (training-compatible format)."""
    pka_values = {
        "D": 3.9, "E": 4.3, "H": 6.0, "C": 8.3,
        "Y": 10.1, "K": 10.5, "R": 12.5,
    }
    seq_len = len(sequence)
    feats = np.zeros((seq_len, 6), dtype=np.float32)
    for i, aa in enumerate(sequence):
        feats[i, 0] = pka_values.get(aa, 7.0)
    print(f"  PROPKA features: {feats.shape}")
    return feats


def align_pdb_to_uniprot(pdb_seq, uniprot_seq):
    """Pairwise-align PDB sequence to UniProt reference.

    Returns:
        pdb_to_uni: dict {pdb_idx → uniprot_pos_0based}
        uni_to_pdb: dict {uniprot_pos_0based → pdb_idx}
        identity: float, fraction of aligned identical residues
    """
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -5
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(uniprot_seq, pdb_seq)
    best = alignments[0]

    # Use .aligned to get block coordinates (handles local alignment offsets)
    # aligned[0] = [[start, end], ...] for target (uniprot)
    # aligned[1] = [[start, end], ...] for query (pdb)
    blocks_uni = best.aligned[0]
    blocks_pdb = best.aligned[1]

    pdb_to_uni = {}
    uni_to_pdb = {}
    matches = 0
    aligned = 0

    for (uni_start, uni_end), (pdb_start, pdb_end) in zip(blocks_uni, blocks_pdb):
        for uni_pos, pdb_pos in zip(
            range(uni_start, uni_end), range(pdb_start, pdb_end)
        ):
            pdb_to_uni[pdb_pos] = uni_pos
            uni_to_pdb[uni_pos] = pdb_pos
            aligned += 1
            if uniprot_seq[uni_pos] == pdb_seq[pdb_pos]:
                matches += 1

    identity = matches / max(aligned, 1)

    return pdb_to_uni, uni_to_pdb, identity


def extract_features_if1(pdb_info, cache_path, model_cache=None, device="cpu"):
    """IF1 + PROPKA → (N_pdb, 518). Cache result.

    Handles multi-chain PDBs by extracting per-chain and concatenating.

    Returns: (features_tensor, pdb_seq_concatenated)
    """
    cache = Path(cache_path)
    if cache.exists():
        print(f"  Loading cached features from {cache} ...")
        data = torch.load(cache, map_location="cpu", weights_only=False)
        return data["features"], data["pdb_seq"]

    pdb_path = pdb_info["path"]
    chains = pdb_info["chains"]

    all_emb = []
    all_seq = []

    for chain_id in chains:
        if1_emb, pdb_seq = extract_if1_features(
            pdb_path, chain_id, model_cache=model_cache, device=device
        )
        all_emb.append(if1_emb)
        all_seq.append(pdb_seq)

    combined_emb = np.concatenate(all_emb, axis=0)  # (N_total, 512)
    combined_seq = "".join(all_seq)

    propka_feats = extract_propka_features(combined_seq)  # (N_total, 6)
    features = np.concatenate([combined_emb, propka_feats], axis=1)  # (N_total, 518)
    features = torch.from_numpy(features).float()

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "pdb_seq": combined_seq}, cache)
    print(f"  Features saved to {cache}  shape={features.shape}")
    return features, combined_seq


# ── Step 3 ─────────────────────────────────────────────────────────────────
class _CompatBiLSTMGNN(BiLSTMGNN):
    """BiLSTMGNN variant that can skip long_skip_norm for older checkpoints."""

    def __init__(self, use_long_skip=True, affine_norm=False, **kwargs):
        super().__init__(**kwargs)
        self._use_long_skip = use_long_skip
        hidden_dim = kwargs["hidden_dim"]
        if affine_norm:
            self.input_norm = torch.nn.LayerNorm(hidden_dim)
            self.pre_attn_norm = torch.nn.LayerNorm(hidden_dim)
            self.post_lstm_norm = torch.nn.LayerNorm(hidden_dim)
            self.ffn_norm = torch.nn.LayerNorm(hidden_dim)
        if not use_long_skip:
            # Replace with identity so the attribute still exists
            self.long_skip_norm = torch.nn.Identity()

    def forward(self, x, batch=None):
        import torch.nn.functional as F
        x_orig = x

        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.gelu(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        x_proj = x

        # Single sequence (inference mode)
        x = x_proj.unsqueeze(0)
        x = self.rope(x)

        attn_out, _ = self.pre_attention(x, x, x)
        x = self.pre_attn_norm(x + self.dropout(attn_out))

        residual = x
        lstm_out, _ = self.lstm(x)
        x = self.post_lstm_norm(lstm_out + residual)

        x = x.squeeze(0)
        x = self.gated_fusion(x_orig, x)

        residual = x
        ffn_out = self.ffn(x)
        x = self.ffn_norm(residual + self.dropout(ffn_out))

        if self._use_long_skip:
            x = self.long_skip_norm(x + x_proj)

        x = self.output_proj(x)
        return x.squeeze(-1)


def load_model(checkpoint_path, device="cpu"):
    """Load BiLSTMGNN model, auto-detecting architecture from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    input_dim = sd["input_proj.weight"].shape[1]
    has_affine = "input_norm.weight" in sd
    has_long_skip = any("long_skip" in k for k in sd.keys())

    model = _CompatBiLSTMGNN(
        use_long_skip=has_long_skip,
        affine_norm=has_affine,
        input_dim=input_dim, hidden_dim=256, output_dim=1,
        num_layers=2, dropout=0.2,
    )
    model.load_state_dict(sd, strict=False)
    epoch_info = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Loaded checkpoint (epoch {epoch_info}, input_dim={input_dim})")
    model.eval()
    return model, input_dim


def predict(model, features, device="cpu"):
    """Run per-residue cleavage prediction."""
    model.to(device)
    features = features.to(device)
    with torch.no_grad():
        logits = model(features, batch=None)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ── Step 4 ─────────────────────────────────────────────────────────────────
def map_epitope_boundaries(epitopes, spike_seq):
    """Map epitopes to Spike positions, return boundary set and epitope info."""
    mapped = []
    boundary_positions = set()

    for ep in epitopes:
        seq = ep["seq"]
        # Try hardcoded start first, then fallback to str.find
        start_0 = ep["start"] - 1  # convert to 0-based
        if spike_seq[start_0 : start_0 + len(seq)] == seq:
            pass  # position confirmed
        else:
            # Fallback: search in spike
            idx = spike_seq.find(seq)
            if idx == -1:
                print(f"  WARNING: epitope {seq} not found in Spike, skipping")
                continue
            start_0 = idx

        end_0 = start_0 + len(seq) - 1
        n_term = start_0       # N-terminal boundary (0-based)
        c_term = end_0          # C-terminal boundary (0-based)

        boundary_positions.add(n_term)
        boundary_positions.add(c_term)

        mapped.append({
            "seq": seq,
            "start_0": start_0,
            "end_0": end_0,
            "n_term": n_term,
            "c_term": c_term,
            "source": ep["source"],
        })

    print(f"  Mapped epitopes: {len(mapped)}/{len(epitopes)}")
    print(f"  Unique boundary positions: {len(boundary_positions)}")
    return mapped, boundary_positions


def compute_min_distance_to_boundary(seq_len, boundary_positions):
    """For each residue, compute minimum distance to any boundary."""
    boundaries = np.array(sorted(boundary_positions))
    positions = np.arange(seq_len)
    # Broadcast: |positions - boundaries|
    dists = np.abs(positions[:, None] - boundaries[None, :])
    min_dists = dists.min(axis=1)
    return min_dists


# ── Step 5 ─────────────────────────────────────────────────────────────────
def enrichment_permutation_test(probs, boundary_positions, seq_len, n_perm=10000, rng_seed=42):
    """Permutation test: are boundary positions' mean prob > random?"""
    rng = np.random.RandomState(rng_seed)
    boundary_idx = np.array(sorted(boundary_positions))
    observed = probs[boundary_idx].mean()

    n_boundary = len(boundary_idx)
    count_ge = 0
    perm_means = np.zeros(n_perm)
    for i in range(n_perm):
        rand_idx = rng.choice(seq_len, size=n_boundary, replace=False)
        perm_mean = probs[rand_idx].mean()
        perm_means[i] = perm_mean
        if perm_mean >= observed:
            count_ge += 1

    p_value = (count_ge + 1) / (n_perm + 1)  # +1 for observed itself
    effect_size = observed - perm_means.mean()

    return {
        "observed_mean": float(observed),
        "permutation_mean": float(perm_means.mean()),
        "permutation_std": float(perm_means.std()),
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "n_permutations": n_perm,
    }


def distance_profile_analysis(probs, min_dists, max_dist=20):
    """Mean cleavage probability by distance-to-boundary bin."""
    from scipy.stats import spearmanr

    bins = list(range(max_dist + 1))
    bin_means = []
    bin_counts = []
    for d in bins:
        mask = min_dists == d
        if mask.sum() > 0:
            bin_means.append(float(probs[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_means.append(float("nan"))
            bin_counts.append(0)

    # Spearman correlation (distance vs mean prob) - expect negative
    valid = [(d, m) for d, m in zip(bins, bin_means) if not np.isnan(m)]
    if len(valid) >= 3:
        d_vals, m_vals = zip(*valid)
        rho, p = spearmanr(d_vals, m_vals)
    else:
        rho, p = float("nan"), float("nan")

    return {
        "bins": bins,
        "mean_prob": bin_means,
        "counts": bin_counts,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }


def boundary_auc_analysis(probs, boundary_positions, seq_len):
    """ROC AUC treating boundary positions as positives."""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.zeros(seq_len, dtype=int)
    for pos in boundary_positions:
        labels[pos] = 1

    auc = roc_auc_score(labels, probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)

    return {
        "auc": float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def precision_recall_at_distance(probs, min_dists, thresholds=[0, 1, 2, 3, 4, 5]):
    """Precision/recall treating positions within distance threshold as positive."""
    results = {}
    pred_positive = probs > 0.5

    for d in thresholds:
        actual_positive = min_dists <= d
        tp = (pred_positive & actual_positive).sum()
        fp = (pred_positive & ~actual_positive).sum()
        fn = (~pred_positive & actual_positive).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"dist<={d}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "n_positive": int(actual_positive.sum()),
            "n_predicted": int(pred_positive.sum()),
        }
    return results


# ── Step 6 ─────────────────────────────────────────────────────────────────
def plot_cleavage_profile(probs, mapped_epitopes, protein_seq, out_path,
                          protein_name=""):
    """Per-residue cleavage probability profile with epitope regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 5))

    positions = np.arange(1, len(protein_seq) + 1)  # 1-based
    ax.plot(positions, probs, color="steelblue", linewidth=0.5, alpha=0.8)

    # Highlight epitope regions - auto-assign colors to sources
    palette = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#9B59B6", "#E67E22", "#1ABC9C"]
    source_colors = {}
    labeled = set()
    for ep in mapped_epitopes:
        src = ep["source"]
        if src not in source_colors:
            source_colors[src] = palette[len(source_colors) % len(palette)]
        label = src if src not in labeled else None
        labeled.add(src)
        ax.axvspan(
            ep["start_0"] + 1, ep["end_0"] + 1,
            alpha=0.15, color=source_colors[src], label=label,
        )
        # Mark boundaries (may be absent for partial PDB coverage)
        if "n_term" in ep:
            ax.axvline(ep["n_term"] + 1, color="red", alpha=0.3, linewidth=0.5, linestyle="--")
        if "c_term" in ep:
            ax.axvline(ep["c_term"] + 1, color="red", alpha=0.3, linewidth=0.5, linestyle="--")

    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Cleavage Probability")
    ax.set_title(f"GeoCleav Cleavage Profile on {protein_name} with CD4+ Epitope Regions")
    ax.set_xlim(1, len(protein_seq))
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_distance_histogram(dist_profile, out_path):
    """Bar chart of mean cleavage probability by distance to boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = dist_profile["bins"]
    means = dist_profile["mean_prob"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#E74C3C" if d <= 2 else "#3498DB" for d in bins]
    ax.bar(bins, means, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Distance to Nearest Epitope Boundary (residues)")
    ax.set_ylabel("Mean Cleavage Probability")
    ax.set_title(
        f"Cleavage Probability vs Distance to Epitope Boundary\n"
        f"(Spearman rho={dist_profile['spearman_rho']:.3f}, "
        f"p={dist_profile['spearman_p']:.4f})"
    )
    ax.set_xticks(bins)

    # Reference line: global mean
    global_mean = np.nanmean(means)
    ax.axhline(global_mean, color="gray", linestyle="--", linewidth=1,
               label=f"Global mean = {global_mean:.4f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_boundary_roc(auc_result, out_path):
    """ROC curve for boundary detection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(auc_result["fpr"], auc_result["tpr"], color="steelblue", linewidth=2,
            label=f"AUC = {auc_result['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Epitope Boundary Detection by Cleavage Probability")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Step 7 ─────────────────────────────────────────────────────────────────
def save_predictions_csv(probs, protein_seq, min_dists, boundary_positions, out_path):
    """Save per-residue predictions."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["position", "residue", "probability", "predicted",
                         "is_boundary", "min_dist_to_boundary"])
        for i in range(len(protein_seq)):
            writer.writerow([
                i + 1,
                protein_seq[i],
                f"{probs[i]:.6f}",
                int(probs[i] > 0.5),
                int(i in boundary_positions),
                int(min_dists[i]),
            ])
    print(f"  Saved: {out_path}")


def save_summary(enrichment, dist_profile, auc_result, pr_results,
                 mapped_epitopes, boundary_positions, probs, out_path,
                 protein_name="", accession=""):
    """Save human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("GeoCleav CD4+ Epitope Validation Summary")
    lines.append("=" * 70)

    lines.append(f"\nProtein: {protein_name} ({accession}, {len(probs)} aa)")
    lines.append(f"Epitopes mapped: {len(mapped_epitopes)}")
    lines.append(f"Unique boundary positions: {len(boundary_positions)}")
    lines.append(f"Predicted positives (p>0.5): {(probs > 0.5).sum()}/{len(probs)}")

    lines.append(f"\n--- Enrichment Test (Permutation, n={enrichment['n_permutations']}) ---")
    lines.append(f"  Boundary mean prob:    {enrichment['observed_mean']:.4f}")
    lines.append(f"  Random mean prob:      {enrichment['permutation_mean']:.4f} +/- {enrichment['permutation_std']:.4f}")
    lines.append(f"  Effect size:           {enrichment['effect_size']:.4f}")
    lines.append(f"  p-value:               {enrichment['p_value']:.4f}")
    sig = "***" if enrichment["p_value"] < 0.001 else "**" if enrichment["p_value"] < 0.01 else "*" if enrichment["p_value"] < 0.05 else "n.s."
    lines.append(f"  Significance:          {sig}")

    lines.append(f"\n--- Distance Profile (Spearman) ---")
    lines.append(f"  rho = {dist_profile['spearman_rho']:.4f}")
    lines.append(f"  p   = {dist_profile['spearman_p']:.4f}")
    lines.append(f"  (Negative rho = higher prob closer to boundaries)")

    lines.append(f"\n--- Boundary Detection ROC ---")
    lines.append(f"  AUC = {auc_result['auc']:.4f}")

    lines.append(f"\n--- Precision/Recall at Distance Thresholds ---")
    for key, val in pr_results.items():
        lines.append(f"  {key}: P={val['precision']:.3f}  R={val['recall']:.3f}  "
                      f"F1={val['f1']:.3f}  (n_pos={val['n_positive']}, n_pred={val['n_predicted']})")

    lines.append(f"\n--- Per-Epitope Boundary Probabilities ---")
    for ep in mapped_epitopes:
        lines.append(f"  {ep['seq']}  ({ep['source']})")
        parts = []
        if "n_term" in ep:
            n_prob = probs[ep["n_term"]]
            parts.append(f"N-term pos {ep['n_term']+1}: {n_prob:.4f}")
        else:
            parts.append("N-term: outside PDB")
        if "c_term" in ep:
            c_prob = probs[ep["c_term"]]
            parts.append(f"C-term pos {ep['c_term']+1}: {c_prob:.4f}")
        else:
            parts.append("C-term: outside PDB")
        lines.append(f"    {'   '.join(parts)}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    Path(out_path).write_text(text)
    print(f"  Saved: {out_path}")
    return text


# ── PDB coverage check ────────────────────────────────────────────────────

def check_pdb_coverage():
    """Check PDB coverage for all targets and report epitope mapping status."""
    from esm.inverse_folding.util import load_structure, extract_coords_from_structure

    print("=" * 70)
    print("PDB Coverage Check")
    print("=" * 70)

    for tname, tdata in TARGETS.items():
        pdb_info = tdata.get("pdb")
        if not pdb_info:
            print(f"\n{tname}: NO PDB CONFIGURED")
            continue

        protein_seq = tdata["sequence"]
        print(f"\n{tname} ({tdata['accession']}, {len(protein_seq)} aa)")
        print(f"  PDB: {pdb_info['id']} chain(s) {pdb_info['chains']}")

        pdb_path = pdb_info["path"]
        if not Path(pdb_path).exists():
            print(f"  ERROR: PDB file not found: {pdb_path}")
            continue

        # Extract PDB sequences and concatenate
        all_seq = []
        for chain_id in pdb_info["chains"]:
            try:
                structure = load_structure(pdb_path, chain_id)
                _, seq = extract_coords_from_structure(structure)
                all_seq.append(seq)
                print(f"  Chain {chain_id}: {len(seq)} residues")
            except Exception as e:
                print(f"  Chain {chain_id}: ERROR: {e}")

        if not all_seq:
            continue

        pdb_seq = "".join(all_seq)
        print(f"  Total PDB residues: {len(pdb_seq)}")

        # Alignment
        _, uni_to_pdb, identity = align_pdb_to_uniprot(pdb_seq, protein_seq)
        covered = sum(1 for v in uni_to_pdb.values() if v is not None)
        print(f"  Alignment: {covered}/{len(protein_seq)} residues covered "
              f"({100*covered/len(protein_seq):.1f}%), identity={identity:.3f}")

        # Check epitopes
        n_full = 0
        n_partial = 0
        n_none = 0
        for ep in tdata["epitopes"]:
            start_0 = ep["start"] - 1
            end_0 = start_0 + len(ep["seq"]) - 1
            n_boundary = uni_to_pdb.get(start_0) is not None
            c_boundary = uni_to_pdb.get(end_0) is not None
            if n_boundary and c_boundary:
                n_full += 1
            elif n_boundary or c_boundary:
                n_partial += 1
            else:
                n_none += 1

        print(f"  Epitopes: {n_full} full, {n_partial} partial, {n_none} uncovered "
              f"(of {len(tdata['epitopes'])} total)")


# ── Main ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="CD4+ epitope validation for GeoCleav",
    )
    parser.add_argument("--target", nargs="+",
                        default=["spike"],
                        choices=list(TARGETS.keys()) + ["all"],
                        help="Target protein(s) to validate (default: spike)")
    parser.add_argument("--checkpoint", default="results/bilstm_best.pt",
                        help="Path to BiLSTM checkpoint")
    parser.add_argument("--output-dir", default="cd4_validation",
                        help="Output directory")
    parser.add_argument("--epitope-csv", default=None,
                        help="Optional CSV with extra epitopes (columns: sequence, start, source)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Number of permutations for enrichment test")
    parser.add_argument("--check-pdb", action="store_true",
                        help="Only check PDB coverage, do not run validation")
    parser.add_argument("--epitope-type", default="elution",
                        choices=["all", "elution", "hardcoded"],
                        help="Epitope source: elution (naturally processed, default), all (IEDB all), hardcoded")
    return parser.parse_args()


def run_validation(target_key, args, model=None, model_input_dim=None,
                   if1_model_cache=None):
    """Run validation pipeline for a single target protein.

    Uses IF1 (inverse folding, PDB structure-based) features.
    Predictions are per-PDB-residue, then mapped to UniProt coordinates
    for epitope boundary analysis.
    """
    target = get_target(target_key)
    protein_seq = target["sequence"]
    protein_name = target["name"]
    pdb_info = target.get("pdb")
    tag = target_key

    print("\n" + "=" * 70)
    print(f"  Target: {protein_name} ({target['accession']}, {len(protein_seq)} aa)")
    if pdb_info:
        print(f"  PDB: {pdb_info['id']} chain(s) {pdb_info['chains']}")
    print("=" * 70)

    # Output dirs
    out = Path(args.output_dir) / tag
    for sub in ("data", "predictions", "analysis", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Epitope data ──
    print("\n[1/7] Loading epitope data ...")
    epitopes = load_epitopes(target, args.epitope_csv)

    # ── Step 2: Feature extraction (IF1 + PROPKA) ──
    print("\n[2/7] Extracting features (IF1 + PROPKA, 518-dim) ...")
    if not pdb_info or not Path(pdb_info["path"]).exists():
        print(f"  ERROR: PDB not available for {target_key}, skipping")
        return model, model_input_dim, if1_model_cache

    cache_path = out / "data" / f"features_{tag}_if1.pt"
    features, pdb_seq = extract_features_if1(
        pdb_info, cache_path, model_cache=if1_model_cache, device=args.device
    )
    print(f"  Feature shape: {features.shape}")
    print(f"  PDB sequence length: {len(pdb_seq)}")

    # ── Step 2b: PDB-to-UniProt alignment ──
    print("\n  Aligning PDB sequence to UniProt reference ...")
    pdb_to_uni, uni_to_pdb, identity = align_pdb_to_uniprot(pdb_seq, protein_seq)
    covered = sum(1 for v in uni_to_pdb.values() if v is not None)
    print(f"  Alignment: {covered}/{len(protein_seq)} UniProt residues covered "
          f"({100*covered/len(protein_seq):.1f}%), identity={identity:.3f}")

    # ── Step 3: Prediction ──
    print("\n[3/7] Running GeoCleav prediction ...")
    if model is None:
        model, model_input_dim = load_model(args.checkpoint, device=args.device)
    feat = features
    if feat.shape[1] != model_input_dim:
        print(f"  Adjusting features {feat.shape[1]} -> {model_input_dim}")
        if feat.shape[1] < model_input_dim:
            pad_cols = torch.zeros(feat.shape[0], model_input_dim - feat.shape[1])
            feat = torch.cat([feat, pad_cols], dim=1)
        else:
            feat = feat[:, :model_input_dim]
    pdb_probs = predict(model, feat, device=args.device)
    print(f"  Predictions (PDB-residue): {pdb_probs.shape}")
    print(f"  Prob range: [{pdb_probs.min():.4f}, {pdb_probs.max():.4f}]")
    print(f"  Mean prob: {pdb_probs.mean():.4f}")
    print(f"  Predicted positives (>0.5): {(pdb_probs > 0.5).sum()}/{len(pdb_probs)}")

    # ── Step 4: Epitope boundary mapping (UniProt → PDB coords) ──
    print("\n[4/7] Mapping epitope boundaries (UniProt → PDB) ...")
    mapped = []
    boundary_positions_pdb = set()  # PDB indices
    skipped_epitopes = []

    for ep in epitopes:
        seq_ep = ep["seq"]
        start_0 = ep["start"] - 1
        end_0 = start_0 + len(seq_ep) - 1

        n_pdb = uni_to_pdb.get(start_0)
        c_pdb = uni_to_pdb.get(end_0)

        if n_pdb is None and c_pdb is None:
            skipped_epitopes.append({"seq": seq_ep, "reason": "both boundaries outside PDB"})
            continue

        info = {
            "seq": seq_ep,
            "uni_start_0": start_0,
            "uni_end_0": end_0,
            "source": ep["source"],
        }

        if n_pdb is not None:
            boundary_positions_pdb.add(n_pdb)
            info["n_term"] = n_pdb
        if c_pdb is not None:
            boundary_positions_pdb.add(c_pdb)
            info["c_term"] = c_pdb

        # For compatibility with plotting (use start_0/end_0 in PDB space)
        info["start_0"] = n_pdb if n_pdb is not None else c_pdb
        info["end_0"] = c_pdb if c_pdb is not None else n_pdb

        mapped.append(info)

    print(f"  Mapped epitopes: {len(mapped)}/{len(epitopes)}")
    if skipped_epitopes:
        print(f"  Skipped (outside PDB): {len(skipped_epitopes)}")
        for s in skipped_epitopes:
            print(f"    {s['seq'][:30]} — {s['reason']}")
    print(f"  Unique boundary positions (PDB): {len(boundary_positions_pdb)}")

    # Use PDB sequence length for analysis
    seq_len_pdb = len(pdb_seq)
    min_dists = compute_min_distance_to_boundary(seq_len_pdb, boundary_positions_pdb)
    print(f"  Distance range: [{min_dists.min()}, {min_dists.max()}]")

    # ── Step 5: Statistical analysis ──
    print("\n[5/7] Statistical analysis ...")

    print("  Running enrichment permutation test ...")
    enrichment = enrichment_permutation_test(
        pdb_probs, boundary_positions_pdb, seq_len_pdb, n_perm=args.n_perm,
    )
    print(f"    Boundary mean: {enrichment['observed_mean']:.4f}")
    print(f"    Random mean:   {enrichment['permutation_mean']:.4f}")
    print(f"    p-value:       {enrichment['p_value']:.4f}")

    print("  Computing distance profile ...")
    dist_profile = distance_profile_analysis(pdb_probs, min_dists)
    print(f"    Spearman rho: {dist_profile['spearman_rho']:.4f}")

    print("  Computing boundary ROC ...")
    auc_result = boundary_auc_analysis(pdb_probs, boundary_positions_pdb, seq_len_pdb)
    print(f"    AUC: {auc_result['auc']:.4f}")

    print("  Computing precision/recall at distance thresholds ...")
    pr_results = precision_recall_at_distance(pdb_probs, min_dists)

    # ── Step 6: Visualization ──
    print("\n[6/7] Creating visualizations ...")
    plot_cleavage_profile(pdb_probs, mapped, pdb_seq,
                          out / "plots" / f"{tag}_cleavage_profile.png",
                          protein_name=f"{protein_name} (PDB {pdb_info['id']})")
    plot_distance_histogram(dist_profile,
                            out / "plots" / f"{tag}_distance_histogram.png")
    plot_boundary_roc(auc_result,
                      out / "plots" / f"{tag}_boundary_roc.png")

    # ── Step 7: Save outputs ──
    print("\n[7/7] Saving outputs ...")
    save_predictions_csv(
        pdb_probs, pdb_seq, min_dists, boundary_positions_pdb,
        out / "predictions" / f"{tag}_predictions.csv",
    )

    # Enrichment report JSON
    report = {
        "protein": f"{protein_name} ({target['accession']})",
        "pdb": pdb_info["id"],
        "pdb_chains": pdb_info["chains"],
        "uniprot_length": len(protein_seq),
        "pdb_length": len(pdb_seq),
        "alignment_coverage": covered,
        "alignment_identity": round(identity, 4),
        "n_epitopes_mapped": len(mapped),
        "n_epitopes_skipped": len(skipped_epitopes),
        "n_boundary_positions": len(boundary_positions_pdb),
        "enrichment": enrichment,
        "distance_profile": {k: v for k, v in dist_profile.items()
                             if k != "counts"},
        "auc": auc_result["auc"],
        "precision_recall": pr_results,
    }
    if skipped_epitopes:
        report["skipped_epitopes"] = skipped_epitopes

    report_path = out / "analysis" / "enrichment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    summary_text = save_summary(
        enrichment, dist_profile, auc_result, pr_results,
        mapped, boundary_positions_pdb, pdb_probs,
        out / "analysis" / "validation_summary.txt",
        protein_name=f"{protein_name} (PDB {pdb_info['id']})",
        accession=target["accession"],
    )
    print("\n" + summary_text)

    return model, model_input_dim, if1_model_cache


def main():
    args = parse_args()

    if args.check_pdb:
        check_pdb_coverage()
        return

    # Reload epitopes if needed
    if args.epitope_type != "hardcoded":
        _load_iedb_epitopes(epitope_type=args.epitope_type)
        _autofix_epitope_positions()

    print("=" * 70)
    print("GeoCleav CD4+ Epitope Validation Pipeline (IF1)")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epitope type: {args.epitope_type}")

    # Resolve target list
    targets = args.target
    if "all" in targets:
        targets = list(TARGETS.keys())
    print(f"Targets: {targets}")

    # Load model once, reuse across targets
    model, model_input_dim = load_model(args.checkpoint, device=args.device)
    if1_model_cache = {}  # shared IF1 model across targets

    for target_key in targets:
        model, model_input_dim, if1_model_cache = run_validation(
            target_key, args,
            model=model, model_input_dim=model_input_dim,
            if1_model_cache=if1_model_cache,
        )

    print("\n" + "=" * 70)
    print("All targets completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
