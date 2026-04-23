# IF-Cleave

Inverse folding-based deep learning for MHC-II antigen processing cleavage site prediction.

IF-Cleave combines ESM-IF1 inverse folding embeddings (512D) with PROPKA physicochemical features (6D) to predict where endosomal proteases cleave antigen proteins for MHC class II presentation — a critical step in CD4+ T cell immune responses.

## Quick Start

Pre-trained 4-fold checkpoints are shipped in `checkpoints/`. The Makefile chains data prep → inference/training → evaluation; each step skips if its output already exists.

```bash
make inference   # reproduce metrics with shipped checkpoints
make train       # 4-fold CV from scratch, then evaluate
```

Override defaults with `make inference PKL=path/to/features.pkl WINDOW=11 RESULTS=results`.

## Installation

```bash
conda create -n ifcleave python=3.10
conda activate ifcleave

# Install PyTorch matching your CUDA driver version (check with nvidia-smi)
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Prepare features (one-time)

Needed before `make inference` or `make train` if `all_datasets_fixed.pkl` is not already present.

```bash
python data/build_db.py          # Build PDB database from IEDB/UniProt/RCSB
python data/extract_features.py  # Extract IF1 + PROPKA features -> all_datasets_fixed.pkl
```

## Project Structure

```
if-cleave/
├── Makefile        # End-to-end `make inference` / `make train` orchestration
├── predict.py      # 4-fold ensemble inference
├── model/          # IFCleave model architecture
├── train/          # K-fold cross-validation training
├── eval/           # Window-based evaluation
├── data/           # Data pipeline (IEDB → PDB → features → windows) + index.csv
├── utils/          # Shared utilities (metrics, dataset, standardization)
└── checkpoints/    # 4-fold pre-trained weights
```

## Citation

```bibtex
@article{ifcleave2026,
  title={IF-Cleave: Inverse Folding-Based Deep Learning for MHC-II Antigen Processing Cleavage Site Prediction},
  journal={Bioinformatics},
  year={2026}
}
```
