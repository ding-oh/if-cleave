# IF-Cleave

Inverse folding-based deep learning for MHC-II antigen processing cleavage site prediction.

IF-Cleave predicts where endosomal proteases cleave antigen proteins for MHC class II presentation — a critical step in CD4+ T cell immune responses.

## Predict on a single PDB

Extracts ESM-IF1 embeddings, runs `propka3`, and returns per-residue cleavage probabilities from the 4-fold ensemble.

```bash
python predict.py --pdb path/to/protein.pdb --chain A --output predictions.tsv
```

Output columns: `res_num`, `aa`, `prob`, `cleaved` (thresholded at 0.5).

## Reproduce benchmark metrics

Pre-trained 4-fold checkpoints are shipped in `checkpoints/`. The Makefile chains data prep → predict/train → evaluation; each step skips if its output already exists.

```bash
make reproduce   # reproduce benchmark metrics with shipped checkpoints
make train       # 4-fold CV from scratch, then evaluate
```

Override defaults with `make reproduce PKL=path/to/features.pkl WINDOW=11 RESULTS=results`.

## Installation

```bash
conda create -n ifcleave python=3.10
conda activate ifcleave

# Install PyTorch matching your CUDA driver version (check with nvidia-smi)
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# torch_scatter matching your torch/CUDA build (required by ESM-IF1)
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

pip install -r requirements.txt
```

## Prepare features (one-time)

Needed before `make reproduce` or `make train` if `all_datasets_fixed.pkl` is not already present.

```bash
python data/build_db.py          # Build PDB database from IEDB/UniProt/RCSB
python data/extract_features.py  # Extract IF1 + PROPKA features -> all_datasets_fixed.pkl
```

## Project Structure

```
if-cleave/
├── Makefile        # End-to-end `make reproduce` / `make train` orchestration
├── predict.py      # Single-PDB inference (IF1 + PROPKA extraction + ensemble)
├── reproduce.py    # 4-fold ensemble prediction on the preprocessed test split
├── model/          # IFCleave model architecture
├── train/          # K-fold cross-validation training
├── eval/           # Window-based evaluation
├── data/           # Data pipeline (IEDB → PDB → features → windows) + index.csv
├── utils/          # Shared utilities (metrics, dataset, standardization)
└── checkpoints/    # 4-fold pre-trained weights
```

