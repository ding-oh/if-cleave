# IF-Cleave

Inverse folding-based deep learning for MHC-II antigen processing cleavage site prediction.

IF-Cleave combines ESM-IF1 inverse folding embeddings (512D) with PROPKA physicochemical features (6D) to predict where endosomal proteases cleave antigen proteins for MHC class II presentation — a critical step in CD4+ T cell immune responses.

## Installation

```bash
conda create -n ifcleave python=3.10
conda activate ifcleave

# Install PyTorch matching your CUDA driver version (check with nvidia-smi)
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Data Pipeline

```bash
# 1. Build PDB database from IEDB/UniProt/RCSB
python data/build_db.py

# 2. Extract IF1 + PROPKA features
python data/extract_features.py

# 3. Prepare window-based training data (w=11 for training, w=1 for evaluation ground truth)
python data/prepare_data.py --window_size 11
python data/prepare_data.py --window_size 1
```

## Training

```bash
# 4-fold cross-validation
python train/train.py \
    --hidden_dim 256 --dropout 0.4 \
    --epochs 500 --batch_size 32 --lr 0.001 --weight_decay 0.005 \
    --patience 20 --n_folds 4 --seed 42 \
    --label_smoothing 0.05 --grad_clip 1.0 \
    --data_dir data_if1_w11 --output_dir results
```

## Inference (reproduce paper numbers)

Pre-trained 4-fold checkpoints are in `checkpoints/` (ensemble test MCC 0.260).

```bash
# 1. Produce ensemble predictions (applies per-fold standardization)
python eval/predict.py --data_dir data_if1_w11 --output results/bilstm_predictions.npz

# 2. Window-based metrics
python eval/evaluate.py --pred_file results/bilstm_predictions.npz --data_dir_w1 data_if1_w1
```

## Project Structure

```
if-cleave/
├── model/          # IFCleave model architecture
├── train/          # K-fold cross-validation training
├── eval/           # Inference (predict.py) and window-based evaluation
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
