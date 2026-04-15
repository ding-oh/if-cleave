# IF-Cleave

Inverse folding-based deep learning for MHC-II antigen processing cleavage site prediction.

IF-Cleave combines ESM-IF1 inverse folding embeddings (512D) with PROPKA physicochemical features (6D) to predict where endosomal proteases cleave antigen proteins for MHC class II presentation — a critical step in CD4+ T cell immune responses.

## Installation

```bash
conda create -n ifcleave python=3.10
conda activate ifcleave
pip install -r requirements.txt
```

## Data Pipeline

```bash
# 1. Build PDB database from IEDB/UniProt/RCSB
python data/build_cleavage_db.py

# 2. Extract IF1 + PROPKA features
python data/build_if1_propka_dataset.py

# 3. Prepare window-based training data
python data/prepare_data.py --window_size 11
```

## Training

```bash
# 4-fold cross-validation
python train/train_kfold.py \
    --model bilstm --hidden_dim 256 --dropout 0.4 \
    --epochs 500 --batch_size 32 --lr 0.001 --weight_decay 0.005 \
    --loss bce --patience 20 --n_folds 4 --seed 42 \
    --label_smoothing 0.05 --grad_clip 1.0 \
    --data_dir data_if1_w11 --output_dir results
```

## Evaluation

```bash
# Window-based metrics
python eval/evaluate.py --results_dir results --data_dir data_if1_w11

# CD4+ epitope boundary validation
python eval/validate_cd4.py --target all --device cuda

# Publication figures
python eval/generate_figures.py
```

## Project Structure

```
if-cleave/
├── model/          # BiLSTMGNN architecture + feature extraction
├── train/          # K-fold cross-validation training
├── eval/           # Evaluation, biological validation, figures
├── data/           # Data pipeline (IEDB → PDB → features → windows)
├── utils/          # Shared utilities (metrics, dataset, feature ablation)
├── configs/        # Training configuration
└── scripts/        # Shell scripts for ablation studies
```

## Citation

```bibtex
@article{ifcleave2026,
  title={IF-Cleave: Inverse Folding-Based Deep Learning for MHC-II Antigen Processing Cleavage Site Prediction},
  journal={Bioinformatics},
  year={2026}
}
```
