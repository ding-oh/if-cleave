PKL ?= all_datasets_fixed.pkl
WINDOW ?= 11
RESULTS ?= results

.PHONY: reproduce train clean

data_if1_w%:
	python data/prepare_data.py --window_size $* --pkl_file $(PKL)

reproduce: data_if1_w$(WINDOW) data_if1_w1
	python reproduce.py --data_dir data_if1_w$(WINDOW) --output $(RESULTS)/bilstm_predictions.npz
	python eval/evaluate.py --pred_file $(RESULTS)/bilstm_predictions.npz --data_dir_w1 data_if1_w1

train: data_if1_w$(WINDOW) data_if1_w1
	python train/train.py \
	    --hidden_dim 256 --dropout 0.4 \
	    --epochs 500 --batch_size 32 --lr 0.001 --weight_decay 0.005 \
	    --patience 20 --n_folds 4 --seed 42 \
	    --label_smoothing 0.05 --grad_clip 1.0 \
	    --data_dir data_if1_w$(WINDOW) --output_dir $(RESULTS)
	python eval/evaluate.py --pred_file $(RESULTS)/bilstm_predictions.npz --data_dir_w1 data_if1_w1

clean:
	rm -rf $(RESULTS) data_if1_w*
