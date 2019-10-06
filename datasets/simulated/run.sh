#!/bin/sh

# Generate training data
source activate DeepWave
for N_src in 1 2 3 4 5 6 7 8 9 10; do
    python3 ./simulate.py --dataset=./dataset/D_${N_src}.npz --N_sample=2000 --N_src=${N_src} --intensity;
done
python3 ./../../scripts/merge_dataset.py --out=./dataset/D.npz ./dataset/D_*.npz;

# Train DeepWave
python3 ./../../scripts/train_crnn.py --dataset=./dataset/D.npz --parameter=./dataset/D_train.npz --D_lambda=0.10000000 --tau_lambda=0.10000000 --mu=0.9 --N_layer=5 --psf_threshold=0.00000100 --tanh_lin_limit=1.00000000 --loss=relative-l2 --tv_ratio=0.20000000 --lr=1e-07 --N_epoch=10 --batch_size=200 --seed=0;
