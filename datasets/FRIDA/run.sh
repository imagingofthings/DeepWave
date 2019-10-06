#!/bin/sh

# Download FRIDA dataset
git clone git@github.com:LCAV/FRIDA.git
cd FRIDA
wget https://zenodo.org/record/345132/files/FRIDA_recordings.tar.gz
tar xzfv FRIDA_recordings.tar.gz

# Extract training data
cd ..
source activate DeepWave
for folder in one_speaker two_speakers three_speakers; do
    for file in ./FRIDA/recordings/20160908/data_pyramic/segmented/${folder}/*.wav; do
        python3 ./extract_dataset.py --data="${file}";
    done
done
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    python3 ./../../scripts/merge_dataset.py --out=./dataset/D_freq${idx_freq}.npz ./dataset/D_*_freq${idx_freq}_cold.npz;
done

# Train DeepWave
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    python3 ./../../scripts/train_crnn.py --dataset="./dataset/D_freq${idx_freq}.npz" --parameter="./dataset/D_freq${idx_freq}_train.npz" --D_lambda=0.10000000 --tau_lambda=0.10000000 --mu=0.9 --N_layer=5 --psf_threshold=0.00000100 --tanh_lin_limit=1.00000000 --loss=relative-l2 --tv_ratio=0.20000000 --lr=1e-08 --N_epoch=10 --batch_size=100 --seed=0;
done
