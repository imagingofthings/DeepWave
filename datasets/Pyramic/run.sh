#!/bin/sh

# Download Pyramic dataset
git clone git@github.com:fakufaku/pyramic-dataset.git
cd pyramic-dataset
./download_segmented.sh

# Extract training data
cd ..
source activate DeepWave
python3 ./extract_dataset.py --N_src=1 --N_sample=-1 --seed=0
python3 ./extract_dataset.py --N_src=3 --N_sample=3000 --seed=1
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    python3 ./../../scripts/merge_dataset.py --out=./dataset/D_freq${idx_freq}.npz ./dataset/D_freq${idx_freq}_NSRC_*_NSAMPLE_*_sample_*.npz;
done
rm ./dataset/D_freq*_NSRC_*_NSAMPLE_*_sample_*.npz

# Train DeepWave (held-out directions)
# The split is the same across frequencies.
python3 ./split_dataset.py --dataset="./dataset/D_freq4.npz" --out="./dataset/D_idx_split.npz" --tv_ratio=0.2 --test_only_angles='list(range(0, 358, 2))[::18]'
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    python3 ./../../scripts/train_crnn.py --dataset="./dataset/D_freq${idx_freq}.npz" --parameter="./dataset/D_freq${idx_freq}_train.npz" --D_lambda=0.10000000 --tau_lambda=0.10000000 --mu=0.9 --N_layer=5 --psf_threshold=0.00000100 --tanh_lin_limit=1.00000000 --loss=relative-l2 --tv_index="./dataset/D_idx_split.npz" --lr=1e-08 --N_epoch=5 --batch_size=100 --seed=0;
done

# Extract test-only images.
python3 ./extract_dataset.py --N_src=3 --angles='list(range(0, 358, 2))[::18][:5]' --N_sample=200 --seed=1
python3 ./extract_simulated_dataset.py
for idx_freq in 0 1 2 3 4 5 6 7 8; do
    python3 ./../../scripts/merge_dataset.py --out=./dataset/D_freq${idx_freq}_plot.npz ./dataset/D_freq${idx_freq}_NSRC_*_NSAMPLE_*_sample_*.npz ./dataset/D_freq${idx_freq}_sim.npz;
done
rm ./dataset/D_freq*_NSRC_*_NSAMPLE_*_sample_*.npz
rm ./dataset/D_freq*_sim.npz

conda install basemap
python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=DAS  --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_full_sky --show_catalog --lon_ticks='np.linspace(-180, 180, 5)'
python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=RNN  --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_full_sky --show_catalog --lon_ticks='np.linspace(-180, 180, 5)'
python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=APGD --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_full_sky --show_catalog --lon_ticks='np.linspace(-180, 180, 5)'
# python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=DAS  --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_part_sky --show_catalog --lon_ticks='np.linspace(-20, 150, 3)'
# python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=RNN  --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_part_sky --show_catalog --lon_ticks='np.linspace(-20, 150, 3)'
# python3 color_plot.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=APGD --img_idx='np.arange(4141)' --mode=disk --out=./dataset/color_plots_part_sky --show_catalog --lon_ticks='np.linspace(-20, 150, 3)'


# Raw RGB export (for diagnostics)
python3 export_rgb.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=DAS  --img_idx='np.arange(4141)' --out='./dataset/I_das_rgb.npz' --lon_ticks='np.linspace(-180, 180, 5)'
python3 export_rgb.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=RNN  --img_idx='np.arange(4141)' --out='./dataset/I_rnn_rgb.npz' --lon_ticks='np.linspace(-180, 180, 5)'
python3 export_rgb.py --datasets ./dataset/D_freq[0-8]_plot.npz --parameters ./dataset/D_freq[0-8]_train.npz --img_type=APGD --img_idx='np.arange(4141)' --out='./dataset/I_apgd_rgb.npz' --lon_ticks='np.linspace(-180, 180, 5)'
