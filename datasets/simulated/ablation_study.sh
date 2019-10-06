# ############################################################################
# ablation_study.sh
# =================
# Author : Sepand KASHANI [sepand.kashani@epfl.ch]
# ############################################################################

# Run ablation study on network parameter.

source activate DeepWave
python3 ./split_dataset.py --dataset="./dataset/D.npz" --out="./dataset/D_idx_split.npz" --tv_ratio=0.2 --seed=0

dataset="./dataset/D.npz"
D_lambda=0.1
tau_lambda=0.1
mu=0.9
N_layer=5
psf_threshold=0.00000100
tanh_lin_limit=1.0
tv_index="./dataset/D_idx_split.npz"
learning_rate='1e-07'
N_epoch=5
batch_size=200
seed=0

python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_000.npz"                            --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_001.npz"                  --fix_tau --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_010.npz"          --fix_D           --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_011.npz"          --fix_D --fix_tau --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_100.npz" --fix_mu                   --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_101.npz" --fix_mu         --fix_tau --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_110.npz" --fix_mu --fix_D           --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch="${N_epoch}" --batch_size="${batch_size}" --seed="${seed}";
python3 ./../../scripts/train_crnn.py --dataset="${dataset}" --parameter="./dataset/D_train_111.npz" --fix_mu --fix_D --fix_tau --D_lambda="${D_lambda}" --tau_lambda="${tau_lambda}" --mu="${mu}" --N_layer="${N_layer}" --psf_threshold="${psf_threshold}" --tanh_lin_limit="${tanh_lin_limit}" --loss=relative-l2 --tv_index="${tv_index}" --lr="${learning_rate}" --N_epoch=1            --batch_size="${batch_size}" --seed="${seed}";

python3 ./export_ablation_results.py
cat ./dataset/ablation_study.csv
