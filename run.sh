#!/bin/bash

# python train_3d_synth.py --exp_name finetune-noise500-no_edge-balanced --gpu 3 --preprocessed_path molecule_synth_data/sampled_data-noised-500.pt --no_edge_index --checkpoint 3d_synth_noised_log/exp_crsd_noise500-20230407-0622/best_checkpoint.pt --fine_tune --batch_size 8 --lr 1e-5 --weight_decay 1e-3 --early_stop_threshold 15 --num_epochs 15 --ft_data_path molecule_synth_data/crossdocked100k_balanced-noised-500.pt

# python train_3d_synth.py --exp_name test-finetune-noise200-no_edge-balanced --gpu 3 --preprocessed_path molecule_synth_data/crossdocked100k_balanced.pt --no_edge_index --checkpoint 3d_synth_noised_log/finetune-noise200-no_edge-balanced-20230408-1017/best_checkpoint.pt --batch_size 32 --lr 1e-4 --weight_decay 0 --eval

python train_3d_synth.py --exp_name test --gpu 0 --preprocessed_path molecule_synth_data/crossdocked100k_balanced.pt --checkpoint 3d_synth_noised_log/exp_crsd_balanced-20230420-0719/best_checkpoint.pt --eval --no_edge_index 