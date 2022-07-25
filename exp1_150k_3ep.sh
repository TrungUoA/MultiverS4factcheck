##!/bin/bash

#source venv/bin/activate

export CUDA_VISIBLE_DEVICES=6

export EXP=1_150k_3ep

mkdir /home/tugn232/projects/experiments/multivers/$EXP

python longchecker/train.py --starting_checkpoint checkpoints/fever_sci.ckpt --train_file data/train.csv --test_file data/test.csv --batch_size 4 --accelerator gpu --device 1 --max_epochs 3 --result_dir /home/tugn232/projects/experiments/multivers/1_150k_3ep/lightning_logs --experiment_name multivers1_150k_3ep --mydata 1
