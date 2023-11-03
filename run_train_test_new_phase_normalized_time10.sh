#!/bin/bash
#SBATCH --account=staff
#SBATCH --job-name=run_train_test_new_phase_normalized_time10
#SBATCH --cpus-per-task=3         # CPU cores/threads
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --output=run_train_test_new_phase_normalized_time10.log
#SBATCH --partition=lvlWork
bash /home/shijun/epoch_project/FastSpeech2Epoch/train.sh