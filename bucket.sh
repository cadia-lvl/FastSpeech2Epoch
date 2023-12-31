#!/bin/bash
#SBATCH --account=staff
#SBATCH --job-name=bucket
#SBATCH --cpus-per-task=3         # CPU cores/threads
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus-per-node=1
#SBATCH --output=bucket.log
#SBATCH --partition=lvlWork
bash /home/shijun/epoch_project/FastSpeech2Epoch/train.sh