#!/bin/bash
python train.py -p /home/shijun/epoch_project/FastSpeech2Epoch/config/LJSpeech_epoch/preprocess.yaml \
    -m /home/shijun/epoch_project/FastSpeech2Epoch/config/LJSpeech_epoch/model.yaml \
    -t /home/shijun/epoch_project/FastSpeech2Epoch/config/LJSpeech_epoch/train.yaml \
    --checkpoint_path /work/shijun/FastSpeechEpochResultBucket/checkpoint/80000.ckpt

