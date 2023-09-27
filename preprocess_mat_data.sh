#!/bin/bash
python preprocess_mat_data.py --mat_path=/work/shijun/ljspeech/epoch_raw2/LJSpeech \
    --tgt_path=/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/TextGrid/TextGrid/LJSpeech \
    --raw_transcrption_path=/work/shijun/ljspeech/LJSpeech-1.1/metadata.csv \
    --save_path=/work/shijun/ljspeech/epoch_processed2