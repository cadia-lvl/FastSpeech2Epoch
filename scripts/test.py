import mat73
import tgt
import numpy as np
import tqdm
import glob
import os
import matplotlib.pyplot as plt

import librosa

# __import__('ipdb').set_trace()

def get_alignment(textgrid_path):
    
    tier = tgt.io.read_textgrid(textgrid_path).get_tier_by_name("phones")
    
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    start_time = 0
    end_time = 0

    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text
        
        if p in sil_phones:
            p = 'sp'
        
        phones.append((p, s, e))

    return phones

def read_mat(mat_path):
    # Load the .mat file
    mat_data = mat73.loadmat(mat_path)

    filename = mat_data['name']
    mel = mat_data['melbm']
    phase = mat_data['diffmelbu']
    frames_t = mat_data['frames_t']
    
    return (filename, mel, phase, frames_t)

def plot_spectrogram(specgram, save_path, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")

    plt.savefig(save_path)  # Save the figure to the specified path
    plt.close()  # Close the figure after saving to free up memory

epochdur_file = '/work/shijun/ljspeech/epoch_processed/epoch_dur/LJSpeech-epochdur-LJ032-0248.npy'
origin_mat_file = '/work/shijun/ljspeech/epoch_raw/LJSpeech/LJ032-0248.mat'
tg_file = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/TextGrid/TextGrid/LJSpeech/LJ032-0248.TextGrid'

epochdur = np.load(epochdur_file)
_, mel, phase, frames_t = read_mat(origin_mat_file)
phonemes = get_alignment(tg_file)
# print(mel.shape)
# print(mel)
plot_spectrogram(mel, save_path='/home/shijun/epoch_project/FastSpeech2_Epoch/mel.png')
plot_spectrogram(phase, save_path='/home/shijun/epoch_project/FastSpeech2_Epoch/phase.png')

# print(epochdur)
# print(phonemes)
# print(frames_t[:100])

