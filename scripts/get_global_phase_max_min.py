import mat73
import numpy as np
import tqdm
import glob
import os

def read_mat(mat_path):
    # Load the .mat file
    mat_data = mat73.loadmat(mat_path)


    filename = mat_data['name']
    mel = mat_data['melbm']
    phase = mat_data['melbu']
    frames_t = mat_data['frames_t']
    
    return phase

mat_root_path = '/work/shijun/ljspeech/epoch_raw3/LJSpeech_raw'

min_phase = float('inf')
max_phase = float('-inf')

for mat_path in tqdm.tqdm(glob.glob(os.path.join(mat_root_path, "*.mat"))):
    phase = read_mat(mat_path)
    min_phase = min(np.min(phase), min_phase)
    max_phase = max(np.max(phase), max_phase)


print(min_phase)
print(max_phase)