import mat73
import numpy as np
import tqdm
import glob
import os




mat_root_path = '/work/shijun/ljspeech/epoch_processed3/epoch_len/'

min_phase = float('inf')
max_phase = float('-inf')

for mat_path in tqdm.tqdm(glob.glob(os.path.join(mat_root_path, "*.npy"))):
    phase = np.load(mat_path)* 1000
    # phase = phase*1000
    min_phase = min(np.min(phase), min_phase)
    max_phase = max(np.max(phase), max_phase)


print(min_phase)
print(max_phase)