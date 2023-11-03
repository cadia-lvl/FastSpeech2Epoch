import torch
import torch.nn as nn
import numpy as np
import os
import shutil

epolen_bins = nn.Parameter(
            torch.linspace(0.0024999999999995026, 0.02400000000000002, 257 - 1),
            requires_grad=False,
)

# print(torch.bucketize(torch.tensor([0.0024]), epolen_bins))
# exit()

epolen_bins = epolen_bins.cpu().numpy()
epolen_bins = {idx:bin for idx, bin in zip(range(len(epolen_bins)), epolen_bins)}

# print(epolen_bins)

epolen_pred = np.load('/work/shijun/FastSpeechEpochResultBucket/synthesis/600000/predict_epolen.npy')
epolen_pred = [epolen_bins[p] for p in epolen_pred]

frames_t = [epolen_pred[0]]
for p in epolen_pred[1:]:
    frames_t.append(frames_t[-1]+p)
    
frames_t = np.array(frames_t)
np.save(os.path.join('for_reconvert', 'frames_t.npy'), frames_t)

shutil.copy('/work/shijun/FastSpeechEpochResultBucket/synthesis/600000/predict_mel.npy', os.path.join('for_reconvert', 'melbm.npy'))
shutil.copy('/work/shijun/FastSpeechEpochResultBucket/synthesis/600000/predict_phase.npy', os.path.join('for_reconvert', 'melbu.npy'))