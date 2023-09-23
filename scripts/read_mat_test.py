import mat73
import tgt
import numpy as np

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

mat_path = '/work/shijun/ljspeech/epoch_raw/LJSpeech/LJ001-0001.mat'
tg_path = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/TextGrid/TextGrid/LJSpeech/LJ001-0001.TextGrid'

phones = get_alignment(tg_path)
print(phones)

_, mel, _, frames_t = read_mat(mat_path)

phoneme2epoch = {}
current_epoch_start_idx, current_epoch_start = 0, 0.0
for current_phoneme_idx, (current_p, current_p_start, current_p_end) in enumerate(phones):
    epoch_step = 1
    while True:
        current_epoch_end_idx = current_epoch_start_idx + epoch_step
        current_epoch_end = frames_t[current_epoch_end_idx]
        if current_p_start <= current_epoch_start and current_p_end > current_epoch_end:
            epoch_step += 1
        else:
            phoneme2epoch[current_phoneme_idx] = {'phoneme': current_p, 
                                                'p_start': current_p_start,
                                                'p_end': current_p_end,
                                                'epoch_start_idx': current_epoch_start_idx,
                                                'epoch_end_idx': current_epoch_end_idx-1,
                                                'epoch_start': current_epoch_start,
                                                'epoch_end': frames_t[current_epoch_end_idx-1]}
            current_epoch_start_idx = current_epoch_end_idx
            current_epoch_start = current_epoch_end
            break

print(phoneme2epoch[len(phones)-1]['epoch_end_idx'])
print(mel.shape)