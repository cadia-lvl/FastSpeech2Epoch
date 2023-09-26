import mat73
import tgt
import numpy as np
import tqdm
import glob
import os
import argparse

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


def generate_epochdur(phonemes, frames_t):
    result_dur = []
    current_epoch_start_idx, current_epoch_start = 0, 0.0
    for current_phoneme_idx, (current_p, current_p_start, current_p_end) in enumerate(phonemes):
        epoch_step = 1
        while True:
            current_epoch_end_idx = current_epoch_start_idx + epoch_step
            current_epoch_end = frames_t[current_epoch_end_idx]
            if current_p_start <= current_epoch_start and current_p_end > current_epoch_end:
                epoch_step += 1
            else:
                result_dur.append(current_epoch_end_idx-current_epoch_start_idx)
                current_epoch_start_idx = current_epoch_end_idx
                current_epoch_start = current_epoch_end
                break

    return 0, current_epoch_end_idx, np.array(result_dur)

def convert_epoch_len(frames_t):
    current_start = 0
    result = []
    for frame in frames_t:
        result.append(frame - current_start)
        current_start = frame
        
    return np.array(result)

def main(args):
    
    mat_root_path = args.mat_path
    tg_root_path = args.tgt_path
    save_root_path = args.save_path

    mel_save_root = os.path.join(save_root_path, 'mel')
    phase_save_root = os.path.join(save_root_path, 'phase')
    epochdur_save_root = os.path.join(save_root_path, 'epoch_dur')
    epochlen_save_root = os.path.join(save_root_path, 'epoch_len')

    os.makedirs(mel_save_root, exist_ok=True)
    os.makedirs(phase_save_root, exist_ok=True)
    os.makedirs(epochdur_save_root, exist_ok=True)
    os.makedirs(epochlen_save_root, exist_ok=True)

    for tg_path in tqdm.tqdm(glob.glob(os.path.join(tg_root_path, "*.TextGrid"))):
        
        base_name = os.path.basename(tg_path).split('.TextGrid')[0]
        
        mat_path = os.path.join(mat_root_path, base_name+'.mat')
        
        phonemes = get_alignment(tg_path)
        _, mel, phase, frames_t = read_mat(mat_path)
        
        epoch_start_idx, epoch_end_idx, epoch_durs = generate_epochdur(phonemes, frames_t)
        total_len = np.sum(epoch_durs)
        
        assert epoch_end_idx - epoch_start_idx == total_len

        mel = mel[:, epoch_start_idx:epoch_end_idx]
        phase = phase[:, epoch_start_idx:epoch_end_idx]
        frames_t = frames_t[epoch_start_idx:epoch_end_idx]
        
        epoch_lengths = convert_epoch_len(frames_t)
        
        np.save(os.path.join(mel_save_root, f'LJSpeech-mel-{base_name}.npy'), mel)
        np.save(os.path.join(phase_save_root, f'LJSpeech-phase-{base_name}.npy'), phase)
        np.save(os.path.join(epochdur_save_root, f'LJSpeech-epochdur-{base_name}.npy'), epoch_durs)
        np.save(os.path.join(epochlen_save_root, f'LJSpeech-epochlen-{base_name}.npy'), epoch_lengths)
        
if __name__ == '__main__':
    
    '''
        This script will create four new folders based on the mat files.
        mel: mel files
        phase: phase files
        epoch_dur: durtion files
        epoch_len: durtion length files.
    '''
    
    parser = argparse.ArgumentParser(description='Convert the mat files into necessary numpy files.')
    parser.add_argument('--mat_path', type=str, help='Path to the mat folder.')
    parser.add_argument('--tgt_path', type=str, help='Path to the MFA textgrid folder.')
    parser.add_argument('--save_path', type=str, help='Where do you want to save')
    
    args = parser.parse_args()
    main(args)