import numpy as np

duration_path = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/duration/LJSpeech-duration-LJ002-0321.npy'
energy_path = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/energy/LJSpeech-energy-LJ002-0321.npy'
mel_path = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/mel/LJSpeech-mel-LJ002-0321.npy'
pitch_path = '/home/shijun/epoch_project/FastSpeech2/preprocessed_data/LJSpeech/pitch/LJSpeech-pitch-LJ002-0321.npy'

phoneme = 'P R IH1 N T IH0 NG sp IH1 N DH IY0 OW1 N L IY0 S EH1 N S W IH1 DH sp W IH1 CH W IY1 AA1 R AE1 T P R EH1 Z AH0 N T K AH0 N S ER1 N D sp D IH1 F ER0 Z sp F R AH1 M M OW1 S T IH1 F N AA1 T F R AH1 M AO1 L DH IY0 AA1 R T S AH0 N D K R AE1 F T S R EH2 P R IH0 Z EH1 N T IH0 D IH1 N DH IY0 EH2 K S AH0 B IH1 SH AH0 N'

duration = np.load(duration_path)
energy = np.load(energy_path)
mel = np.load(mel_path)
pitch = np.load(pitch_path)

print(len(phoneme.split()))
print(duration.shape)
print(energy.shape)
print(mel.shape)
print(pitch.shape)