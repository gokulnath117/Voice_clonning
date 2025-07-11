import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MelDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.mel_data = []
        self.speaker_ids = []

        for _, row in df.iterrows():
            mel = np.array([float(x) for x in row['mel_flat'].strip().split()])
            mel = mel.reshape(80, -1)  # [80, T]
            mel = mel / (mel.max() + 1e-6)
            self.mel_data.append(torch.tensor(mel, dtype=torch.float32))
            self.speaker_ids.append(int(row['speaker_id']))

    def __len__(self):
        return len(self.mel_data)

    def __getitem__(self, idx):
        return self.mel_data[idx], self.speaker_ids[idx]

def collate_fn(batch):
    mels, speakers = zip(*batch)
    max_len = max(mel.shape[1] for mel in mels)

    padded_mels = []
    for mel in mels:
        pad_len = max_len - mel.shape[1]
        padded = torch.nn.functional.pad(mel, (0, pad_len), mode='constant', value=0.0)
        padded_mels.append(padded)

    padded_mels = torch.stack(padded_mels)     # [B, 80, T_max]
    speaker_ids = torch.tensor(speakers)       # [B]

    return padded_mels, speaker_ids