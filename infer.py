# infer_reference.py

import torch
from model import SpeakerEncoder, MelDecoder
from data_loader import MelDataset
import os

encoder = SpeakerEncoder()
decoder = MelDecoder()

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))

encoder.eval()
decoder.eval()

reference_dataset = MelDataset('data/mel_reference.csv')

os.makedirs("cloned_outputs", exist_ok=True)

for idx in range(len(reference_dataset)):
    mel, speaker_id = reference_dataset[idx]
    mel = mel.unsqueeze(0)  # [1, 80, T]
    with torch.no_grad():
        emb = encoder(mel)
        pred = decoder(emb, mel.shape[2])  # [1, 80, T]
        output_path = f"cloned_outputs/speaker_{speaker_id}_mel.pt"
        torch.save(pred.squeeze(0), output_path)
        print(f"Saved: {output_path}")