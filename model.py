import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_dim=80, emb_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(mel_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, mel):  # mel: [B, 80, T]
        x = self.conv(mel)  # [B, 128, 1]
        x = x.squeeze(-1)
        return self.fc(x)   # [B, emb_dim]

class MelDecoder(nn.Module):
    def __init__(self, emb_dim=128, mel_dim=80):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, mel_dim)
        )

    def forward(self, speaker_emb, T):
        # Repeat for each frame (T) during training
        expanded = speaker_emb.unsqueeze(2).repeat(1, 1, T)  # [B, emb_dim, T]
        expanded = expanded.transpose(1, 2)  # [B, T, emb_dim]
        out = self.fc(expanded)  # [B, T, 80]
        return out.permute(0, 2, 1)  # [B, 80, T]
