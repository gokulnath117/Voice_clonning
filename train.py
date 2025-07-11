from data_loader import MelDataset, collate_fn
from model import SpeakerEncoder, MelDecoder
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

def composite_loss(pred, target, lambda_cos=0.01):
    # pred, target: [B, 80, T]
    
    # Mean Squared Error
    mse = F.mse_loss(pred, target)
    
    # Cosine Similarity Loss
    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1)  # [B]
    cosine_loss = 1 - cosine_sim.mean()  # Higher when vectors misalign

    return mse + lambda_cos * cosine_loss

def evaluate(model_enc, model_dec, loader):
    model_enc.eval()
    model_dec.eval()
    total_loss = 0
    with torch.no_grad():
        for mel, _ in loader:
            emb = model_enc(mel)
            pred = model_dec(emb, mel.shape[2])
            loss = composite_loss(pred, mel)
            total_loss += loss.item()
    return total_loss / len(loader)


train_dataset = MelDataset('data/mel_train.csv')
val_dataset = MelDataset('data/mel_val.csv')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

encoder = SpeakerEncoder()
decoder = MelDecoder()
opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-3)

for epoch in range(50):
    total_loss = 0
    for mel, speaker_id in train_loader:
        # mel: [B, 80, T]
        emb = encoder(mel)  # [B, emb_dim]
        pred = decoder(emb, mel.shape[2])  # [B, 80, T]
        loss = composite_loss(pred, mel)  # consistent shape
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        val_loss = evaluate(encoder, decoder, val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")


torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")