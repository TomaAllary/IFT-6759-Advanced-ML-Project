import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
import wandb
import os

from self_attention_pooling import SelfAttentionPooling

# Emotion Label Mapping
emotion2idx = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

# Transformer Decoder model for autoregressive prediction of next utterance embeddings
class UtteranceDecoder(nn.Module):
    def __init__(self, input_dim=1536, model_dim=1536, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        # Self attention pooling layer
        self.pooler = SelfAttentionPooling(input_dim)


        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 128, model_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


    def generate_square_subsequent_mask(self, sz):
        # Mask future positions (upper triangular mask with -inf)
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        # x: [batch, dialogue_len, utterance_len, 1536]
        batch_size, dialogue_len, utterance_len, input_dim = x.shape

        # 1. Pool across utterance dimension
        pooled = []
        for i in range(dialogue_len):
            pooled_utt = self.pooler(x[:, i, :, :])  # [batch, 1536]
            pooled.append(pooled_utt)

        # 2. Stack pooled utterances → [batch, dialogue_len, 1536]
        pooled_seq = torch.stack(pooled, dim=1)
        memory = pooled_seq

        # Add positional encoding (truncate to current T)
        pos_embed = self.pos_embedding[:, :memory.size(1), :]  # [1, T, model_dim]
        memory = memory + pos_embed

        outputs = []
        for t in range(1, dialogue_len):
            tgt = pooled_seq[:, :t, :]                # [batch, t, 1536]
            pos_tgt = self.pos_embedding[:, :tgt.size(1), :]
            tgt = tgt + pos_tgt
            tgt_mask = self.generate_square_subsequent_mask(t).to(x.device)  # [t, t]

            decoded = self.decoder(tgt, memory, tgt_mask=tgt_mask)           # [batch, t, model_dim]
            pred = decoded[:, -1, :]  # [batch, 1536]
            outputs.append(pred)

        return torch.stack(outputs, dim=1)  # [batch, dialogue_len - 1, 1536]

# Dataset for utterance embeddings grouped by dialogue
class DialogueDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row["Features"]  # List of tensors (one per utterance)
        emotion_labels = row["Emotion"]

        # Ensure all features are tensors
        utterance_tensors = []
        for f in features:
            if not isinstance(f, torch.Tensor):
                f = torch.tensor(f, dtype=torch.float32)
            utterance_tensors.append(f)

        label_ids = torch.tensor([emotion2idx[e.lower()] for e in emotion_labels], dtype=torch.long)
        return utterance_tensors, label_ids


# Collate function to pad variable-length dialogue sequences
def collate_fn(batch):
    # batch: list of (list of utterance tensors, label tensor)
    dialogue_inputs, dialogue_targets = zip(*batch)

    padded_dialogues = []
    max_utt_len = max(max(utt.size(0) for utt in dialog) for dialog in dialogue_inputs)

    for dialog in dialogue_inputs:
        padded_utterances = []
        for utt in dialog:
            # Pad each utterance to max_utt_len
            if utt.size(0) < max_utt_len:
                pad = torch.zeros(max_utt_len - utt.size(0), utt.size(1))
                utt = torch.cat([utt, pad], dim=0)
            padded_utterances.append(utt)

        # Stack utterances and pad dialogues
        padded_utterances = torch.stack(padded_utterances)  # [num_utts, max_utt_len, 1536]
        padded_dialogues.append(padded_utterances)

    # Pad to max_dialogue_len
    max_dialogue_len = max([d.size(0) for d in padded_dialogues])
    for i in range(len(padded_dialogues)):
        if padded_dialogues[i].size(0) < max_dialogue_len:
            pad = torch.zeros(max_dialogue_len - padded_dialogues[i].size(0), max_utt_len, 1536)
            padded_dialogues[i] = torch.cat([padded_dialogues[i], pad], dim=0)

    # Final batch shape: [batch_size, max_dialogue_len, max_utt_len, 1536]
    batch_inputs = torch.stack(padded_dialogues)

    # Pad target sequences with -1 for ignored label loss
    max_target_len = max([len(t) for t in dialogue_targets])
    padded_targets = torch.full((len(dialogue_targets), max_target_len), fill_value=-1, dtype=torch.long)
    for i, t in enumerate(dialogue_targets):
        padded_targets[i, :len(t)] = t

    return batch_inputs, padded_targets


# Cosine similarity loss between predicted and target embeddings
def cosine_similarity_loss(pred_embs, target_embs):
    pred_proj = F.normalize(pred_embs, dim=-1)
    tgt_proj = F.normalize(target_embs, dim=-1)
    loss = 1 - F.cosine_similarity(pred_proj, tgt_proj, dim=-1).mean()
    return loss

if __name__ == "__main__":
    # Initialize wandb logging
    wandb.init(project="utterance-prediction-transformer", config={
        "input_dim": 1536,
        "model_dim": 1536,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.3,
        "lr": 1e-4,
        "epochs": 50,
        "batch_size": 8,
        "patience": 5,
    })
    config = wandb.config

    # Load dialogue-level features
    df = pd.read_pickle("features_per_dialog.pkl")
    dataset = DialogueDataset(df)

    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Instantiate model
    model = UtteranceDecoder(
        input_dim=config.input_dim,
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_total_loss = 0
        for x, _ in train_loader:
            if x.size(1) < 2:
                continue
            x = x.to(device)
            optimizer.zero_grad()
            pred_embs = model(x)  # Predict next utterance embeddings
            
            # Pool ground-truth embeddings
            target_embs = []
            for i in range(1, x.size(1)):  # for each t+1 step
                pooled = model.pooler(x[:, i, :, :])  # x[:, i] = [batch, utterance_len, 1536]
                target_embs.append(pooled)

            target_embs = torch.stack(target_embs, dim=1)  # [batch, dialogue_len - 1, 1536]

            loss = cosine_similarity_loss(pred_embs, target_embs)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item() * x.size(0)

        avg_train_loss = train_total_loss / len(train_dataset)

        # Validation loop
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                if x.size(1) < 2:
                    continue
                x = x.to(device)
                pred_embs = model(x)

                # Pool ground-truth embeddings
                target_embs = []
                for i in range(1, x.size(1)):  # for each t+1 step
                    pooled = model.pooler(x[:, i, :, :])  # x[:, i] = [batch, utterance_len, 1536]
                    target_embs.append(pooled)

                target_embs = torch.stack(target_embs, dim=1)  # [batch, dialogue_len - 1, 1536]

                loss = cosine_similarity_loss(pred_embs, target_embs)
                val_total_loss += loss.item() * x.size(0)

        avg_val_loss = val_total_loss / len(val_dataset)

        # Logging
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({"decoder": model.state_dict()}, "best_utterance_decoder.pt")

        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    wandb.finish()
