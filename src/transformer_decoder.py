import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import wandb
import os

# Initialize Weights & Biases
wandb.init(project="utterance-prediction-transformer", config={
    "input_dim": 1536,
    "model_dim": 512,
    "num_heads": 8,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-4,
    "epochs": 50,
    "batch_size": 2,
    "patience": 5
})
config = wandb.config

# Decoder-Only Transformer to Predict the Next Utterance Embedding
class UtterancePredictor(nn.Module):
    def __init__(self, input_dim=1536, model_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super(UtterancePredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, memory):
        memory = self.input_proj(memory).transpose(0, 1)  # (seq_len, batch, model_dim)
        target = self.input_proj(memory[-1:])  # last utterance as decoder input
        output = self.transformer_decoder(target, memory)
        output = self.output_proj(output)
        return output.squeeze(0)  # (batch, input_dim)

# Custom Dataset for Utterance Sequences
class DialogueDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row["Features"]
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)
        return features

# Collate function to stack utterance sequences (excluding last one as target)
def collate_fn(batch):
    memory = []
    target = []
    for dialog in batch:
        if dialog.shape[1] < 2:
            continue
        memory.append(dialog[:, :-1, :].squeeze(0))  # (seq-1, 1536)
        target.append(dialog[:, -1, :].squeeze(0))   # (1536)

    memory = torch.nn.utils.rnn.pad_sequence(memory, batch_first=True)  # (batch, seq_len, 1536)
    target = torch.stack(target)  # (batch, 1536)
    return memory, target

# Load Pickle File
df_features = pd.read_pickle("features_per_sentence.pkl")
dataset = DialogueDataset(df_features)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

# Model Init
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = UtterancePredictor(
    input_dim=config.input_dim,
    model_dim=config.model_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    dropout=config.dropout
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.lr)
num_epochs = config.epochs
patience = config.patience

train_loss_values = []
val_loss_values = []

best_val_loss = float("inf")
epochs_no_improve = 0
model_path = "best_transformer_decoder.pt"

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for memory, target in train_loader:
        memory, target = memory.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(memory)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if device == "mps":
            torch.mps.empty_cache()

        running_loss += loss.item() * memory.size(0)

    epoch_train_loss = running_loss / len(train_dataset)
    train_loss_values.append(epoch_train_loss)

    model.eval()
    running_val_loss = 0.0
    running_val_cos_sim = 0.0
    with torch.no_grad():
        for memory, target in val_loader:
            memory, target = memory.to(device), target.to(device)
            output = model(memory)
            loss = criterion(output, target)
            cos_sim = nn.functional.cosine_similarity(output, target, dim=1)  # (batch,)
            mean_cos_sim = cos_sim.mean().item()

            running_val_loss += loss.item() * memory.size(0)
            running_val_cos_sim += mean_cos_sim * memory.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    epoch_val_cos_sim = running_val_cos_sim / len(val_dataset)
    val_loss_values.append(epoch_val_loss)

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "val_loss": epoch_val_loss,
        "val_cosine_similarity": epoch_val_cos_sim
    })

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val CosSim: {epoch_val_cos_sim:.4f}")

    # Early stopping check
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Plot Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss_values) + 1), train_loss_values, marker='o', linestyle='-', label="Train Loss")
plt.plot(range(1, len(val_loss_values) + 1), val_loss_values, marker='s', linestyle='-', label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

wandb.finish()