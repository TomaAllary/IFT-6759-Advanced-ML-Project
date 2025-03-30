import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split

# Emotion Mapping
emotion2idx = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

# Transformer Model with Padding Support
class AudioTextTransformer(nn.Module):
    def __init__(self, input_dim=1536, model_dim=256, num_heads=4, num_layers=1, num_classes=7, dropout=0.1):
        super(AudioTextTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(model_dim, num_classes)
        
    def forward(self, x, mask):
        x = self.input_proj(x)  # (batch, seq_length, model_dim)
        x = x.transpose(0, 1)   # (seq_length, batch, model_dim)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1).transpose(1, 2)  # (batch, model_dim, seq_length)
        x = self.pool(x).squeeze(-1)  # (batch, model_dim)
        logits = self.fc(x)
        return logits

# Custom Dataset for Dialogue Features
class DialogueDataset(Dataset):
    def __init__(self, dataframe, label_map):
        self.data = dataframe
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row["Features"]
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float)  # Convert to tensor if not already
        label = self.label_map[row["Emotion"].lower()]
        return features, label

# Padding Function for Variable-Length Sequences
def collate_fn(batch):
    features, labels = zip(*batch)  # Separate features and labels

    # Find max sequence length in batch
    max_seq_len = max(f.shape[1] for f in features)

    # Pad sequences to the max length
    padded_features = []
    padding_masks = []
    
    for f in features:
        seq_len = f.shape[1]
        pad_size = max_seq_len - seq_len
        padded_f = torch.cat([f, torch.zeros((1, pad_size, f.shape[2]))], dim=1)  # Pad with zeros
        padded_features.append(padded_f)

        # Create padding mask (True where padding exists)
        padding_mask = torch.cat([torch.zeros(seq_len), torch.ones(pad_size)]).bool()
        padding_masks.append(padding_mask)

    padded_features = torch.stack(padded_features)  # (batch, 1, max_seq_len, 1536)
    padded_features = padded_features.squeeze(1)    # (batch, max_seq_len, 1536)
    padding_masks = torch.stack(padding_masks)      # (batch, max_seq_len)

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, labels, padding_masks

# Load Features from Pickle File
df_features = pd.read_pickle("features_per_dialog.pkl")
dataset = DialogueDataset(df_features, emotion2idx)


# Define split ratio (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Model Initialization
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AudioTextTransformer(input_dim=1536, model_dim=256, num_heads=4, num_layers=1, num_classes=len(emotion2idx)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20

# Store loss values
train_loss_values = []
val_loss_values = []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_features, batch_labels, padding_masks in train_loader:
        batch_features = batch_features.to(device)  # (batch, max_seq_len, 1536)
        batch_labels = batch_labels.to(device)
        padding_masks = padding_masks.to(device)  # (batch, max_seq_len)

        optimizer.zero_grad()
        outputs = model(batch_features, padding_masks)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        if device == "mps":
            torch.mps.empty_cache()  # Free memory for MPS
        
        running_loss += loss.item() * batch_features.size(0)

    epoch_train_loss = running_loss / len(dataset)
    train_loss_values.append(epoch_train_loss)  # Store loss for graphing

    ### Validation Phase ###
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():  # No gradient calculation for validation
        for batch_features, batch_labels, padding_masks in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            padding_masks = padding_masks.to(device)

            outputs = model(batch_features, padding_masks)
            loss = criterion(outputs, batch_labels)

            running_val_loss += loss.item() * batch_features.size(0)

    epoch_val_loss = running_val_loss / len(val_dataset)
    val_loss_values.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# Plot Training vs. Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_loss_values, marker='o', linestyle='-', label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_loss_values, marker='s', linestyle='-', label="Validation Loss", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
