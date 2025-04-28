import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

# Emotion Mapping
emotion2idx = {
        "anger": 0, "disgust": 1, "fear": 2,
        "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6
    }
config = {
    "input_dim": 1536,
    "num_heads": 8,
    "num_layers":1,
    "dropout": 0.45,
    "label_smoothing": 0.1,
    "lr": 0.00055,
    "epochs": 25,
    "patience": 5,
    "batch_size": 8,
    "optimizer": "sgd"

}
# Transformer Model
class AudioTextTransformer(nn.Module):
    def __init__(self, input_dim=1536, num_classes=7):
        super(AudioTextTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=config["num_heads"], dropout=config["dropout"],dim_feedforward=1048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc1(x)
        return x

# Dataset Class
class UtteranceDataset(Dataset):
    def __init__(self, dataframe, label_map, context_size=1):
        self.data = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - (self.context_size - 1)

    def __getitem__(self, idx):
        # Collect a sequence of `context_size` features
        feature_seq = []
        for i in range(self.context_size):
            row = self.data.iloc[idx + i]
            features = row["Features"]
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float)
            features = features.mean(dim=1, keepdim=True).squeeze(0)  # [1536]
            feature_seq.append(features.unsqueeze(0))  # [1, 1536]

        label = self.label_map[self.data.iloc[idx + self.context_size - 1]["Emotion"].lower()]
        feature_seq = torch.cat(feature_seq, dim=0)  # [context_size, 1536]
        feature_seq = feature_seq.mean(dim=0).squeeze()
        return feature_seq, label


# Load and Split Data
df_features = pd.read_pickle("features/features_per_sentence_next.pkl")
df_features_val = pd.read_pickle("features/features_per_sentence_val_next.pkl")
train_dataset = UtteranceDataset(df_features, emotion2idx)
val_dataset = UtteranceDataset(df_features_val, emotion2idx)
targets = [target for _,target in train_dataset]
class_sample_counts = pd.Series(targets).value_counts().sort_index()
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# Device and Model
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = AudioTextTransformer(input_dim=config["input_dim"], num_classes=len(emotion2idx)).to(device)

class_counts = torch.tensor(class_sample_counts.tolist(), dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(device),label_smoothing=config["label_smoothing"])

if config["optimizer"] == "adam":
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
elif config["optimizer"] == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
else:
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")



# Training Loop
best_val_loss = float("inf")
epochs_no_improve = 0
train_loss_values, val_loss_values = [], []
train_acc_values, val_acc_values = [], []
model_path = "best_transformer_encoder.pt"

for epoch in range(config["epochs"]):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if device == "mps":
            torch.mps.empty_cache()

    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total
    train_loss_values.append(epoch_train_loss)
    train_acc_values.append(epoch_train_acc)

    


    # Validation
    model.eval()
    running_val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = running_val_loss / val_total
    epoch_val_acc = val_correct / val_total
    val_loss_values.append(epoch_val_loss)
    val_acc_values.append(epoch_val_acc)

    # Compute F1 and Confusion Matrix
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Epoch [{epoch+1}/{config['epochs']}]")
    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("-" * 50)