import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score, confusion_matrix

def main():
    # Initialize Weights & Biases
    run = wandb.init(project="utterance-prediction-transformer-encoder", config={
        "input_dim": 1536,
        "patience": 5
    })
    config = wandb.config
    print(config)

    # Emotion Mapping
    emotion2idx = {
        "anger": 0, "disgust": 1, "fear": 2,
        "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6
    }

    # Transformer Model
    class AudioTextTransformer(nn.Module):
        def __init__(self, input_dim=1536, num_classes=7):
            super(AudioTextTransformer, self).__init__()
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=config.num_heads, dropout=config.dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(input_dim, num_classes)
            
        def forward(self, x):
            x = x.transpose(0, 1)
            x = self.transformer_encoder(x)
            x = x.transpose(0, 1).transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    # Dataset Class
    class UtteranceDataset(Dataset):
        def __init__(self, dataframe, label_map):
            self.data = dataframe
            self.label_map = label_map
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            features = row["Features"]
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float)
            label = self.label_map[row["Emotion"].lower()]
            features = features.mean(dim=1, keepdim=True).squeeze(0)
            return features, label

    # Load and Split Data
    df_features = pd.read_pickle("features_per_sentence.pkl")
    df_features_val = pd.read_pickle("features_per_sentence_val.pkl")
    train_dataset = UtteranceDataset(df_features, emotion2idx)
    val_dataset = UtteranceDataset(df_features_val, emotion2idx)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    #train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    targets = [target for _,target in train_dataset]
    class_sample_counts = pd.Series(targets).value_counts().sort_index()
    sample_weights = [1.0 / class_sample_counts[t] for t in targets]
    #print(f"Sample weights: {sample_weights}")
    print(f"Class sample counts: {class_sample_counts}")
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,  sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Device and Model
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = AudioTextTransformer(input_dim=config.input_dim, num_classes=len(emotion2idx)).to(device)

    # Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    if config.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")



    # Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_loss_values, val_loss_values = [], []
    train_acc_values, val_acc_values = [], []
    model_path = "best_transformer_encoder.pt"

    for epoch in range(config.epochs):
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

        # Log to W&B
        wandb.log({
            "Train Loss": epoch_train_loss,
            "Validation Loss": epoch_val_loss,
            "Train Accuracy": epoch_train_acc,
            "Validation Accuracy": epoch_val_acc,
            "F1 Score": f1,
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=list(emotion2idx.keys())
            ),
            "Epoch": epoch + 1
        })


        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        # Early Stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print("Early stopping triggered.")
                break

  

sweep_configuration = {
    "name": "transformer_sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "Validation Loss"},
    "parameters": {
        "lr": {"min": 0.0001, "max": 0.01},
        "batch_size": {"values": [4, 8, 16]},
        "epochs": {"values": [10, 15, 20]},
        "num_heads": {"values": [2, 4, 8]},
        "num_layers": {"values": [1, 2, 3]},
        "dropout": {"min": 0.1, "max": 0.5},
        "optimizer": {"values": ["adam", "sgd"]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="utterance-prediction-transformer-encoder")
wandb.agent(sweep_id, function=main, count=10)
wandb.finish()
