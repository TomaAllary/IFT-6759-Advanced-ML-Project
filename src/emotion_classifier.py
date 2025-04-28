import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import wandb
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import random

# Set seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# MLP Emotion Classifier with Attention Pooling
class EmotionClassifier(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=1024, num_classes=7, dropout=0.7):
        super(EmotionClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        return self.classifier(x)

# Custom Dataset and Collate function remain unchanged
class EmotionDataset(Dataset):
    def __init__(self, dataframe, label_map):
        self.samples = []
        self.label_map = label_map

        for _, row in dataframe.iterrows():
            features = row["Features"]
            emotions = row["Emotion"]

            # If a single emotion is provided, wrap into a list.
            if isinstance(emotions, str):
                features = [features]
                emotions = [emotions]

            for feature, emotion in zip(features, emotions):
                emotion = emotion.strip().lower()
                if not isinstance(feature, torch.Tensor):
                    feature = torch.tensor(feature, dtype=torch.float32)
                else:
                    feature = feature.float()
                feature = torch.squeeze(feature)  # Remove dimensions of size 1
                self.samples.append((feature, self.label_map[emotion]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    features, labels = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels

if __name__ == "__main__":
    # Initialize Weights & Biases with hyperparameters.
    wandb.init(project="emotion-classification", config={
        "lr": 5e-5,
        "batch_size": 16,
        "epochs": 100,            
        "patience": 15,
        "hidden_dim": 1024,  
        "dropout": 0.7, 
    })
    config = wandb.config

    # Load Data
    df = pd.read_pickle("features_per_sentence.pkl")
    dataset = EmotionDataset(df, emotion2idx)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionClassifier(hidden_dim=config.hidden_dim, dropout=config.dropout).to(device)

    # Compute class weights on mapped labels
    mapped_labels = [label for _, label in dataset]
    print("Mapped label distribution:", Counter(mapped_labels))
    class_weights = compute_class_weight(class_weight='balanced',
                                        classes=np.unique(mapped_labels),
                                        y=mapped_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=config.patience, verbose=True)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Training Loop
    for epoch in range(config.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_preds, train_labels = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            train_preds.extend(pred.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        train_acc = correct / total
        train_loss /= len(train_dataset)
        train_f1 = f1_score(train_labels, train_preds, average='macro')

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item() * x.size(0)
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = correct / total
        val_loss /= len(val_dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        }, step=epoch)

        # Optional: log per-class F1 and confusion matrix
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
        for emotion, metrics in report.items():
            if emotion in class_names:
                wandb.log({f"val_f1_{emotion}": metrics['f1-score']}, step=epoch)
        cm = confusion_matrix(val_labels, val_preds)
        wandb.log({
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=val_labels,
                preds=val_preds,
                class_names=class_names
            )
        }, step=epoch)

        print(f"Epoch {epoch+1}/{config.epochs} - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        scheduler.step(val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_emotion_classifier.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    wandb.finish()
