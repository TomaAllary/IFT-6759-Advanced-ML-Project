import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import wandb
import numpy as np

from emotion_classifier import EmotionClassifier
from transformer_decoder import UtteranceDecoder

# === Initialize W&B and config ===
wandb.init(project="joint-train-pipeline", config={
    "input_dim": 1536,
    "model_dim": 1536,
    "num_heads": 4,
    "num_layers": 6,
    "dropout": 0.3,
    "clf_hidden": 1024,
    "clf_dropout": 0.3,
    "batch_size": 16,
    "lr": 1e-4,
    "epochs": 100,
    "patience": 30,
    "lambda_cosine": 0.1
})
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = torch.load("dialogue_tensor_data_train.pt", weights_only=True)
X_all, y_all = df["features"], df["labels"]
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
train_data = list(zip(X_train, y_train))
val_data = list(zip(X_val, y_val))

# Instantiate models
decoder = UtteranceDecoder(
    input_dim=config.input_dim,
    model_dim=config.model_dim,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    dropout=config.dropout
).to(device)
classifier = EmotionClassifier(
    input_dim=config.input_dim,
    hidden_dim=config.clf_hidden,
    dropout=config.clf_dropout
).to(device)

# Load pretrained decoder weights
ckpt = torch.load("best_utterance_decoder.pt", weights_only=True)
decoder.load_state_dict(ckpt["decoder"])
params = list(decoder.parameters()) + list(classifier.parameters())

# Loss & optimizer setup
all_labels = [label for dlg in y_train for label in dlg]
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

optimizer = optim.AdamW(list(decoder.parameters()) + list(classifier.parameters()), lr=config.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5, verbose=True)
lambda_cosine = config.lambda_cosine

# Collate function
def collate_dialogue_batch(batch):
    dialogs, labels = zip(*batch)
    max_T = max(len(d) for d in dialogs)
    max_U = max(utt.size(0) for d in dialogs for utt in d)
    D = dialogs[0][0].size(1)
    padded_x, padded_y = [], []
    for d, lab in zip(dialogs, labels):
        utts = []
        for utt in d:
            if utt.size(0) < max_U:
                utt = torch.cat([utt, torch.zeros(max_U - utt.size(0), D)], dim=0)
            utts.append(utt)
        while len(utts) < max_T:
            utts.append(torch.zeros(max_U, D))
        padded_x.append(torch.stack(utts))
        shifted = lab[1:]
        pad = torch.full((max_T - 1,), -1, dtype=torch.long)
        pad[:len(shifted)] = torch.tensor(shifted, dtype=torch.long)
        padded_y.append(pad)
    return torch.stack(padded_x), torch.stack(padded_y)

# Vectorized target embeddings
def get_target_embeddings(x_batch, pooler):
    B, T, U, D = x_batch.shape
    flat = x_batch[:, 1:, :, :].reshape(B * (T - 1), U, D)
    pooled = pooler(flat)
    return pooled.view(B, T - 1, D)

# Training loop
best_val_f1 = -1
patience_counter = 0

for epoch in range(config.epochs):
    decoder.train(); classifier.train()
    train_preds, train_trues = [], []
    train_loss = 0

    for x_batch, y_batch in DataLoader(train_data, batch_size=config.batch_size,
                                       shuffle=True, collate_fn=collate_dialogue_batch):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        pred_embs = decoder(x_batch)
        tgt_embs = get_target_embeddings(x_batch, decoder.pooler)

        B, Tm, D = pred_embs.shape
        flat_pred = pred_embs.view(B * Tm, D)
        flat_tgt = tgt_embs.view(B * Tm, D)
        mask = (y_batch.view(-1) != -1)

        valid_pred = flat_pred[mask]
        valid_tgt = flat_tgt[mask]

        cosine_loss = 1 - F.cosine_similarity(F.normalize(valid_pred, dim=-1),
                                              F.normalize(valid_tgt, dim=-1), dim=-1).mean()

        logits = classifier(valid_pred)
        ce_loss = criterion(logits, y_batch.view(-1)[mask])

        loss = ce_loss + lambda_cosine * cosine_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        

        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().tolist())
        train_trues.extend(y_batch.view(-1)[mask].cpu().tolist())
        train_loss += loss.item() * valid_pred.size(0)

    train_f1 = f1_score(train_trues, train_preds, average='macro', zero_division=0)
    avg_train_loss = train_loss / len(train_trues)

    decoder.eval(); classifier.eval()
    val_preds, val_trues, val_loss = [], [], 0
    with torch.no_grad():
        for x_batch, y_batch in DataLoader(val_data, batch_size=config.batch_size,
                                           collate_fn=collate_dialogue_batch):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred_embs = decoder(x_batch)
            tgt_embs = get_target_embeddings(x_batch, decoder.pooler)

            B, Tm, D = pred_embs.shape
            flat_pred = pred_embs.view(B * Tm, D)
            flat_tgt = tgt_embs.view(B * Tm, D)
            mask = (y_batch.view(-1) != -1)

            valid_pred = flat_pred[mask]
            valid_tgt = flat_tgt[mask]

            cosine_loss = 1 - F.cosine_similarity(F.normalize(valid_pred, dim=-1),
                                                  F.normalize(valid_tgt, dim=-1), dim=-1).mean()
            logits = classifier(valid_pred)
            ce_loss = criterion(logits, y_batch.view(-1)[mask])
            loss = ce_loss + lambda_cosine * cosine_loss

            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_trues.extend(y_batch.view(-1)[mask].cpu().tolist())
            val_loss += loss.item() * valid_pred.size(0)

    val_f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)
    avg_val_loss = val_loss / len(val_trues)

    lr = optimizer.param_groups[0]['lr']
    wandb.log({
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'train_f1': train_f1,
        'val_loss': avg_val_loss,
        'val_f1': val_f1,
        'lr': lr
    }, step=epoch)

    print(f"Epoch {epoch+1} – Train F1: {train_f1:.3f}, Val F1: {val_f1:.3f}, LR: {lr:.2e}")

    scheduler.step(val_f1)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save({'decoder': decoder.state_dict(), 'classifier': classifier.state_dict()},
                   'best_joint_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= config.patience:
            print("Early stopping.")
            break

wandb.finish()
