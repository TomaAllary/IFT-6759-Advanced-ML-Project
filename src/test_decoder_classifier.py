import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import wandb

from emotion_classifier import EmotionClassifier
from transformer_decoder import UtteranceDecoder

# Initialize W&B with the same architecture as training
i = wandb.init(project="joint-train-pipeline", config={
    "input_dim": 1536,
    "model_dim": 1536,
    "num_heads": 4,
    "num_layers": 6,
    "dropout": 0.3,
    "clf_hidden": 1024,
    "clf_dropout": 0.5,
    "batch_size": 16
})
config = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dialogues
df = torch.load("dialogue_tensor_data_test.pt", weights_only=True)
X_all, y_all = df["features"], df["labels"]
test_data = list(zip(X_all, y_all))

# Collate full dialogues and shifted labels (same as training)
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

# Load models
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

# Restore weights
ckpt = torch.load("best_joint_model.pt")
decoder.load_state_dict(ckpt["decoder"])
classifier.load_state_dict(ckpt["classifier"])

decoder.eval()
classifier.eval()

# Inference
all_preds, all_labels = [], []
with torch.no_grad():
    loader = DataLoader(test_data, batch_size=config.batch_size, collate_fn=collate_dialogue_batch)
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)    # [B, T, U, D]
        y_batch = y_batch.to(device)    # [B, T-1]
        pred_embs = decoder(x_batch)    # [B, T-1, D]
        B, Tm, D = pred_embs.shape
        flat_pred = pred_embs.view(B * Tm, D)
        flat_labels = y_batch.view(-1)
        mask = flat_labels != -1
        valid_pred = flat_pred[mask]
        valid_labels = flat_labels[mask]
        logits = classifier(valid_pred)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(valid_labels.cpu().tolist())

# Compute and log metrics
test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
test_acc = accuracy_score(all_labels, all_preds)
print(f"Test F1 over all utterances: {test_f1:.4f}")
print(f"Test Accuracy over all utterances: {test_acc:.4f}")

# Log to W&B
class_names = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
wandb.log({
    "test_f1_all_utts": test_f1,
    "test_accuracy_all_utts": test_acc,
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names
    )
})
wandb.finish()
