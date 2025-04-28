import os
import torch
import torchaudio
import subprocess
import tempfile
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMModel, AutoTokenizer, RobertaModel

# Import your label mapping here
from emotion_classifier import emotion2idx

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_audio_from_mp4(mp4_path):
    temp_wav = tempfile.mktemp(suffix=".wav")
    command = ["ffmpeg", "-i", mp4_path, "-ac", "1", "-ar", "16000", "-y", temp_wav]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        if result.returncode != 0:
            print(f"Skipping {mp4_path} due to FFmpeg error:\n{result.stderr}")
            return None
        return temp_wav
    except Exception as e:
        print(f"Error processing {mp4_path}: {e}")
        return None

def extract_audio_features(mp4_path, feature_extractor, wavlm_model, device, max_audio_len=160000):
    wav_path = extract_audio_from_mp4(mp4_path)
    if wav_path is None:
        return None
    
    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)

    # Truncate if too long
    if waveform.shape[1] > max_audio_len:
        waveform = waveform[:, :max_audio_len]

    inputs = feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = wavlm_model(**inputs)

    return outputs.last_hidden_state.cpu()

def extract_text_features(utterance, tokenizer, roberta_model, device):
    inputs = tokenizer(utterance, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.last_hidden_state.cpu()

def combine_audio_text_features(audio_features, text_features):
    seq_length = min(audio_features.shape[1], text_features.shape[1])
    audio_features = audio_features[:, :seq_length, :]
    text_features = text_features[:, :seq_length, :]
    return torch.cat((audio_features, text_features), dim=-1)

def main(split_type):
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "MELD.Raw")

    if split_type == "train":
        MP4_DIR = os.path.join(DATA_DIR, "train_splits")
        CSV_FILE = os.path.join(DATA_DIR, "train_sent_emo.csv")
        PICKLE_FILE = os.path.join(BASE_DIR, "features_per_sentence.pkl")
    else:
        MP4_DIR = os.path.join(DATA_DIR, "output_repeated_splits_test")
        CSV_FILE = os.path.join(DATA_DIR, "test_sent_emo.csv")
        PICKLE_FILE = os.path.join(BASE_DIR, "features_per_sentence_test.pkl")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

    # Step 1: Extract and save per-utterance features
    df = pd.read_csv(CSV_FILE)
    data = []

    for index, row in df.iterrows():
        utterance = row["Utterance"]
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        emotion = row["Emotion"]
        sentiment = row["Sentiment"]
        
        mp4_filename = f'dia{dialogue_id}_utt{utterance_id}.mp4'
        mp4_path = os.path.join(MP4_DIR, mp4_filename)

        if not os.path.exists(mp4_path):
            print(f"Skipping missing file: {mp4_path}")
            continue

        print(f"Processing {mp4_filename}...")

        audio_features = extract_audio_features(mp4_path, feature_extractor, wavlm_model, device)
        text_features = extract_text_features(utterance, tokenizer, roberta_model, device)

        if audio_features is None or text_features is None:
            print(f"Skipping {mp4_filename} due to missing features")
            continue

        combined_features = combine_audio_text_features(audio_features, text_features)

        data.append({
            "Dialogue_ID": dialogue_id,
            "Utterance_ID": utterance_id,
            "Utterance": utterance,
            "Emotion": emotion,
            "Sentiment": sentiment,
            "Features": combined_features
        })

    df_features = pd.DataFrame(data)
    df_features.to_pickle(PICKLE_FILE)
    print(f"✅ Saved: {PICKLE_FILE}")

    # Step 2: Build dialogue features
    df = pd.read_pickle(PICKLE_FILE)
    dialogue_groups = df.groupby("Dialogue_ID")
    dialogue_data = []

    for dialog_id, group in dialogue_groups:
        sorted_group = group.sort_values("Utterance_ID")
        utterance_embeddings = []
        emotion_labels = []

        for f, emo in zip(sorted_group["Features"], sorted_group["Emotion"]):
            f = f.clone().detach() if isinstance(f, torch.Tensor) else torch.tensor(f)
            utterance_embeddings.append(f.squeeze(0))
            emotion_labels.append(emo)

        if len(utterance_embeddings) < 2:
            continue

        dialogue_data.append({
            "Dialogue_ID": dialog_id,
            "Features": utterance_embeddings,
            "Emotion": emotion_labels
        })

    dialogue_pickle = "features_per_dialog.pkl" if split_type == "train" else "features_per_dialog_test.pkl"
    df_dialog = pd.DataFrame(dialogue_data)
    df_dialog.to_pickle(dialogue_pickle)
    print(f"✅ Saved: {dialogue_pickle}")

    # Step 3: Build tensor dataset
    df = pd.read_pickle(dialogue_pickle)

    all_features = []
    all_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        feature_list = row["Features"]
        label_list = row["Emotion"]

        feature_tensors = []
        for f in feature_list:
            if isinstance(f, list):
                f = torch.tensor(f, dtype=torch.float32)
            elif isinstance(f, torch.Tensor):
                f = f.detach()
            if f.dim() == 3:
                f = f.squeeze(0)
            feature_tensors.append(f)

        label_indices = [emotion2idx.get(lbl.strip().lower(), -1) for lbl in label_list]

        if -1 in label_indices or len(label_indices) != len(feature_tensors):
            continue

        all_features.append(feature_tensors)
        all_labels.append(label_indices)

    tensor_output = "dialogue_tensor_data.pt" if split_type == "train" else "dialogue_tensor_data_test.pt"
    torch.save({"features": all_features, "labels": all_labels}, tensor_output)
    print(f"✅ Saved: {tensor_output} with {len(all_features)} dialogues.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "test"], help="Specify whether to process train or test data.")
    args = parser.parse_args()
    main(args.split)
