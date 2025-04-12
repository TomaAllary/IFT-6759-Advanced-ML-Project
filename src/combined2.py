import os
import torchaudio
import torch
import subprocess
import tempfile
import pandas as pd
import pickle
from transformers import AutoFeatureExtractor, WavLMModel, AutoTokenizer, RobertaModel

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load models
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")

device = "mps" if torch.backends.mps.is_available() else "cpu"
wavlm_model.to(device)
roberta_model.to(device)

# Get current directory
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "MELD.Raw", "train")

MP4_DIR = os.path.join(DATA_DIR, "train_splits")  # Audio files
CSV_FILE = os.path.join(DATA_DIR, "train_sent_emo.csv")  # Text utterances
PICKLE_FILE_UTTERANCE = os.path.join(BASE_DIR, "features_per_utterance.pkl")  # Output file (utterance-level)
PICKLE_FILE_DIALOGUE = os.path.join(BASE_DIR, "features_per_dialogue.pkl")  # Output file (dialogue-level)

# Load CSV file
df = pd.read_csv(CSV_FILE)

def extract_audio_from_mp4(mp4_path):
    """Extracts audio from an MP4 file and converts it to WAV format."""
    temp_wav = tempfile.mktemp(suffix=".wav")
    command = ["ffmpeg", "-i", mp4_path, "-ac", "1", "-ar", "16000", "-y", temp_wav]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
        if result.returncode != 0:
            print(f"Skipping {mp4_path} due to FFmpeg error:")
            return None
        return temp_wav
    except Exception as e:
        print(f"Error processing {mp4_path}: {e}")
        return None

def extract_audio_features(mp4_path):
    """Extracts WavLM features."""
    wav_path = extract_audio_from_mp4(mp4_path)
    if wav_path is None:
        return None
    
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    
    inputs = feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = wavlm_model(**inputs)
    
    return outputs.last_hidden_state.cpu()

def extract_text_features(utterance):
    """Extracts RoBERTa features."""
    inputs = tokenizer(utterance, truncation=True, max_length=128, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    
    return outputs.last_hidden_state.cpu()

def combine_audio_text_features(audio_features, text_features):
    """Aligns sequence length and combines features."""
    seq_length = min(audio_features.shape[1], text_features.shape[1])
    audio_features = audio_features[:, :seq_length, :]
    text_features = text_features[:, :seq_length, :]
    return torch.cat((audio_features, text_features), dim=-1)

# Lists to store results
data_utterance = []
data_dialogue = {}

# Process each utterance
for index, row in df.iterrows():
    dialogue_id = row["Dialogue_ID"]
    utterance_id = row["Utterance_ID"]
    utterance = row["Utterance"]
    emotion = row["Emotion"]  # Last utterance determines dialogue emotion
    sentiment = row["Sentiment"]  # Last utterance determines dialogue sentiment
    
    mp4_filename = f'dia{dialogue_id}_utt{utterance_id}.mp4'
    mp4_path = os.path.join(MP4_DIR, mp4_filename)
    
    if not os.path.exists(mp4_path):
        print(f"Skipping missing file: {mp4_path}")
        continue
    
    print(f"Processing {mp4_filename}...")
    audio_features = extract_audio_features(mp4_path)
    text_features = extract_text_features(utterance)
    
    if audio_features is None or text_features is None:
        print(f"Skipping {mp4_filename} due to missing features")
        continue
    
    combined_features = combine_audio_text_features(audio_features, text_features)
    
    # Store utterance-level features
    data_utterance.append({
        "Dialogue_ID": dialogue_id,
        "Utterance_ID": utterance_id,
        "Utterance": utterance,
        "Emotion": emotion,
        "Sentiment": sentiment,
        "Features": combined_features
    })
    
    # Store dialogue-level features
    if dialogue_id not in data_dialogue:
        data_dialogue[dialogue_id] = {
            "Dialogue_ID": dialogue_id,
            "Emotion": emotion,  # Assign last utterance's emotion
            "Sentiment": sentiment,  # Assign last utterance's sentiment
            "Features": []
        }
    data_dialogue[dialogue_id]["Features"].append(combined_features)

# Convert dialogue features to final tensors
for dialogue_id in data_dialogue:
    data_dialogue[dialogue_id]["Features"] = torch.cat(data_dialogue[dialogue_id]["Features"], dim=1)  # Concatenate utterances

df_features_utterance = pd.DataFrame(data_utterance)
df_features_dialogue = pd.DataFrame(data_dialogue.values())

# Save as Pickle file
df_features_utterance.to_pickle(PICKLE_FILE_UTTERANCE)
df_features_dialogue.to_pickle(PICKLE_FILE_DIALOGUE)

print(f"Saved utterance-level features to {PICKLE_FILE_UTTERANCE}")
print(f"Saved dialogue-level features to {PICKLE_FILE_DIALOGUE}")
