# --- Imoort libraries ---
import os
import torchaudio
import torch
import subprocess
import tempfile
import pandas as pd
import numpy as np
from transformers import AutoFeatureExtractor, WavLMModel, AutoTokenizer, RobertaModel
from pydub import AudioSegment
from collections import defaultdict
import pickle

##########################################################################################
##                                 MELD PROCESSING                                      ##
##########################################################################################
class MELDUtteranceProcessor():

    """
    project_dir                 -> os.getcwd() or whatever to reach project root (depends on notebook dir)
    MELD_folder                 -> "train"
    audio_data_folder           -> "train_splits"
    text_data_csv_filename      -> "train_sent_emo.csv"
    embeddings_pkl_filename     -> "final_tensor_conversation_v2"
    """
    def __init__(self,
                 project_dir,
                 MELD_folder,
                 audio_data_folder,
                 text_data_csv_filename,
                 embeddings_pkl_filename,
                 use_cpu=False):
        # --- Configuration ---
        self.BASE_DIR = project_dir
        self.DATA_DIR = os.path.join(self.BASE_DIR, "MELD.Raw", MELD_folder)

        # File to read
        self.MP4_DIR = os.path.join(self.DATA_DIR, audio_data_folder)  # Audio files
        self.CSV_FILE = os.path.join(self.DATA_DIR, text_data_csv_filename)  # Text utterances
        # File/Folder to create
        self.CODE_FILENAME =  os.path.join(self.DATA_DIR, f"{embeddings_pkl_filename}.pkl")



        # --- Models and Tokenizers ---
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaModel.from_pretrained("roberta-base")

        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.device = "cuda" if torch.cuda.is_available() else self.device

        self.wavlm_model.to(self.device)
        self.roberta_model.to(self.device)

        print(f"ConversationProcessor initialized using device: {self.device}")


    def Process_pipeline(self):
        df = pd.read_csv(self.CSV_FILE)
        processed_data = []

        # Process each utterance
        for index, row in df.iterrows():
            utterance = row["Utterance"]
            dialogue_id = row["Dialogue_ID"]
            utterance_id = row["Utterance_ID"]
            emotion = row["Emotion"]  # Get Emotion label
            sentiment = row["Sentiment"]  # Get Sentiment label

            mp4_filename = f'dia{dialogue_id}_utt{utterance_id}.mp4'
            mp4_path = os.path.join(self.MP4_DIR, mp4_filename)

            if not os.path.exists(mp4_path):
                print(f"Skipping missing file: {mp4_path}")
                continue

            print(f"Processing {mp4_filename}...")

            audio_features = self.extract_audio_features(mp4_path)
            text_features = self.extract_text_features(utterance)

            if audio_features is None or text_features is None:
                print(f"Skipping {mp4_filename} due to missing features")
                continue

            combined_features = self.combine_audio_text_features(audio_features, text_features)

            # Store results in DataFrame-friendly format
            processed_data.append({
                "Dialogue_ID": dialogue_id,
                "Utterance_ID": utterance_id,
                "Utterance": utterance,
                "Emotion": emotion,  # Store Emotion label
                "Sentiment": sentiment,  # Store Sentiment label
                "Features": combined_features  # Convert tensor to list
            })

        # Create DataFrame
        df_features = pd.DataFrame(processed_data)

        # Save as Pickle file
        df_features.to_pickle(self.CODE_FILENAME)

        print(f"Saved features to {self.CODE_FILENAME}")


    #################################### AUDIO FEATURE EXTRACT ####################################
    def extract_audio_from_mp4(self, mp4_path):
        """Extracts audio from an MP4 file and converts it to WAV format."""
        temp_wav = tempfile.mktemp(suffix=".wav")  # Persistent temp file
        command = ["ffmpeg", "-i", mp4_path, "-ac", "1", "-ar", "16000", "-y", temp_wav]

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
            if result.returncode != 0:
                print(f"Skipping {mp4_path} due to FFmpeg error.")
                return None
            return temp_wav
        except Exception as e:
            print(f"Error processing {mp4_path}: {e}")
            return None

    def extract_audio_features(self, mp4_path):
        """Extracts WavLM features while preserving sequence length."""
        wav_path = self.extract_audio_from_mp4(mp4_path)

        if wav_path is None:  # Problem reading mp4 file
            return  # skipping
        waveform, sample_rate = torchaudio.load(wav_path)

        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)

        inputs = self.feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.wavlm_model(**inputs)

        return outputs.last_hidden_state.cpu()  # Shape: (1, audio_seq_length, 768)


    #################################### TEXT FEATURE EXTRACT ####################################
    def extract_text_features(self, utterance):
        """Extracts RoBERTa features while preserving sequence length."""
        inputs = self.tokenizer(utterance, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)

        return outputs.last_hidden_state.cpu()  # Shape: (1, seq_length, 768)


    def combine_audio_text_features(self, audio_features, text_features):
        """Aligns sequence length and combines features."""
        seq_length = min(audio_features.shape[1], text_features.shape[1])
        audio_features = audio_features[:, :seq_length, :]
        text_features = text_features[:, :seq_length, :]
        return torch.cat((audio_features, text_features), dim=-1)  # Shape: (1, seq_length, 1536)


##########################################################################################
##                               IEMOCAP PROCESSING                                     ##
##########################################################################################
class IEMOCAPUtteranceProcessor():
    def __init__(self):
        self.file = 'jgk'


def MergeDataset(dataset_names: list, new_dataset_filename):

    last_dialogue_id = 0
    dataframes = []
    for dataset_name in dataset_names:
        df = pd.read_pickle(dataset_name)
        # increment dataset id based on previous merging set
        df["Dialogue_ID"] = df["Dialogue_ID"].apply(lambda x: int(x + last_dialogue_id))

        dataframes.append(df)
        last_dialogue_id = df["Dialogue_ID"].max()

    merged_df = pd.concat(dataframes)
    merged_df.to_pickle(new_dataset_filename)






