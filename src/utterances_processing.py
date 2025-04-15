# --- Imoort libraries ---
import os
import torchaudio
import torch
import subprocess
import tempfile
import pandas as pd
import numpy as np
from transformers import AutoFeatureExtractor, WavLMModel, AutoTokenizer, RobertaModel
from torchaudio import datasets
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

"""
audio of utterance are in IEMOCAP\Session1\sentences\wav\Ses01F_impro01
text of utterance are in IEMOCAP\Session1\ dialog \ transcriptions \ Ses01F_impro01.txt
"""
class IEMOCAPUtteranceProcessor():
    def __init__(self,
                 base_dir,
                 use_cpu=True):

        # --- Configuration ---
        self.base_dir = base_dir
        self.iemocap_dir = os.path.join(self.base_dir, "IEMOCAP")

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

    def Process_pipeline(self, seq_length=10):
        df = self.create_csv()
        self.create_pickle(df)

        return self.create_dialogue_sets(seq_length=seq_length)

    def create_csv(self):

        dataframe_entries = [] # to append with dictionaries
        """
        dataframe columns=['session', 'transcript', 'utterance_id', 'utterance_idx', 'utterance', 'dialogue_id']
        
        session:            Session1
        transcript:         Ses01F_impro01
        utterance_id:       Ses01F_impro01_M000
        utterance_idx:      000                     # 3 last digit of utterance_id
        utterance_raw_idx:  1                       # index order of sentence 
        utterance:          Do you have your forms
        dialogue_id:        1                       # dialogue appartenance artificially given by us
        """

        last_dialogue_id = 0
        for session in [1, 2, 3, 4, 5]:
            session_name = f"Session{session}"
            print(f"Processing session {session_name} ...")
            transcript_dir = os.path.join(self.iemocap_dir, session_name, "dialog", "transcriptions")
            print(f"Processing transcripts at {transcript_dir}")

            for discussion_file in os.listdir(transcript_dir):
                if not discussion_file.endswith(".txt"):
                    continue

                discussion_file_path = os.path.join(transcript_dir, discussion_file)
                with open(discussion_file_path, 'r') as file:
                    unique_utterance = []
                    transcript = os.path.splitext(discussion_file)[0]  # remove extension from name

                    lines = file.readlines()
                    raw_idx = 0
                    for line in lines:
                        if not line.startswith("Ses"):
                            continue
                        entry = line.replace(":", "")
                        entry = entry.replace("] ", "/")
                        entry = entry.replace(" [", "/")
                        components = entry.split("/")

                        utterance_id = components[0].strip()  # ex: Ses01F_improv01_M000
                        utterance_idx = utterance_id[-3:] # last 3 characters ex: 000
                        utterance = components[2].strip('\n')

                        # unique_utterance.append(utterance_idx)

                        utt_dict = {}
                        utt_dict["session"]             = session_name
                        utt_dict["transcript"]          = transcript
                        utt_dict["utterance_id"]        = utterance_id
                        utt_dict["utterance_idx"]       = utterance_idx
                        utt_dict["utterance_raw_idx"]   = raw_idx
                        utt_dict["utterance"]           = utterance
                        utt_dict["dialogue_id"]         = -1 #unassigned
                        utt_dict["features"]            = torch.empty(1) #unassigned

                        dataframe_entries.append(utt_dict)
                        raw_idx += 1

                        print(f"{utterance_id} processed")

        dataframe = pd.DataFrame(dataframe_entries)
        dataframe.to_csv(os.path.join(self.iemocap_dir, "utterances.csv"), index=True)

        return dataframe

    def __convert_emotion(self, emotion):
        # "neu", "hap", "ang", "sad", "exc", "fru"
        if emotion == "neu":
            return "neutral"
        if emotion == "hap":
            return "joy"
        if emotion == "ang":
            return "anger"
        if emotion == "sad":
            return "sadness"
        if emotion == "exc":
            return "surprise"
        if emotion == "fru": # NOT SURE, frustration is a light version of anger
            return "anger"

    def create_pickle(self, df):
        # Load the dataset
        torch_dataset = datasets.IEMOCAP(root=self.base_dir)

        processed_data = []

        df.set_index("utterance_id", inplace=True) # for faster lookup
        for sample in torch_dataset:
            waveform, sample_rate, utterance_id, emotion, session_id = sample

            # transform emotion to fit MELD

            if utterance_id in df.index:
                # Emotion
                df.loc[utterance_id, "emotion"] = self.__convert_emotion(emotion)

                audio_features = self.extract_audio_features(waveform, sample_rate)
                text_features = self.extract_text_features(df.loc[utterance_id, "utterance"])

                combined_features = self.combine_audio_text_features(audio_features, text_features)

                # Store results in DataFrame-friendly format
                processed_data.append({
                    "Dialogue_ID": -1, # unassigned
                    "Utterance_ID": utterance_id,
                    "Transcript": df.loc[utterance_id, "transcript"],
                    "Utterance": df.loc[utterance_id, "utterance"],
                    "Utterance_IDX": df.loc[utterance_id, "utterance_raw_idx"],
                    "Emotion": self.__convert_emotion(emotion),  # Store Emotion label
                    "Features": combined_features  # Convert tensor to list
                })

                print(f"Processed features for utterance {utterance_id}")


        df_features = pd.DataFrame(processed_data)

        # Save as Pickle file
        pkl_file = os.path.join(self.iemocap_dir, "utterances_raw_features.pkl")
        df_features.to_pickle(pkl_file)

        print(f"Saved raw features to {pkl_file}")

    def create_dialogue_sets(self, seq_length):
        print("Creating dialogue sets...")

        pkl_file = os.path.join(self.iemocap_dir, "utterances_raw_features.pkl")
        if not os.path.exists(pkl_file):
            print("Cant create dialogue sets because features dataframe doesn't exist. Use create_pickle beforehand!")
            return

        df_features = pd.read_pickle(pkl_file)

        window_size = seq_length
        dialogues = []

        skipped_transcript = 0

        # Group by column A
        for group_key, group in df_features.groupby("Transcript"):
            group = group.reset_index(drop=True)

            if len(group) < window_size:
                print(f"Transcript {group_key} is too short (length={len(group)}). Skipping...")
                skipped_transcript += 1
                continue

            for i in range(len(group) - window_size + 1): # nb od dialogue = Dialogue_size - windows + 1
                window = group.iloc[i:i + window_size].copy()
                window["Dialogue_ID"] = len(dialogues)
                dialogues.append(window)

        # Combine all windows
        final_df = pd.concat(dialogues, ignore_index=True)
        final_df.set_index("Dialogue_ID", inplace=True)

        # Save as Pickle file
        pkl_file = os.path.join(self.iemocap_dir, f"utterances_features_window={seq_length}.pkl")
        final_df.to_pickle(pkl_file)

        print(f"Saved dialogue sets to {pkl_file}")
        print(f"{len(dialogues)} dialogue sets were created")
        print(f"{skipped_transcript} transcript were skipped due to small length.")
        return final_df

    def extract_audio_features(self, waveform, sample_rate):
        """Extracts WavLM features while preserving sequence length."""
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)

        inputs = self.feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.wavlm_model(**inputs)

        return outputs.last_hidden_state.cpu()  # Shape: (1, audio_seq_length, 768)

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


