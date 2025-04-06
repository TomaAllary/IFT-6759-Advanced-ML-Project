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

class ConversationProcessor_CNN():

    """
    project_dir                 -> os.getcwd() or whatever to reach project root (depends on notebook dir)
    MELD_folder                 -> "train"
    audio_data_folder           -> "train_splits"
    text_data_csv_filename      -> "train_sent_emo.csv"
    combined_audio_data_folder  -> "train_combined"
    combined_csv_filename       -> "MELD_combined_v2.csv"
    embeddings_plk_filename     -> "final_tensor_conversation_v2"
    """
    def __init__(self,
                 project_dir,
                 MELD_folder,
                 audio_data_folder,
                 text_data_csv_filename,
                 combined_audio_data_folder,
                 combined_csv_filename,
                 embeddings_plk_filename,
                 should_add_label_as_text=True):
        # --- Configuration ---
        self.BASE_DIR = project_dir
        self.DATA_DIR = os.path.join(self.BASE_DIR, "MELD.Raw", MELD_folder)
        self.should_add_label_as_text = should_add_label_as_text

        # File to read
        self.MP4_DIR = os.path.join(self.DATA_DIR, audio_data_folder)  # Audio files
        self.CSV_FILE = os.path.join(self.DATA_DIR, text_data_csv_filename)  # Text utterances
        # File/Folder to create
        self.COMBINED_DIR = os.path.join(self.DATA_DIR, combined_audio_data_folder)  # Combined Audio files
        self.DATA_CSV_FILE = os.path.join(self.DATA_DIR, combined_csv_filename)
        self.CODE_FILENAME =  os.path.join(self.DATA_DIR, f"{embeddings_plk_filename}.plk")
        self.CODE_LABEL_FILENAME = os.path.join(self.DATA_DIR, f"{embeddings_plk_filename}_labels.csv")



        # --- Models and Tokenizers ---
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_model = RobertaModel.from_pretrained("roberta-base")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else self.device

        self.wavlm_model.to(self.device)
        self.roberta_model.to(self.device)

        print(f"ConversationProcessor initialized using device: {self.device}")


    def Process_pipeline(self):

        # Merge all utterances audio file of dialogue into only one for each dialogue
        self.CombineAudio()

        df = self.Create_combined_csv()
        self.Create_features_dataset(df)


    def Create_combined_csv(self):

        # Merge utterances csv into one dialogue. Add ref to newly created combined audio file
        raw_data_df = pd.read_csv(self.CSV_FILE)
        combined_df = self.Combine_AudioText_CSV(raw_data_df, should_add_label_as_text=self.should_add_label_as_text)

        return combined_df

    def Create_features_dataset(self, combined_df):
        # From combined CSV, process text & audio through pre-trained model (WaveLM + RoBERTa)
        # combine both embedding into a single dataset (plk file + csv labels)
        self.process_v2(combined_df)


    def CombineAudio(self):
        """
        Take all mp4 files for a given DialogueID
        and combine them into a single mp4 per DialogueID.
        """
        print("Combining Audio files....")

        # Get all .mp4 audio files
        audio_files = [f for f in os.listdir(self.MP4_DIR) if f.endswith(".mp4")]

        nb_of_files = len(audio_files)
        nb_of_processed_files = 0

        # Group files by dialogue number
        dialogue_dict = defaultdict(list)
        for file in audio_files:
            parts = file.split("_")  # Extract dialogue number
            if len(parts) >= 2:
                dialogue_num = parts[0]  # First part is the dialogue ID
                dialogue_dict[dialogue_num].append(file)

        # Sort files by utterance ID within each dialogue
        for dialogue in dialogue_dict:
            dialogue_dict[dialogue].sort(key=lambda x: int(x.split("_")[1].split(".")[0][3:]))

        # Create sub-folder for combined audio files
        os.makedirs(self.COMBINED_DIR, exist_ok=True)

        # Merge audio files for each dialogue
        for dialogue_num, files in dialogue_dict.items():
            try:
                combined_audio = AudioSegment.empty()
                # Convert first audio file to mono/stereo (if needed)
                combined_audio = combined_audio.set_channels(1)

                for file in files:
                    file_path = os.path.join(self.MP4_DIR, file)
                    audio = AudioSegment.from_file(file_path, format="mp4")  # Load .mp4 file

                    # Convert to the same number of channels as combined_audio
                    audio = audio.set_channels(combined_audio.channels)
                    combined_audio += audio  # Concatenate audio files

                output_path = os.path.join(self.COMBINED_DIR, f"{dialogue_num}_combined.mp4")
                combined_audio.export(output_path, format="mp4")  # Save as .mp4
                print(f"Saved: {output_path}")
                nb_of_processed_files += 1

            except Exception as e:
                print(f"Error processing dialogue {dialogue_num}: {e}")
                continue  # Continue with the next dialogue
        print(f"Combining Audio DONE. {nb_of_processed_files} dialogues processed / {nb_of_files} mp4 files.")


    def Combine_AudioText_CSV(self, dataframe, should_add_label_as_text=True):
        '''
        Save the pre-processed data (MELD)
        Save by conversation (every utterance except last one + label of last one)

        should_add_label_as_text: add label as text at the end of each conversation before feeding it to WaveLM/RoBERTa
        '''

        print("Combining Audio and Text csv files....")

        if should_add_label_as_text:
            # Add emotion to the end of utterance
            dataframe['Utterance'] = dataframe['Utterance'] + ' ' + dataframe['Emotion']

        # Extract the last emotion for each dialogue
        last_emotion = dataframe.groupby('Dialogue_ID').apply(lambda x: x.iloc[-1]['Emotion']).reset_index(name='Last_Emotion')

        # Remove the last utterance of each dialogue
        df_filtered = dataframe.groupby('Dialogue_ID').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

        # Merge uttence into one row
        # Group by Dialogue ID and concatenate utterances
        df_combined_v2 = df_filtered.groupby('Dialogue_ID').agg({
            'Utterance': lambda x: ' '.join(x),  # Combine utterances into one string
            'Emotion': lambda x: list(x),  # Keep emotions as a list
            # 'Sentiment': lambda x: list(x)  # Keep sentiments as a list (optional)
        }).reset_index()

        # Merge the last emotion back into df_combined_v2
        df_combined_v2 = df_combined_v2.merge(last_emotion, on='Dialogue_ID', how='left')

        # Add the 'audio_name' column based on Dialogue_ID
        df_combined_v2['audio_name'] = df_combined_v2['Dialogue_ID'].apply(lambda x: f"dia{x}_combined.mp4")

        # Save the processed dataset
        df_combined_v2.to_csv(self.DATA_CSV_FILE, index=False)
        print("Combining Audio and Text csv files DONE")
        print(f"Processed dataset saved as {self.DATA_CSV_FILE}")

        return df_combined_v2


    def process_v2(self, combined_df):
        """
        Read combined_df and extract feature of each dialogue's audio and text.

        """

        print("Extracting AUDIO and TEXT features....")

        # Dictionary to store combined features
        combined_features_list_v2 = []
        combined_features_list_v2_labels = []
        combined_features_list_v2_labels_ID = []

        # Process each dialogue
        for index, row in combined_df.iterrows():
            try:
                label = row["Last_Emotion"]
                dialogue_ID = row["Dialogue_ID"]

                utterance = row["Utterance"]
                mp4_filename = row["audio_name"]
                mp4_path = os.path.join(self.COMBINED_DIR, mp4_filename)

                if not os.path.exists(mp4_path):
                    print(f"Skipping missing file: {mp4_path}")
                    continue

                print(f"Processing {mp4_filename}...")

                audio_features = self.extract_audio_features(mp4_path)
                text_features = self.extract_text_features(utterance)

                combined_features = self.combine_audio_text_features(audio_features, text_features)
                combined_features_list_v2.append(combined_features)
                combined_features_list_v2_labels.append(label)
                combined_features_list_v2_labels_ID.append(dialogue_ID)

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue  # Continue with the next dialogue

        # Convert list to a single tensor
        final_tensor_conversion_v2 = torch.cat(combined_features_list_v2, dim=0)
        final_tensor_conversion_v2_labels = pd.DataFrame(
            combined_features_list_v2_labels,
            index=combined_features_list_v2_labels_ID,
            columns=["Last_Emotion"])
        final_tensor_conversion_v2_labels.index.name = 'Dialogue_ID'

        print("Finished extracting AUDIO and TEXT features and combining them!")

        print(f"Final tensor shape: {final_tensor_conversion_v2.shape}")  # Expected: (num_dialogues, max_seq_length, 1536)

        # Save the tensor to a .pkl file
        # Save labels to a .csv file
        with open(self.CODE_FILENAME, "wb") as f:
            pickle.dump(final_tensor_conversion_v2, f)
            final_tensor_conversion_v2_labels.to_csv(self.CODE_LABEL_FILENAME)

        print(f"Codes/Embeddings Tensor saved to {self.CODE_FILENAME}")
        print(f"Codes/Embeddings Labels saved to {self.CODE_LABEL_FILENAME}")

    #################################### AUDIO FEATURE EXTRACT ####################################
    def extract_audio_from_mp4(self, mp4_path):
        """Extracts audio from an MP4 file and converts it to WAV format."""
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=True).name

        command = ["ffmpeg", "-i", mp4_path, "-ac", "1", "-ar", "16000", "-y", temp_wav]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return temp_wav

    def extract_audio_features(self, mp4_path):
        """Extracts WavLM features while preserving sequence length."""
        wav_path = self.extract_audio_from_mp4(mp4_path)
        waveform, sample_rate = torchaudio.load(wav_path)

        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = transform(waveform)

        inputs = self.feature_extractor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.wavlm_model(**inputs)

        audio_features = outputs.last_hidden_state.cpu()  # (1, audio_seq_length, 768)

        # Resize sequence length to exactly 128
        audio_features = torch.nn.functional.interpolate(audio_features.permute(0, 2, 1), size=128, mode="linear").permute(0, 2, 1)

        return audio_features  # Shape: (1, audio_seq_length resize to 128, 768) [1, 128, 768])


    #################################### TEXT FEATURE EXTRACT ####################################
    def extract_text_features(self, utterance):
        """Extracts RoBERTa features while preserving sequence length."""
        inputs = self.tokenizer(utterance, truncation=True, max_length=128, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)

        text_features = outputs.last_hidden_state.cpu()

        # Resize sequence length to exactly 128
        text_features = torch.nn.functional.interpolate(text_features.permute(0, 2, 1), size=128, mode="linear").permute(0, 2, 1)

        return text_features  # Shape: (1, text_seq_length resize to 128, 768)  [1, 128, 768])


    def combine_audio_text_features(self, audio_features, text_features):
        return torch.cat((audio_features, text_features), dim=-1)  # Shape: (1, 128, 1536)









