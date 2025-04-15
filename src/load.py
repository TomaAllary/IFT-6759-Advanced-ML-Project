import os

import pandas as pd
import torch.cuda

from utterances_processing import MELDUtteranceProcessor, IEMOCAPUtteranceProcessor


##########################################################################################
##                                 MELD PROCESSING                                      ##
##########################################################################################

############# PROCESS ##############

# Train set
# myProcessor1 = MELDUtteranceProcessor(
#     project_dir=os.getcwd(),
#     MELD_folder='train',
#     audio_data_folder='train_splits',
#     text_data_csv_filename='train_sent_emo.csv',
#     embeddings_pkl_filename='train_features_per_sentence'
# )
# if myProcessor1.device == 'cuda':
#     torch.cuda.empty_cache()
# myProcessor1.Process_pipeline()

# Dev set
# myProcessor2 = MELDUtteranceProcessor(
#     project_dir=os.getcwd(),
#     MELD_folder='dev',
#     audio_data_folder='dev_splits_complete',
#     text_data_csv_filename='dev_sent_emo.csv',
#     embeddings_pkl_filename='dev_features_per_sentence'
# )
# if myProcessor2.device == 'cuda':
#     torch.cuda.empty_cache()
# myProcessor2.Process_pipeline()

# Test set
# myProcessor3 = MELDUtteranceProcessor(
#     project_dir=os.getcwd(),
#     MELD_folder='test',
#     audio_data_folder='output_repeated_splits_test',
#     text_data_csv_filename='test_sent_emo.csv',
#     embeddings_pkl_filename='test_features_per_sentence',
#     use_cpu=True,
# )
# if myProcessor3.device == 'cuda':
#     torch.cuda.empty_cache()
# myProcessor3.Process_pipeline()




############## LOAD ################
# train_df_name = os.path.join(os.getcwd(), "MELD.Raw", "train", "train_features_per_sentence.pkl")
# dev_df_name = os.path.join(os.getcwd(), "MELD.Raw", "dev", "dev_features_per_sentence.pkl")
# test_df_name = os.path.join(os.getcwd(), "MELD.Raw", "test", "test_features_per_sentence.pkl")
#
# df_features1 = pd.read_pickle(train_df_name)
# df_features2 = pd.read_pickle(dev_df_name)
# df_features3 = pd.read_pickle(test_df_name)
#
# # Check structure
# print(f"Train features shape: {df_features1.shape}")
# print(f"Dev features shape: {df_features2.shape}")
# print(f"Test features shape: {df_features3.shape}")
#
# print(df_features1.head())
# print(df_features1.iloc[0])
# print(df_features1.iloc[0]["Features"].shape)


##########################################################################################
##                               IEMOCAP PROCESSING                                     ##
##########################################################################################

project_root_dir = os.path.dirname(os.getcwd())
myProcessor = IEMOCAPUtteranceProcessor(base_dir=project_root_dir)

# myCSV_df = myProcessor.create_csv() # step 1 -> create "utterances.csv"
# # Heavy features extraction here
# myProcessor.create_pickle(myCSV_df) # step 2 -> create "utterances_raw_features.pkl" (all feature, but no dialogueID)

# dataframe = myProcessor.create_dialogue_sets(seq_length=8) # step 3 -> create a bunch of dialogue of length=8

dataframe = pd.read_pickle(os.path.join(myProcessor.iemocap_dir, "utterances_features_window=8.pkl"))

print(dataframe.head())
print(dataframe.columns)

