# To set up the virtual environement

1. Create environement using python 3.10:
conda create -n some_env_name python=3.10
conda activate some_env_name

2. Install torch using the version supporting CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Install the requirement
Navigate to the right folder then do:
pip install -r requirement.txt


# CNN Architecture
### Creating processed datasets
- From https://github.com/declare-lab/MELD/tree/master, you can download the zip file or simply click [here](http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz)
- Make sure you unzip Train and Test set to have this exact file hierarchy (highlighted files/folders): 
 ![Alt text](https://github.com/TomaAllary/IFT-6759-Advanced-ML-Project/blob/main/README_IMGs/MELD_files_hierarchy.png)
** If a xxx_sent_emo.csv is missing from zip file, you can retrieve it from github page's data folder directly**
- Open *cnn_emotion_prediction_MELD.ipynb* notebook and run the second cell once for both train and test informations. You should now have a hierarchy similar to above image.

### To train new models
Run all cell in order. Make sure your new model info is used in sections "Train one model" and/or "Hyperparameter search"

### To simply evaluate/use existing models
- Make sure the datasets were created
- Skip all the cells down to "TEST EXISTING MODELS"
- Run cells under "TEST EXISTING MODELS" and "LOAD & EVALUATE MODELS". You will need your wandb api key to download the models.

Choose one of the 3 existing model (see commented lines):
*loaded_model = load_wandb_model("best_models", "v1", "cnn_model_1")
loaded_model = load_wandb_model("best_models", "v2", "cnn_model_2")
loaded_model = load_wandb_model("best_models", "v4", "cnn_model_3")*

