# To set up the virtual environement

1. Create environement using python 3.10:
conda create -n some_env_name python=3.10
conda activate some_env_name

2. Install torch using the version supporting CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Install the requirement
Navigate to the right folder then do:
pip install -r requirement.txt
