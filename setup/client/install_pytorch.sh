# Build and install PyTorch 1.3.0: took around 12 hours on Nano
export USE_NCCL=0
export PYTORCH_CUDA_ARCH_LIST='5.3;6.2;7.2'
export PYTORCH_BUILD_VERSION=1.3.0
export PYTORCH_BUILD_NUM=1
python setup.py bdist_wheel
pip install dist/*

# Build and install torchvision 0.4.2
cd ~/builds
git clone --branch v0.4.2 https://github.com/pytorch/vision
cd vision
python setup.py bdist_wheel
pip install dist/*

# Return to setup directory
cd $SETUP_DIR
