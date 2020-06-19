# Set cmake prefix
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# Build PyTorch 1.3.0 from source
# We need to do this to enable OpenMPI compiled with CUDA support
mkdir -p ~/builds
cd ~/builds
git clone --recursive --branch v1.3.0 https://github.com/pytorch/pytorch
cd pytorch
python setup.py bdist_wheel
pip install dist/*

# Build torchvision 0.4.2 from source
cd ~/builds
git clone --branch v0.4.2 https://github.com/pytorch/vision torchvision
cd torchvision
python setup.py bdist_wheel
pip install dist/*

# Return to setup directory
cd $SETUP_DIR
