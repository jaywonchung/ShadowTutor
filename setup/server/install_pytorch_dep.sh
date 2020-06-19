# Pytorch
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
conda install -y -c pytorch magma-cuda100 # Assuming CUDA 10.0
pip install Pillow==6.2.2

# Torchvision
echo $PW | sudo -S apt-get -y install libjpeg-dev zlib1g-dev

# CUDA toolkit
conda install -y -c pytorch cudatoolkit=10.1
