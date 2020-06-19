# Add NVIDIA repository
release="ubuntu"$(lsb_release -sr | sed -e "s/\.//g")
echo $PW | sudo -S apt-key adv --fetch-keys "http://developer.download.nvidia.com/compute/cuda/repos/"$release"/x86_64/7fa2af80.pub"
echo $PW | sudo -S sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/'$release'/x86_64 /" > /etc/apt/sources.list.d/nvidia-cuda.list'
echo $PW | sudo -S sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/'$release'/x86_64 /" > /etc/apt/sources.list.d/nvidia-machine-learning.list'
echo $PW | sudo -S apt update

# Install CUDA 10.0
echo $PW | sudo -S apt-get -y install cuda-10-0

# Install CuDNN 7
echo $PW | sudo -S apt-get -y install libcudnn7-dev
