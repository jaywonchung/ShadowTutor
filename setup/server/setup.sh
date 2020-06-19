################################################
# Setup environment for server
# Author: Jaewon Chung <jaywonchung@snu.ac.kr>
################################################

# Securely read in the sudo password.
read -s -p "Enter sudo password: " PW

# Install CUDA and CuDNN
read -p "Did you install CUDA and CuDNN [y/n]? " CUDA_INSTALLED
if [ $CUDA_INSTALLED == 'n' ]; then
  source install_cuda.sh
  echo ""
  echo "##########################"
  echo "#         REBOOT         #"
  echo "##########################"
  echo ""
  exit 0
fi

# Paths to remember
REPO_PATH=$(pwd)/../..
SETUP_DIR=$(pwd)

# Install build tools
echo $PW | sudo -S  apt-get update
echo $PW | sudo -S  apt-get -y install build-essential

# Install Anaconda3
source install_anaconda.sh

# Create and activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda create -y -n shadowtutor python==3.6.9
conda activate shadowtutor

# Install dependencies
source install_opencv_dep.sh
source install_pytorch_dep.sh
source install_detectron2_dep.sh

# Install OpenMPI 4.0.2
source install_openmpi.sh

# Install OpenCV 4.1.0
source install_opencv.sh

# Install PyTorch 1.3.0 and torchvision 0.4.2
source install_pytorch.sh

# Install Detectron2 submodule
source install_detectron2.sh

# Install other pip packages used
pip install tqdm matplotlib h5py pycocotools==2.0.0

# Install scripts
python utils.py mpiscript --host server --env shadowtutor
cp "$REPO_PATH/scripts/run_log.sh" ~


echo "setup.sh: Done."
