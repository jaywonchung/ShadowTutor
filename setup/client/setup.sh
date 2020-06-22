################################################
# Setup environment for client (Jetson Nano)
# Author: Jaewon Chung <jaywonchung@snu.ac.kr>
################################################

# Install an extra 5GB swap disk: required for Jetson Nano
read -p "Did you install extra swap [y/n]? " SWAP_CREATED
if [ $SWAP_CREATED == 'n' ]; then	
  echo "Run"
  echo "    sudo create_swap.sh 5"
  echo "and reboot Jetson."
  exit 0
fi

# Securely read in the sudo password.
read -s -p "Enter sudo password: " PW

# Remember repo path
REPO_PATH=$(pwd)/../..
SETUP_DIR=$(pwd)

# Set maximum performance
echo $PW | sudo -S nvpmodel -m 0

# Install pip and venv
echo $PW | sudo -S apt-get update
echo $PW | sudo -S apt-get install -y python3-venv python3-pip
pip3 install -U pip
echo $PW | sudo -S sed 's/from pip import main/from pip import __main__/' /usr/bin/pip3
echo $PW | sudo -S sed 's/sys.exit(main())/sys.exit(__main__._main())/' /usr/bin/pip3

# Create and activate python venv
python3 -m venv ~/shadowtutor
source ~/shadowtutor/bin/activate
pip install -U pip

# Install dependencies
source install_opencv_dep.sh
source install_pytorch_dep.sh
source install_detectron2_dep.sh

# Install OpenMPI 4.0.2
source install_openmpi.sh

# Install PyTorch 1.3.0 and torchvision 0.4.2
source install_pytorch.sh

# Install OpenCV 4.1.0
source install_opencv.sh

# Install Detectron2 submodule 
source install_detectron2.sh

# Install other pip packages used
pip install tqdm matplotlib h5py

# Install scripts
python "$REPO_PATH/utils.py" mpiscript --host client --env shadowtutor


echo "setup.sh: Done."
