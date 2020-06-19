# Download Pytorch source
mkdir -p ~/builds
cd ~/builds
git clone --recursive --branch v1.3.0 http://github.com/pytorch/pytorch
cd pytorch

# Pytorch and torchvision
echo $PW | sudo -S apt-get install -y cmake libopenblas-dev liblapack-dev libjpeg-dev zlib1g-dev
pip install -U setuptools
pip install -r requirements.txt
pip install scikit-build
pip install ninja

# Pillow 6.2.2
cd ~/builds
git clone --branch 6.2.x https://github.com/python-Pillow/Pillow
cd Pillow
python setup.py bdist_wheel
pip install dist/*

