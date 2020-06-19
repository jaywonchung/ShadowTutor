# Remove original OpenMPI
echo $PW | sudo -S apt-get purge -y openmpi-bin

# Build OpenMPI source for cuda support
mkdir -p ~/builds
cd ~/builds
wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.2.tar.gz
gunzip -c openmpi-4.0.2.tar.gz | tar xf -

cd openmpi-4.0.2
./configure --prefix=/home/$USER/.openmpi --with-cuda
make all install

# Set paths
echo '' >> ~/.bashrc
echo '# OpenMPI PATH config' >> ~/.bashrc
echo 'export PATH="$HOME/.openmpi/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
export PATH="$HOME/.openmpi/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.openmpi/lib:$LD_LIBRARY_PATH"

# Return to setup directory
cd $SETUP_DIR
