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

echo '# OpenMPI PATH config' >> ~/.bashrc
echo 'PATH="$PATH:/home/$USER/.openmpi/bin"' >> ~/.bashrc
echo 'LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"' >> ~/.bashrc
source ~/.bashrc

# Return to setup directory
cd $SETUP_DIR
