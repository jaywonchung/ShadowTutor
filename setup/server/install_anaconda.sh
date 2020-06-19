# Download anaconda install script
mkdir -p /tmp
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

# Invoke installation
bash Anaconda3-2019.10-Linux-x86_64.sh

# Remove anaconda install script
rm Anaconda3-2019.10-Linux-x86_64.sh
