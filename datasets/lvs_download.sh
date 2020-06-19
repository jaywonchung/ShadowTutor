# Download the LVS dataset. Expected to be run in this directory.
# Give the name of the video, e.g. 'badminton', as the first argument.
wget -e robots=off -r -nH --cut-dirs 1 --no-check-certificate -np 'https://olimar.stanford.edu/hdd/lvsdataset/'$1
