#########################################################
# This sctipt creates a swap area with a designated size
# given via commandline argument.
# Author: Jaewon chung <jaywonchung@snu.ac.kr>
# Tested on: 
##########################################################

if [ -z "$1" ] 
then
    echo "Provide the size of swap (in unit G)."
    echo "For example, ./create_swap.sh 5 will create a 5GB swap space."
    exit 1
fi

SWAP_FILE_PATH=/mnt/"$1"GB.swap

fallocate -l "$1"G $SWAP_FILE_PATH
chmod 600 $SWAP_FILE_PATH
mkswap $SWAP_FILE_PATH
swapon $SWAP_FILE_PATH
echo "$SWAP_FILE_PATH none swap sw 0 0" >> /etc/fstab
echo "vm.swappiness=60" >> /etc/sysctl.conf

echo "###############################"
echo "                               "
echo "        REBOOT SYSTEM!         "
echo "                               "
echo "###############################"
