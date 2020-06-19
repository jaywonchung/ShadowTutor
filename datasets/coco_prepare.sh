# This script prepares the COCO dataset.
# Expected to be run on in this directory.

# Download and unzip COCO2017
mkdir -p coco

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/train2017.zip

unzip annotations_trainval2017.zip
unzip val2017.zip
unzip train2017.zip

rm annotations_trainval2017.zip
rm val2017.zip
rm train2017.zip

# Make semantic segmentation labels
conda activate off
cd ..
python datasets/coco_sem_seg.py
