"""
Handles the LVS dataset.
"""
import os

import cv2
import torch
from torch.utils.data import Dataset
from detectron2.data import MetadataCatalog

import dod_common as dod


# Register the LVS semantic segmentation dataset.
meta = MetadataCatalog.get('lvs_sem_seg')
# Corresponds to COCO class indices 0, 1, (2,5,7), 14, 16, 17, 20, 23
meta.stuff_classes = ['person', 'bicycle', 'auto', 'bird', 'dog', 'horse', 'elephant', 'giraffe', 'background']
meta.stuff_colors = [[220, 20, 60], [119, 11, 32], [0, 0, 142], [165, 42, 42], [0, 226, 252], [182, 182, 255], [110, 76, 0], [72, 0, 118], [255,255,255]]


class LVSDataset(Dataset):
    """
    Opens the given video path.
    Outputs one unit8 Tensor of shape [1080, 1920, 3] each iteration.
    """
    def __init__(self, video_path, logger):
        assert os.path.exists(video_path), f'{video_path} does not exist'
        assert os.path.isfile(video_path), f'{video_path} is not a file'

        self.video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f'Video name: {video_name}')
        logger.info(f'Video size: {height} x {width}')
        logger.info(f'FPS: {fps}')
        logger.info(f'Number of frames: {self.length}')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.video.read()[1]


class LVSTimedDataset(Dataset):
    """
    Opens the given video path.
    Outputs one unit8 Tensor of shape [1080, 1920, 3] each iteration.
    Times frame fetch time.
    """
    def __init__(self, video_path, logger):
        assert os.path.exists(video_path), f'{video_path} does not exist'
        assert os.path.isfile(video_path), f'{video_path} is not a file'
        self.video_path = video_path

        self.video = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path)
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f'Video name: {video_name}')
        logger.info(f'Video size: {height} x {width}')
        logger.info(f'FPS: {fps}')
        logger.info(f'Number of frames: {self.length}')

        self.fetch_timer = dod.TimeAverageMeter('Frame fetch', 'CLIENT')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        self.fetch_timer.tic()
        frame = self.video.read()[1]
        self.fetch_timer.toc()
        return frame
