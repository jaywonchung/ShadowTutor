import os
import time
import logging
import argparse
from datetime import datetime

import cv2
import h5py
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

import detectron2
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer

import student
import datasets.lvs_sem_seg


class TimeAverageMeter:
    """
    Keeps track of time average between tic and toc
    Synchronizes CUDA, hence incurs big overhead
    """
    def __init__(self, name, device_name, sync_cuda=False, warmup_steps=20):
        self.name = name
        self.device_name = device_name
        self.sync_cuda = sync_cuda
        self.start = time.perf_counter()
        self.warmup_steps = warmup_steps
        self.steps_passed = 0
        self.average = 0

    @property
    def steps(self):
        return max(self.steps_passed - self.warmup_steps, 0)
    
    def tic(self):
        self.start = time.perf_counter()
    
    def pause(self):
        self.pause_time = time.perf_counter()
    
    def resume(self):
        self.start = time.perf_counter() - (self.pause_time - self.start)
    
    def toc(self):
        if self.sync_cuda:
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - self.start) * 1000 # ms
        self.steps_passed += 1
        logger = logging.getLogger(self.device_name)
        if self.steps_passed <= self.warmup_steps:
            logger.info(
                f'{self.name} during warmup {self.steps_passed}/{self.warmup_steps}: took {int(elapsed)}ms'
            )
        else:
            self.average = (self.average * (self.steps - 1) + elapsed) / self.steps
            logger.info(f'{self.name}: took {int(elapsed)}ms, mean {int(self.average)}ms')


def get_args():
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument('--threshold', type=float, default=0.8, help='Student accuracy threshold, THRESHOLD in paper')

    # video data
    parser.add_argument('--video-path', type=str, required=True, help='Path to video, set on client')
    parser.add_argument('--max-frames', type=int, default=int(1e6), help='The number of frames to run on')

    # student
    parser.add_argument('--lr', type=float, default=0.01, help='Student distillation learning rate')
    parser.add_argument('--full', action='store_true', help='Whether to disable partial distillation')

    # save
    # Note: Logs are saved by redirecting the outputs of mpirun to a file. Refer to scripts/run_log.sh.
    parser.add_argument('--save-hdf5', action='store_true', help='Save frame predictions to hdf5 format')
    parser.add_argument('--save-video', action='store_true', help='Save frame predictions to mp4 video')

    # for experiments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--exp-name', type=str, default='', help='Will be used as the name of the ouptut directory')
    parser.add_argument('--time', action='store_true', help='Time various operations at the cost of performance')
    parser.add_argument('--deterministic', action='store_true', help='Makes all operations deterministic at the cost of performance')
    parser.add_argument('--random-seed', type=int, default=1753, help='Random seed to use when deterministic mode is on')

    args = parser.parse_args()
    return parser, args


def get_config(device_name):
    cfg = get_cfg()
    if device_name == 'SERVER':
        cfg.merge_from_file('configs/Teacher_dod.yaml')
        cfg.MODEL.WEIGHTS = \
            model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    elif device_name == 'CLIENT':
        cfg.DOD = CfgNode()
        cfg.DOD.MIN_STRIDE = 8
        cfg.DOD.MAX_STRIDE = 64
        cfg.DOD.NUM_UPDATES = 8
        cfg.DATASETS.NUM_CLASSES = -1   # will be overrided by config file
        cfg.merge_from_file('configs/Student_dod.yaml')
    return cfg


def set_deterministic(seed):
    import random
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup(device_name):
    # Parse commandline arguments
    parser, args = get_args()

    # Set config
    cfg = get_config(device_name)

    # Set output directory
    if args.save_hdf5 or args.save_video:
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_DIR,
            os.path.basename(args.video_path).split('.')[0],
            args.exp_name if args.exp_name else f"{str(datetime.now())[:-7].replace(' ','-')}"
        )

    # Create ouptut directory if necessary
    if args.save_hdf5 or args.save_video:
        if not os.path.exists(os.path.split(cfg.OUTPUT_DIR)[0]):
            os.mkdir(os.path.split(cfg.OUTPUT_DIR)[0])
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

    # Set logger
    logger = logging.getLogger(device_name)
    logger.setLevel(eval(f'logging.{args.log_level}'))

    formatter = logging.Formatter('[%(asctime)s %(name)s] %(message)s')

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    
    # Create output HDF5 file
    if args.save_hdf5:
        save_hdf5.f = h5py.File(os.path.join(cfg.OUTPUT_DIR, 'predictions.hdf5'), 'w')

    # Create opencv video writer
    if args.save_video:
        cap = cv2.VideoCapture(args.video_path)
        save_video.writer = cv2.VideoWriter(
            filename=os.path.join(cfg.OUTPUT_DIR, 'predictions.mp4'),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=cap.get(cv2.CAP_PROP_FPS),
            frameSize=(1280, 720),
            isColor=True
        )
        cap.release()
        
    # CUDNN benchmark
    if cfg.CUDNN_BENCHMARK:
        if args.deterministic:
            logger.critical('Ignoring CUDNN_BENCHMARK: True in cfg since --deterministic is on')
            torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
        logger.critical('Enabling CUDNN benchmark')
    else:
        torch.backends.cudnn.benchmark = False

    # Set deterministic mode. Lowers performance significantly.
    if args.deterministic:
        set_deterministic(args.random_seed)

    # Max frames
    if args.max_frames != parser.get_default('max_frames'):
        logger.critical(f'Maximum number of frames set to {args.max_frames}')

    logger.critical(args)
    logger.critical('Setup for args, config, output_dir, and logger complete')
    return args, cfg


def get_student(args, cfg, logger):

    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        
        def parameters(self, *args, **kwargs):
            return self.model.parameters(*args, **kwargs)
            
        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)
        
        def state_dict(self, *args, **kwargs):
            return self.model.state_dict(*args, **kwargs)
        
        def load_state_dict(self, *args, **kwargs):
            self.model.load_state_dict(*args, **kwargs)

        @torch.no_grad()
        def inference(self, im : torch.Tensor) -> torch.Tensor:
            return self.model.single_forward(im)
        
        def forward(self, im : torch.Tensor) -> torch.Tensor:
            return self.model.single_forward(im)
        
        def distill_step(self, pred, label, weight):
            self.model.zero_grad()
            self.model.single_spatially_weighted_CE(pred, label, weight).backward()
            self.optimizer.step()
    
    class ModelTimerWrapper:
        def __init__(self, model):
            self.model = model
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            self.inference_timer = TimeAverageMeter('Student inference', logger.name, True)
            self.forward_timer = TimeAverageMeter('Student forward', logger.name, True)
            self.distill_timer = TimeAverageMeter('Student distill', logger.name, True)
        
        def parameters(self, *args, **kwargs):
            return self.model.parameters(*args, **kwargs)
            
        def named_parameters(self, *args, **kwargs):
            return self.model.named_parameters(*args, **kwargs)
        
        def state_dict(self, *args, **kwargs):
            return self.model.state_dict(*args, **kwargs)
        
        def load_state_dict(self, *args, **kwargs):
            self.model.load_state_dict(*args, **kwargs)
        
        @torch.no_grad()
        def inference(self, im : torch.Tensor) -> torch.Tensor:
            self.inference_timer.tic()
            pred = self.model.single_forward(im)
            self.inference_timer.toc()
            return pred
        
        def forward(self, im : torch.Tensor) -> torch.Tensor:
            self.forward_timer.tic()
            pred = self.model.single_forward(im)
            self.forward_timer.toc()
            return pred
        
        def distill_step(self, pred, label, weight):
            self.distill_timer.tic()
            self.model.zero_grad()
            self.model.single_spatially_weighted_CE(pred, label, weight).backward()
            self.optimizer.step()
            self.distill_timer.toc()

    model = build_model(cfg)

    # Load model weights
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))

    # Disable gradients for frontend network
    if not args.full:
        model.in_conv1.requires_grad_(False)
        model.in_conv2.requires_grad_(False)
        model.enc1.requires_grad_(False)
        model.enc2.requires_grad_(False)
        model.enc3.requires_grad_(False)
        model.dec1.requires_grad_(False)

    if args.time:
        model = ModelTimerWrapper(model)
    else:
        model = ModelWrapper(model)
    logger.info('Loaded student model from checkpoint')

    return model


def get_teacher(args, cfg, logger):

    class ModelWrapper:
        def __init__(self, model):
            self.model = model
        
        @torch.no_grad()
        def __call__(self, img : torch.Tensor) -> dict:
            img = img.permute(2, 0, 1).type(torch.float32)
            input = {'image':img, 'height': 720, 'width': 1280}
            return self.model([input])[0]
    
    class ModelTimerWrapper:
        def __init__(self, model):
            self.model = model
            self.inference_timer = TimeAverageMeter('MRCNN inference', 'SERVER', True)
        
        @torch.no_grad()
        def __call__(self, img : torch.Tensor) -> dict:
            self.inference_timer.tic()
            img = img.permute(2, 0, 1).type(torch.float32)
            input = {'image':img, 'height': 720, 'width': 1280}
            output = self.model([input])[0]
            self.inference_timer.toc()
            return output

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    if args.time:
        model = ModelTimerWrapper(model)
    else:
        model = ModelWrapper(model)
    logger.critical('Loaded teacher from model zoo')
    return model


def get_next_ratio(threshold, accuracy):
    """
    The piecewise linear function used to determine the ratio in algorithm 2
    """
    if accuracy > threshold:
        ratio = (accuracy + 1.0 - 2.0 * threshold) / (1.0 - threshold)
    else:
        ratio = accuracy / threshold
    return ratio


def get_next_stride(args, cfg, current, accuracy):
    """Algorithm 2, key frame striding"""
    ratio = get_next_ratio(args.threshold, accuracy)
    return min(cfg.DOD.MAX_STRIDE, max(cfg.DOD.MIN_STRIDE, int(ratio * current)))


def init_process(device_name, desired_rank):
    """Initialize OpenMPI. Server, client, shake hands."""
    logger = logging.getLogger(device_name)
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    assert rank == desired_rank, f'{device_name} has rank {rank}'
    logger.critical(f'{device_name} initialized with rank {rank}')


def barrier(device_name):
    """
    Synchronization barrier.
    In addition, test if the server and the client can communicate.
    """
    logger = logging.getLogger(device_name)
    ping = torch.ones(1)
    logger.critical('Meet sync barrier')
    dist.broadcast(ping, src=0)
    logger.critical('Pass sync barrier')


def draw_prediction(frame, pred):
    """Draw semantic segmentation prediction on input image."""
    pred = pred.argmax(dim=0).type(torch.uint8).cpu().numpy()
    vis = Visualizer(frame.numpy()[:,:,::-1], MetadataCatalog.get('lvs_sem_seg'))
    vis = vis.draw_sem_seg(pred, alpha=0.5)
    return vis.get_image()


def save_hdf5(frame, pred):
    """Dump prediction to HDF5 format."""
    sem_seg_pred = draw_prediction(frame, pred) # np.array with shape [720, 1280, 3], RGB
    save_hdf5.f.create_dataset(str(time.time()), dtype=h5py.h5t.STD_U8BE, data=sem_seg_pred)


def save_video(frame, pred):
    """Save prediction to mp4."""
    sem_seg_pred = draw_prediction(frame, pred) # np.array with shape [720, 1280, 3], RGB
    save_video.writer.write(sem_seg_pred[:,:,::-1])


@torch.no_grad()
def miou(pred, label):
    """The mean Intersection over Union metric."""
    pred = F.one_hot(pred.argmax(dim=0), num_classes=9)
    label = F.one_hot(label, num_classes=9)
    intersection = (pred * label).sum(dim=(0,1))
    union = (pred + label).sum(dim=(0,1)) - intersection
    valid_class_mask = union > 0
    mean_iou = torch.mean(intersection[valid_class_mask].float() / union[valid_class_mask])
    return mean_iou.item()
