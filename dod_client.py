import os
import signal
import logging

import torch
import torch.distributed as dist
from detectron2.modeling import build_model

import dod_common as dod
from datasets.lvs_sem_seg import LVSDataset, LVSTimedDataset


device_name = 'CLIENT'
logger = logging.getLogger(device_name)


@torch.no_grad()
def main(args, cfg, student, dataloader):
    stride = cfg.DOD.MIN_STRIDE
    step = stride    # iters since last key frame
    received = True # indicates whether parameter receive is done

    # Tensor to hold next stride value
    stride_tensor = torch.ones(1, dtype=torch.uint8) * cfg.DOD.MIN_STRIDE

    # List of student parameters to receive (only those that require_grad)
    recv_keys = list(dict(filter(lambda p: p[1].requires_grad, student.named_parameters())).keys())
    state_dict = student.state_dict()
    logger.info(f'Parameter tensors to recv: {len(recv_keys)} of {len(state_dict)}')

    # frame: torch.Tensor with shape [1080, 1920, 3], BGR and dtype torch.uint8
    for idx, frame in zip(range(args.max_frames), dataloader):
        logger.info(f'Frame {idx+1}/{len(dataloader)}, Step {step}/{stride}')

        # Key frame
        if step == stride:
            # Send key frame to teacher
            dist.broadcast(frame, src=1, async_op=True)
            logger.info('frame: send op registered')

            # Receive updated parameters
            for i, key in enumerate(recv_keys):
                dist.broadcast(state_dict[key], src=0, async_op=True)
            logger.info('param: recv op registered')
            
            # Receive next stride tensor
            recv_job = dist.broadcast(stride_tensor, src=0, async_op=True)
            logger.info('strid: recv op registered')

            step = 0
            received = False

        # Inference with student model, same for key and non-key frames
        im = frame.cuda().type(torch.float32).permute(2, 0, 1)
        pred = student.inference(im)

        step += 1

        # Wait for updated weights
        if not received and step == cfg.DOD.MIN_STRIDE:
            logger.info('Wait for updated weights')
            recv_job.wait()

        # Check for receive complete
        if not received and recv_job.is_completed():
            stride = stride_tensor.item()
            student.load_state_dict(state_dict)
            logger.info(f'Received updated weights after delay {step}')
            logger.info(f'Next stride is {stride}')
            received = True

        if args.save_hdf5:
            dod.save_hdf5(frame, pred.detach())
        if args.save_video:
            dod.save_video(frame, pred.detach())


if __name__ == "__main__":
    # Setup args, config, and logger
    args, cfg = dod.setup(device_name)

    # Initialize OpenMPI backend process group
    #   rank 0: SERVER (teacher model)
    # v rank 1: CLIENT (student model)
    dod.init_process(device_name, desired_rank=1)

    # Create and load model
    student = dod.get_student(args, cfg, logger)

    # Create dataloader
    if args.time:
        dataset = LVSTimedDataset(args.video_path, logger)
    else:
        dataset = LVSDataset(args.video_path, logger)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, pin_memory=True)

    # Both nodes must have finished setup to pass this barrier
    dod.barrier(device_name)

    # Begin system
    logger.critical(f'Begin ShadowTutor')
    main(args, cfg, student, dataloader)

    # Terminate OpenMPI
    #   Without this, OpenMPI hangs even if the maximum
    #   number of frames has been reached.
    logger.critical('Job done. Terminating!')
    os.kill(os.getppid(), signal.SIGKILL)
