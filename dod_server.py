import os
import copy
import logging

import torch
import torch.distributed as dist

import dod_common as dod


device_name = 'SERVER'
logger = logging.getLogger(device_name)


def main(args, tcfg, scfg, teacher, student):
    # Tensor to hold key frame
    frame = torch.empty((720, 1280, 3), dtype=torch.uint8, device='cuda')

    # Tensor to hold next stride value
    stride_tensor = torch.ones(1, dtype=torch.uint8) * scfg.DOD.MIN_STRIDE

    # List of student parameters to send (only those that require_grad)
    send_keys = list(dict(filter(lambda p: p[1].requires_grad, student.named_parameters())).keys())
    state_dict = student.state_dict()
    logger.info(f'Parameter tensors to send: {len(send_keys)} of {len(state_dict)}')

    # Set box expansion tensor
    dilate = 0.15   # dilate 15% each side
    s = dilate / 2 + 1
    loss_mask.expander = torch.cuda.FloatTensor([
        [s, 0, 1-s, 0], [0, s, 0, 1-s], [1-s, 0, s, 0], [0, 1-s, 0, s]
    ])

    # The server never stops
    while True:
        # Recieve key frame from client
        dist.broadcast(frame, src=1, async_op=False)
        logger.info('frame: recv complete')

        # Perform teacher inference
        teacher_pred = teacher(frame)

        # Perform student inference
        im = frame.type(torch.float32).permute(2, 0, 1)
        pred = student.forward(im)

        # Make semantic segmentation label
        target = sem_seg_label(teacher_pred).long()
        
        # Accuracy of first forward
        acc = dod.miou(pred, target)
        skip = (acc > args.threshold)
        logger.info(f'First time accuracy: mIOU {acc} -> {"skip" if skip else "perform"} distillation')

        # Perform distillation
        distill_step = -1
        if not skip:
            # Make loss weight
            weight = loss_mask(teacher_pred)

            # Distillation
            best_acc = acc
            best_student = copy.deepcopy(student)
            for distill_step in range(scfg.DOD.NUM_UPDATES):
                student.distill_step(pred, target, weight)
                pred = student.forward(im)
                acc = dod.miou(pred, target)
                logger.info(f'Distill step {distill_step+1}/{scfg.DOD.NUM_UPDATES}: mIOU {acc}')
                if acc > args.threshold:
                    break
                if acc > best_acc:
                    best_acc = acc
                    if distill_step != scfg.DOD.NUM_UPDATES-1:
                        best_student = copy.deepcopy(student)
            else: # finished loop without breaking
                if acc < best_acc: # final accuracy was not the best
                    acc = best_acc
                    student = best_student
                    logger.info(f'Restore to best accuracy {acc}')
        
        logger.info(f'Key frame accuracy: {acc}')
        logger.info(f'Key frame distill steps: {distill_step+1}')
        
        # Adapt next stride based on accuracy
        stride_tensor[0] = dod.get_next_stride(args, scfg, stride_tensor.item(), acc)
        
        # Send updated student parameters and stride
        state_dict = student.state_dict()
        for key in send_keys:
            dist.broadcast(state_dict[key], src=0, async_op=True)
        dist.broadcast(stride_tensor, src=0, async_op=True)
        logger.info('all  : send op registered')


# From the output of the teacher, create the semantic segmentation label
# Note that Mask R-CNN is an instance segmentation model, not a semantic 
#   segmentation model
def sem_seg_label(pred):
    classes = pred['instances'].pred_classes
    masks = pred['instances'].pred_masks

    class_masks = [
        torch.any(masks[(classes==0).nonzero().squeeze(1)], dim=0).bool(),  # person
        torch.any(masks[(classes==1).nonzero().squeeze(1)], dim=0).bool(),  # bicycle
        torch.any(masks[((classes==2)|(classes==5)|(classes==7)).nonzero().squeeze(1)], dim=0).bool(),  # auto = car + bus + truck
        torch.any(masks[(classes==14).nonzero().squeeze(1)], dim=0).bool(), # bird
        torch.any(masks[(classes==16).nonzero().squeeze(1)], dim=0).bool(), # dog
        torch.any(masks[(classes==17).nonzero().squeeze(1)], dim=0).bool(), # horse
        torch.any(masks[(classes==20).nonzero().squeeze(1)], dim=0).bool(), # elephant
        torch.any(masks[(classes==23).nonzero().squeeze(1)], dim=0).bool() # giraffe
    ]
    label = torch.ones((720, 1280), dtype=torch.uint8, device='cuda') * 8
    for class_index, class_mask in enumerate(class_masks):
        label[class_mask] = class_index

    return label


# Scale the loss values inside and around objects by a factor of 5.0
def loss_mask(pred):
    boxes = pred['instances'].pred_boxes.tensor

    expanded_boxes = torch.matmul(boxes, loss_mask.expander).int()
    expanded_boxes[:, 0].clamp_(min=0, max=1280)
    expanded_boxes[:, 1].clamp_(min=0, max=720)
    expanded_boxes[:, 2].clamp_(min=0, max=1280)
    expanded_boxes[:, 3].clamp_(min=0, max=720)

    mask = torch.ones((720, 1280), device='cuda')
    for x1, y1, x2, y2 in expanded_boxes:
        mask[y1:y2+1, x1:x2+1] = 5.0
    
    return mask


if __name__ == "__main__":
    # Setup args, config, and logger
    args, cfg = dod.setup(device_name)

    # Initialize OpenMPI backend process group
    # v rank 0: SERVER (teacher model)
    #   rank 1: CLIENT (student model)
    dod.init_process(device_name, desired_rank=0)

    # Create and load model
    teacher = dod.get_teacher(args, cfg, logger)
    student_cfg = dod.get_config('CLIENT')
    student = dod.get_student(args, student_cfg, logger)

    # Synchronization barrier
    dod.barrier(device_name)

    # Begin DOD
    logger.critical(f'Begin ShadowTutor')
    main(args, cfg, student_cfg, teacher, student)
