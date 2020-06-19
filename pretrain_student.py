# Modified detectron2/tools/plain_train_net.py

import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader
)
from detectron2.data.datasets import load_sem_seg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.evaluation import (
    DatasetEvaluators,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

# Register student model
# We don't use this module, but should import it to register
#   the student meta-architecture.
import student

# Register coco_2017_sem_train and coco_2017_sem_val
import datasets.coco_sem_seg


logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    sem_seg = SemSegEvaluator(
        dataset_name,
        distributed=False,
        num_classes=81,
        output_dir=output_folder,
        ignore_label=80
    )
    vis = datasets.coco_sem_seg.SemSegVisualizer(
        dataset_name,
        num_save=5,
        output_dir=output_folder
    )
    return DatasetEvaluators([sem_seg, vis])


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        TensorboardXWriter(cfg.OUTPUT_DIR)
    ]
    
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            storage.put_scalars(total_loss=losses, **loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                scheduler.step()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    cfg = get_cfg()
    cfg.DATASETS.NUM_CLASSES = -1
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
    else:
        do_train(cfg, model, resume=args.resume)
        
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    results = main(args)
    print(results)
