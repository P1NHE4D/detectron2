#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor

logger = logging.getLogger("detectron2")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.OUTPUT_DIR = "results/training"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("robust_misc_train")
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # use pretrained weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # one class

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def main(args):
    # Register custom datasets
    register_coco_instances(
        "robust_misc_train",
        {},
        "data/robust_misc/annotations/instances_train.json",
        "data/robust_misc/train"
    )
    register_coco_instances(
        "robust_misc_val",
        {},
        "data/robust_misc/annotations/instances_val.json",
        "data/robust_misc/val"
    )

    cfg = setup(args)
    classes = MetadataCatalog.get("robust_misc_train").thing_classes
    print(f"Classes: {classes}")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if args.eval_only:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # testing threshold for model
        cfg.DATASETS.TEST = ("robust_misc_val")

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        args=(args,)
    )
