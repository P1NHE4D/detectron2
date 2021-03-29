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
import random

import torch
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from detectron2.utils.visualizer import Visualizer

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
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

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
    print(cfg)

    if args.eval_only:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ("robust_misc_val",)
        predictor = DefaultPredictor(cfg)
        dataset_dicts = DatasetCatalog.get("robust_misc_val")
        for d in random.sample(dataset_dicts, 10):
            img = cv2.imread(d["file_name"])
            outputs = predictor(img)
            v = Visualizer(img)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            os.makedirs("visualizations", exist_ok=True)
            cv2.imwrite(f"visualizations/{d['file_name']}", v.get_image())
        return

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
