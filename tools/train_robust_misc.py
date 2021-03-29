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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
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
    register_coco_instances(
        "robust_misc_test_stage1",
        {},
        "data/robust_misc_testing/Stage_1/annotations/instances_test.json",
        "data/robust_misc_testing/Stage_1/test"
    )
    register_coco_instances(
        "robust_misc_test_stage2",
        {},
        "data/robust_misc_testing/Stage_2/annotations/instances_test.json",
        "data/robust_misc_testing/Stage_2/test"
    )
    register_coco_instances(
        "robust_misc_test_stage3",
        {},
        "data/robust_misc_testing/Stage_3/annotations/instances_test.json",
        "data/robust_misc_testing/Stage_3/test"
    )

    cfg = setup(args)
    print(cfg)

    if args.vis_only:
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
            res = v.get_image()
            f_name = d['file_name'].split("/")[-1]
            cv2.imwrite(f"visualizations/{f_name}", res)
        return

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if args.eval_only:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        model = build_model(cfg)
        DetectionCheckpointer(model).load(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"))
        os.makedirs("evaluation/stage_1", exist_ok=True)
        os.makedirs("evaluation/stage_2", exist_ok=True)
        os.makedirs("evaluation/stage_3", exist_ok=True)
        cfg.DATASETS.TEST = ("robust_misc_test_stage1", "robust_misc_test_stage2", "robust_misc_test_stage3")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        ev1 = COCOEvaluator("robust_misc_test_stage1", output_dir="evaluation/stage_1")
        st1_loader = build_detection_test_loader(cfg, "robust_misc_test_stage1")
        inference_on_dataset(model, st1_loader, ev1)
        ev2 = COCOEvaluator("robust_misc_test_stage2", output_dir="evaluation/stage_2")
        st2_loader = build_detection_test_loader(cfg, "robust_misc_test_stage2")
        inference_on_dataset(model, st2_loader, ev2)
        ev3 = COCOEvaluator("robust_misc_test_stage3", output_dir="evaluation/stage_3")
        st3_loader = build_detection_test_loader(cfg, "robust_misc_test_stage3")
        inference_on_dataset(model, st3_loader, ev3)
        return

    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--vis_only", type=bool, default=False)
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
