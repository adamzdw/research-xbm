# encoding: utf-8

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import argparse
import torch
import os
import numpy as np

from ret_benchmark.data.evaluations.eval import AccuracyCalculator
from ret_benchmark.utils.feat_extractor import feat_extractor
from ret_benchmark.config import cfg
from ret_benchmark.data import build_data
from ret_benchmark.losses import build_loss
from ret_benchmark.modeling import build_model
from ret_benchmark.solver import build_lr_scheduler, build_optimizer
from ret_benchmark.utils.logger import setup_logger
from ret_benchmark.utils.checkpoint import Checkpointer
from tensorboardX import SummaryWriter


import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def test(cfg):
    logger = setup_logger(name="Train", level=cfg.LOGGER.LEVEL)
    logger.info(cfg)
    model = torch.load('net.pkl')
    model.eval()
    logger.info("Validation")

    val_loader = build_data(cfg, is_train=False)

    labels = val_loader[0].dataset.label_list
    labels = np.array([int(k) for k in labels])
    feats = feat_extractor(model, val_loader[0], logger=logger)
    ret_metric = AccuracyCalculator(include=("precision_at_1", "mean_average_precision_at_r", "r_precision"), exclude=())
    ret_metric = ret_metric.get_accuracy(feats, feats, labels, labels, True)
    mapr_curr = ret_metric['mean_average_precision_at_r']

    logger.info(f"ret_metric: {ret_metric}")

    


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description="Train a retrieval network")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="config file", default=None, type=str
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    test(cfg)
