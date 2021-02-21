# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import yaml
import paddle
import paddle.distributed as dist

paddle.seed(2)

from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model_extend
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import init_model
import tools.program as program
from ppocr.modeling.architectures.extend_model import JointVisDet, JointVisDetFineGrained
from tools.infer import utility
from tools.extend.util import ProtestDataset, ProtestDatasetEval

dist.get_world_size()


def main(config, config_detrec, device, logger, vdl_writer):
    # init dist environment
    if config['Global']['distributed']:
        dist.init_parallel_env()
    # build dataloader
    ProtestDataset
    # build model

    # build loss

    # build optimize

    # build



    # build dataloader
    train_dataloader = build_dataloader(config, 'Train', device, logger)
    if len(train_dataloader) == 0:
        logger.error(
            'No Images in train dataset, please check annotation file and path in the configuration file'
        )
        return

    if config['Eval']:
        valid_dataloader = build_dataloader(config, 'Eval', device, logger)
    else:
        valid_dataloader = None

    # build model
    model = build_model_extend(config['Architecture']["extend_args"], config_detrec)


    if config['Global']['distributed']:
        model = paddle.DataParallel(model)

    # build loss
    loss_class = build_loss(config['Architecture']["extend_args"])

    # build optim
    optimizer, lr_scheduler = build_optimizer(
        config['Optimizer'],
        epochs=config['Global']['epoch_num'],
        step_each_epoch=len(train_dataloader),
        parameters=model.parameters())

    # build metric
    eval_class = build_metric(config['Metric'])
    # load pretrain model
    pre_best_model_dict = init_model(config, model, logger, optimizer)

    logger.info('train dataloader has {} iters, valid dataloader has {} iters'.
                format(len(train_dataloader), len(valid_dataloader)))
    # start train
    program.train(config, train_dataloader, valid_dataloader, device, model,
                  loss_class, optimizer, lr_scheduler, post_process_class,
                  eval_class, pre_best_model_dict, logger, vdl_writer)


def test_reader(config, device, logger):
    loader = build_dataloader(config, 'Train', device, logger)
    import time
    starttime = time.time()
    count = 0
    try:
        for data in loader():
            count += 1
            if count % 1 == 0:
                batch_time = time.time() - starttime
                starttime = time.time()
                logger.info("reader: {}, {}, {}".format(
                    count, len(data[0]), batch_time))
    except Exception as e:
        logger.info(e)
    logger.info("finish reader: {}, Success!".format(count))


if __name__ == '__main__':
    config_detrec = utility.parse_args()
    config_general, device, logger, vdl_writer = program.preprocess(is_train=True)
    main(config_general, config_detrec, device, logger, vdl_writer)
    # test_reader(config, device, logger)
