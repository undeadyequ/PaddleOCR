"""
created by: Donghyeon Won
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import argparse


class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        protest = self.label_frame.iloc[idx, 1:2].to_numpy().astype('float')
        violence = self.label_frame.iloc[idx, 2:3].to_numpy().astype('float')
        visattr = self.label_frame.iloc[idx, 3:].to_numpy().astype('float')
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}

        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample

class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def paddle_infer_args(parser):
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    group1 = parser.add_argument_group('paddle')
    # params for prediction engine
    group1.add_argument("--use_gpu", type=str2bool, default=True)
    group1.add_argument("--ir_optim", type=str2bool, default=True)
    group1.add_argument("--use_tensorrt", type=str2bool, default=False)
    group1.add_argument("--use_fp16", type=str2bool, default=False)
    group1.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    group1.add_argument("--image_dir", type=str)
    group1.add_argument("--det_algorithm", type=str, default='DB')
    group1.add_argument("--det_model_dir", type=str)
    group1.add_argument("--det_limit_side_len", type=float, default=960)
    group1.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    group1.add_argument("--det_db_thresh", type=float, default=0.3)
    group1.add_argument("--det_db_box_thresh", type=float, default=0.5)
    group1.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    group1.add_argument("--max_batch_size", type=int, default=10)
    # EAST parmas
    group1.add_argument("--det_east_score_thresh", type=float, default=0.8)
    group1.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    group1.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    group1.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    group1.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    group1.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    group1.add_argument("--rec_algorithm", type=str, default='CRNN')
    group1.add_argument("--rec_model_dir", type=str)
    group1.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    group1.add_argument("--rec_char_type", type=str, default='ch')
    group1.add_argument("--rec_batch_num", type=int, default=6)
    group1.add_argument("--max_text_length", type=int, default=25)
    group1.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt")
    group1.add_argument("--use_space_char", type=str2bool, default=True)
    group1.add_argument(
        "--vis_font_path", type=str, default="./doc/fonts/simfang.ttf")
    group1.add_argument("--drop_score", type=float, default=0.5)

    # params for text classifier
    group1.add_argument("--use_angle_cls", type=str2bool, default=False)
    group1.add_argument("--cls_model_dir", type=str)
    group1.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    group1.add_argument("--label_list", type=list, default=['0', '180'])
    group1.add_argument("--cls_batch_num", type=int, default=6)
    group1.add_argument("--cls_thresh", type=float, default=0.9)

    group1.add_argument("--enable_mkldnn", type=str2bool, default=False)
    group1.add_argument("--use_pdserving", type=str2bool, default=False)

    return parser



def protest_train_parser(parser):
    group = parser.add_argument_group('protest_train')
    group.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    group.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    group.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    group.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    group.add_argument("--epochs",
                        type = int,
                        default = 100,
                        help = "number of epochs",
                        )
    group.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    group.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    group.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    group.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    group.add_argument('--resume',
                        default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    group.add_argument('--change_lr',
                        action = "store_true",
                        help = "Use this if you want to \
                        change learning rate when resuming")
    group.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    return parser
