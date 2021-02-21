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

from paddle import nn
from ppocr.modeling.transforms import build_transform
from ppocr.modeling.backbones import build_backbone
from ppocr.modeling.necks import build_neck
from ppocr.modeling.heads import build_head

import copy
import torch
import torchvision.models as models
from tools.infer import utility
from tools.infer.predict_system import TextSystem
from tools.infer.predict_det import TextDetector
import statistics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


__all__ = ['JointVisDet', 'JointVisDetFineGrained']


def vis_model():
    model = models.resnet50(pretrained=True)
    return model


def det_model(args):
    text_det = TextDetector(args)
    return text_det


def detrec_model(args):
    text_sys = TextSystem(args)
    return text_sys


def extract_dt_boxes_fts(dt_boxes):
    dt_boxes_cp = copy.deepcopy(dt_boxes)
    # n_bbox, aver_bbox, std_bbox
    bbox_n = len(dt_boxes_cp)
    bbox_long = [r-l for (l, r, t, b) in dt_boxes_cp]  # ck
    bbox_high = [t-b for (l, r, t, b) in dt_boxes_cp]  # ck
    bbox_aver_1 = sum(bbox_long) / len(bbox_long)
    bbox_aver_2 = sum(bbox_high) / len(bbox_high)
    bbox_std_1 = statistics.stdev(bbox_long)
    bbox_std_2 = statistics.stdev(bbox_high)
    return np.array([bbox_n, bbox_aver_1, bbox_aver_2, bbox_std_1, bbox_std_2])


class Tfidf:
    def __init__(self, corpus):
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit_transform(self.corpus)
    def __call__(self, document):
        return self.vectorizer.transform(document)


class JointVisDet(torch.nn.Module):
    def __init__(self, idim=1003, odim=12, args=None):
        super(JointVisDet, self).__init__()
        self.vis_model = vis_model()
        self.det_model = det_model(args)
        self.head = torch.nn.Linear(idim, odim)
    def forward(self, img):
        vis_out = self.vis_model(img)
        dt_boxes = self.det_model(img)
        det_out = torch.from_numpy(extract_dt_boxes_fts(dt_boxes)) # ck
        jot_out = torch.cat((vis_out, det_out), 1)
        out = self.head(jot_out)
        return out


class JointVisDetFineGrained(torch.nn.Module):
    def __init__(self, idim=1512, odim=5, corpus=None, args=None):
        super(JointVisDetFineGrained, self).__init__()
        self.vis_model = vis_model()
        self.detrec_model = detrec_model(args)
        self.head = torch.nn.Linear(idim, odim)
        self.tfidf = Tfidf(corpus)

    def forward(self, img):
        vis_out = self.vis_model(img)
        dt_boxes, rec_res = self.detrec_model(img) # (box_n, 4) (box_n, 2)
        document = [" ".join(d) for d in rec_res]
        detrec_out = self.tfidf([document])   # [1, fts_n]

        jot_out = torch.cat((vis_out, detrec_out[0]), 1)
        out = self.head(jot_out)
        return out