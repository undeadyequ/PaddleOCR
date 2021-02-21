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

import copy

__all__ = ['build_model']

def build_model(config):
    from .base_model import BaseModel
    
    config = copy.deepcopy(config)
    module_class = BaseModel(config)
    return module_class


def build_model_extend(model_ext_name, config):
    from .extend_model import JointVisDetFineGrained, JointVisDet
    config = copy.deepcopy(config)
    module_class = eval(model_ext_name)(args=config)
    return module_class
