# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch
import skimage
from skimage import io, transform
from typing import Tuple, List

from .utils import dboxes300_coco, Encoder


def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    img = skimage.img_as_float(io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img


def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = transform.resize(img, (input_width, res))
    if aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = transform.resize(img, (res, input_height))
    if aspect == 1:
        imgScaled = transform.resize(img, (input_width, input_height))
    return imgScaled


def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img


def prepare_tensor(inputs, fp16=False):
    NHWC = np.array(inputs)
    NCHW = np.swapaxes(np.swapaxes(NHWC, 1, 3), 2, 3)
    tensor = torch.from_numpy(NCHW)
    tensor = tensor.contiguous()
    tensor = tensor.cuda()
    tensor = tensor.float()
    if fp16:
        tensor = tensor.half()
    return tensor


def prepare_input(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize(img)
    return img


def decode_results(predictions: Tuple[torch.Tensor, torch.Tensor]):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ploc, plabel = [val.float() for val in predictions]
    results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)
    return [[pred.detach().cpu() for pred in detections] for detections in results]


def pick_best(detections: List[torch.Tensor], threshold: float = 0.3):
    bboxes, classes, confidences = detections
    best = torch.where(confidences > threshold)[0]
    # best = np.argwhere(confidences > threshold)[:, 0]
    return [pred[best] for pred in detections]


def get_coco_object_dictionary():
    import os
    file_with_coco_names = "category_names.txt"

    if not os.path.exists(file_with_coco_names):
        print("Downloading COCO annotations.")
        import urllib
        import zipfile
        import json
        import shutil
        urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                                   "cocoanno.zip")
        with zipfile.ZipFile("cocoanno.zip", "r") as f:
            f.extractall()
        print("Downloading finished.")
        with open("annotations/instances_val2017.json", 'r') as COCO:
            js = json.loads(COCO.read())
        class_names = [category['name'] for category in js['categories']]
        open("category_names.txt", 'w').writelines([c + "\n" for c in class_names])
        os.remove("cocoanno.zip")
        shutil.rmtree("annotations")
    else:
        class_names = open("category_names.txt").readlines()
        class_names = [c.strip() for c in class_names]
    return class_names
