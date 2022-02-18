"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import sys
import time

import cv2
import matplotlib.image as mpimg
import numpy as np

# Root directory of the project
from floorplans import FloorPlanInferenceConfig, FloorPlansDataset
from mrcnn.model import MaskRCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT_DIR = os.path.abspath("data/")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils, visualize


def predict():
    config = FloorPlanInferenceConfig()
    model = MaskRCNN(mode="inference", config=config,
                     model_dir=DEFAULT_LOGS_DIR)
    weights_path = DEFAULT_LOGS_DIR + '/model2.h5'
    print("Loading weights")
    model.load_weights(weights_path, by_name=True)
    print("Done")

    # Training dataset.
    dataset_train = FloorPlansDataset()
    dataset_train.load_shapes("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FloorPlansDataset()
    dataset_val.load_shapes("val")
    dataset_val.prepare()

    dataset = dataset_train
    i = 0
    samples = 6
    ids = dataset.image_ids
    random.shuffle(ids)
    for image_id in ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        r = model.detect([image], verbose=0)[0]
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # mpimg.imsave("result" + timestr + ".jpg", r.astype(np.uint8))
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=True, show_mask_polygon=True, show_caption=False)
        i += 1
        if i >= samples:
            break

if __name__ == '__main__':
    seed = int(random.random() * 1000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: {0}".format(seed))

    tic = time.time()
    predict()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
