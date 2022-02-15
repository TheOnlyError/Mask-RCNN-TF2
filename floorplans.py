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

import cv2
import matplotlib.image as mpimg
import numpy as np

# Root directory of the project
from mrcnn.model import MaskRCNN

ROOT_DIR = os.path.abspath("/")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class FloorPlansConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "floorplans"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + wall/opening

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    path = "annotations/rooms_augment/"
    files = os.listdir(path)
    files_set = []
    for file in files:
        files_set.append(file.split(".")[0].split("_")[0])
    files_set = list(dict.fromkeys(files_set))
    random.shuffle(files_set)

    train_size = int(0.8 * len(files_set))

    STEPS_PER_EPOCH = train_size
    VALIDATION_STEPS = len(files_set) - train_size


class FloorPlansDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("floorplans", 1, "opening")
        self.add_class("floorplans", 2, "wall")

        path = "annotations/rooms_augment/"
        files = os.listdir(path)
        files_set = []
        for file in files:
            files_set.append(file.split(".")[0].split("_")[0])
        files_set = list(dict.fromkeys(files_set))
        random.shuffle(files_set)

        train_size = int(0.8 * len(files_set))

        if subset == 'train':
            files_set = files_set[:train_size]
        else:
            files_set = files_set[train_size:]

        for image_id in files_set:
            self.add_image(
                "floorplans",
                image_id=image_id)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        img = mpimg.imread('rooms_augment/{}.jpg'.format(image_id))
        return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "floorplans":
            return info["floorplans"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        img = mpimg.imread('rooms_augment/{}_close_wall.png'.format(image_id))
        return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)


def train():
    model = MaskRCNN(mode="training", config=FloorPlansConfig,
                     model_dir=DEFAULT_LOGS_DIR)

    print("Loading weights")
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    print("Done")

    """Train the model."""
    # Training dataset.
    dataset_train = FloorPlansDataset()
    dataset_train.load_shapes("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FloorPlansDataset()
    dataset_val.load_shapes("val")
    dataset_val.prepare()

    LEARNING_RATE = 1

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=LEARNING_RATE,
                epochs=20,
                augmentation=None,
                layers='heads')

    # print("Train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=LEARNING_RATE,
    #             epochs=40,
    #             augmentation=None,
    #             layers='all')


if __name__ == '__main__':
    seed = int(random.random() * 1000)
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: {0}".format(seed))

    train()
