  # -*- coding: utf-8 -*-
"""
Created on 2018/8/7 9:26
@author: gzp
Func: rewrite the train.py
"""
import os, sys, yaml, cv2
import numpy as np
import re
from PIL import Image

sys.path.append('../')
from config import cfgs
from mrcnn.config import Config
from mrcnn import utils
from utils.utils import get_files
import mrcnn.model as modellib

os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.train_gpus
dataset_name = cfgs.dataset_name
assert dataset_name in ["Box", "multibox_roi"], "please check your dataset_name in config/cfg.py!!!"
if dataset_name == "Box":
    from datas.box.dataset import label_name_dict
elif dataset_name == "multibox_roi":
    from datas.multibox_roi.dataset import label_name_dict
else:
    sys.exit(0)


def find_last_model():
    # find the latest model.
    log_dir = cfgs.log_dir
    files = get_files(log_dir)
    files = [file for file in files if file.endswith('.h5')]
    if len(files) == 0:
        return None
    files = sorted(files)
    file_path = os.path.join(log_dir, files[-1])
    return file_path


class DataConfig(Config):
    """
    Configuration for training on the Box dataset.
    Derives from the base Config class and overrides values specific
    to the Box dataset.
    """
    # Give the configuration a recognizable name
    NAME = dataset_name
    # settle the gpu count and images per gpu by config in libs.
    GPU_COUNT = len(cfgs.train_gpus.split(','))
    IMAGES_PER_GPU = cfgs.images_per_gpu

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(label_name_dict.keys())  # including the background

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = cfgs.image_min_dim
    IMAGE_MAX_DIM = cfgs.image_max_dim

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class Dataset(utils.Dataset):
    # image is read from a mask file, which is 16 bits in depth.
    # if you want to do something, we recommend that you use PIL.Image.
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # get instance names from yaml file, the names correspond to the colors in mask file
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # overwrite the draw_mask
    # transfer the 2D mask images into num_obj channels.
    # each channel represents an instance.
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # add a function named load_data, for data providing, no need to modify.
    def load_data(self, image_dir, mask_dir, yaml_dir):
        """
        load box
        :param image_dir: image dir
        :param mask_dir: mask dir
        :param yaml_dir: yaml dir
        :return:
        """
        # Add classes
        for label in label_name_dict.keys():
            name = label_name_dict[label]
            self.add_class(dataset_name, label, name)
        # get all rgb files
        image_files = [file for file in os.listdir(image_dir) if file.endswith('png') or file.endswith('jpg')]
        image_id = 0
        for image_file in image_files:
            yaml_file = image_file.replace('png', 'yaml')
            mask_file = image_file
            yaml_path = os.path.join(yaml_dir, yaml_file)
            mask_path = os.path.join(mask_dir, mask_file)
            image_path = os.path.join(image_dir, image_file)
            img = cv2.imread(image_path)
            height, width = img.shape[0], img.shape[1]
            if not (os.path.exists(yaml_path) and os.path.exists(mask_path)):
                # when yaml or mask file do not exist, continue without adding the images
                continue
            self.add_image(source=dataset_name, image_id=image_id, path=image_path,
                           width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)
            image_id += 1

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        return info["width"], info["height"]

    # overwrite the load_mask
    # Generate instance masks of the given image ID.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            for name in label_name_dict.values():
                pattern = re.compile(r'{}[0-9]'.format(name))
                if re.match(pattern, labels[i]) is not None:
                    labels_form.append(name)
                    break

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def train():
    # summary and trained weights saved path
    MODEL_DIR = cfgs.log_dir

    # basic config
    rgb_train_dir = cfgs.rgb_train_dir
    mask_dir = cfgs.mask_dir
    yaml_dir = cfgs.yaml_dir

    # train set preparing
    dataset_train = Dataset()
    dataset_train.load_data(rgb_train_dir, mask_dir, yaml_dir)
    dataset_train.prepare()
    # config
    config = DataConfig()
    config.display()
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
    if not cfgs.train_from_coco and find_last_model() is not None:
        model.load_weights(model.find_last(), by_name=True)
    else:
        model_path = os.path.join(cfgs.root_path, "mask_rcnn_coco.h5")
        if cfgs.train_from_coco:
            if not os.path.exists(model_path):
                # download the pre-trained weights base on coco
                utils.download_trained_weights(model_path)
        model.load_weights(model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE, epochs=1, layers='heads')
    model.train(dataset_train, dataset_train, learning_rate=config.LEARNING_RATE / 10, epochs=50, layers="all")


if __name__ == '__main__':
    train()
