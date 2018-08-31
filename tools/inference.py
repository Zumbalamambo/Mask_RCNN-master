# -*- coding: utf-8 -*-
"""
Created on 2018/8/7 9:26
@author: gzp
Func: rewrite the train.py
"""
import colorsys
import os, sys, yaml, cv2, time
import random
import tensorflow as tf
import copy
import numpy as np
import skimage.io
from PIL import Image
sys.path.append('../')

from config import cfgs
from mrcnn.config import Config
from mrcnn import utils, visualize
from utils.utils import get_files
import mrcnn.model as modellib
os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.test_gpus
dataset_name = cfgs.dataset_name
assert dataset_name in ["Box", "multibox_roi", "multibox_roi_border"], "please check your dataset_name in config/cfg.py!!!"
if dataset_name == "Box":
    from datas.box.dataset import label_name_dict
elif dataset_name == "multibox_roi":
    from datas.multibox_roi.dataset import label_name_dict
elif dataset_name == "multibox_roi_border":
    from datas.multibox_roi_border.dataset import label_name_dict
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

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def get_min_area_rectangle(coordinate_list, mode=1):
    """
    calculate the rectangle with minimum area.
    :param coordinate_list: [[x1, y1], [x2, y2], [x3, y3]]
    :param mode: 0, ((xc, yc), (w, h)); 1, ((x1,y1),(x2,y2),(x3,y3),(x4,y4)) clock wise
    :return:list of the result, float
    """
    cnt = np.array(coordinate_list)    # change to numpy array
    rect = cv2.minAreaRect(cnt)        # get the ((xc,yc),(w,h), theta)
    if mode == 1:
        box = cv2.boxPoints(rect)      # get four coordinates
        box = box.tolist()
        return box
    else:
        rect = [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]
        return rect
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            np.array(image[:, :, c], np.float32) * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image.astype(np.uint8)

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
                if labels[i].find(name) != -1:
                    labels_form.append(name)
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

class InferenceConfig(DataConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def inference():
    # summary and trained weights saved path
    MODEL_DIR = cfgs.log_dir
    class_names = list(label_name_dict.values())
    # basic config
    rgb_val_dir = cfgs.rgb_val_dir
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model_path = os.path.join(cfgs.log_dir,'multibox_roi_border20180809T0232/mask_rcnn_multibox_roi_border_0100.h5')
    model.load_weights(model_path, by_name=True)
    save_dir = cfgs.val_result_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # test
    files = get_files(rgb_val_dir)
    for file in files:
        print("---+"*30)
        print('inference {}...'.format(file))
        image_path = os.path.join(rgb_val_dir, file)
        img = cv2.imread(image_path)
        h, w = img.shape[0], img.shape[1]
        t0 = time.time()
        original_image = skimage.io.imread(image_path)

        results = model.detect([original_image], verbose=1)

        # Run RPN sub-graph
        # pillar = model.keras_model.get_layer("ROI").output  # node to start searching from
        rpn = model.run_graph([original_image], [
            ("rpn_class", model.keras_model.get_layer("rpn_class").output),
            # ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
            # ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
            # ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
            # ("post_nms_anchor_ix", model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")),
            ("proposals", model.keras_model.get_layer("ROI").output)
        ])
        proposals = rpn['proposals']
        limits = min(100, len(proposals[0]))
        # limits = len(proposals[0])
        proposals = rpn['proposals'][0, :limits, :] * np.array([h, w, h, w])
        print('proposals shape:', proposals.shape)
        img_rpn = copy.deepcopy(img)
        colors_rpn = random_colors(limits)
        for i in range(len(proposals)):
            proposal = proposals[i]
            color = colors_rpn[i]
            color = tuple([v * 255 for v in color])
            x1, y1, x2, y2 = tuple([int(val) for val in proposal])
            cv2.rectangle(img_rpn, (x1, y1), (x2, y2), color, 1)
        rpn_img_path = os.path.join(save_dir, file.replace(".png", "_rpn.png"))
        cv2.imwrite(rpn_img_path, img_rpn)
        t1 = time.time()
        r = results[0]
        rois = r['rois']
        if len(rois) == 0:
            print('no any objects.')
            continue
        class_ids = r['class_ids']
        masks = r['masks']
#         index = np.where((class_ids==1.)|(class_ids==2.))[0]
        index = np.where((class_ids==1.)|(class_ids==2.)|(class_ids==6.))[0]
        if len(index) == 0:
            continue

        rois = rois[index, :]
        class_ids = class_ids[index]
        masks = masks[:, :, index]
        colors = random_colors(len(class_ids))
        img_mask = copy.deepcopy(img)
        for i in range(len(class_ids)):
            mask = masks[:, :, i]
            img_mask = apply_mask(img_mask, mask, colors[i])
            mask_val_index = np.where(mask == True)
            coords = list(map(lambda y, x: [x, y], mask_val_index[0], mask_val_index[1]))
            try:
                box_list = get_min_area_rectangle(coords, mode=1)
            except Exception:
                print(np.max(mask))
                continue
            left_bottom_coord = (int(box_list[0][0]+box_list[2][0])//2, int(box_list[0][1]+box_list[2][1])//2)
            cv2.putText(img, label_name_dict[int(class_ids[i])],
                        left_bottom_coord,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1, color=(0, 0, 255), thickness=1)
            cv2.putText(img_mask, label_name_dict[int(class_ids[i])],
                        left_bottom_coord,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1, color=(0, 0, 255), thickness=1)

            for ii in range(len(box_list)):
                x1 = int(box_list[ii][0])
                y1 = int(box_list[ii][1])
                x2 = int(box_list[(ii + 1) % 4][0])
                y2 = int(box_list[(ii + 1) % 4][1])
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, file.replace(".png", "_mask.png")), img_mask)
        cv2.imwrite(os.path.join(save_dir, file.replace(".png", ".png")), img)
        print('>>>> detect time:{}, post processing time:{}, total time:{}'.format(t1 - t0,
                                                                                   time.time() - t1,
                                                                                   time.time() - t0))


if __name__ == '__main__':
    inference()