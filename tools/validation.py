# -*- coding: utf-8 -*-
"""
Created on 2018/8/7 9:26
@author: gzp
Func: rewrite the train.py
"""
import colorsys
import os, sys, yaml, cv2, time
import matplotlib.pyplot as plt
import random

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

def cal_iou(mask1, mask2):
    assert mask1.shape == mask2.shape, "the shape of masks do not match"
    union = np.count_nonzero(mask1>0) + np.count_nonzero(mask2>0) + 1
    intersection = np.count_nonzero(np.bitwise_and(mask1, mask2))
    return float(intersection / union)

# def get_tp_fp_numGT(gt_masks, gt_class_ids, pred_masks, pred_class_ids, pred_scores, iou_threshold=0.5):
#     fps = -1 * np.ones([len(pred_class_ids)])
#     tps = -1 * np.ones([len(pred_class_ids)])
#     scores = pred_scores
#     for i in range(len(pred_class_ids)):
#         pred_mask = pred_masks[i]
#         pred_class_id = pred_class_ids[i]
#         max_iou = 0.
#         index = 0
#         for j in range(len(gt_class_ids)):
#             gt_mask = gt_masks[:, :, j]
#             tmp_iou = cal_iou(pred_mask, gt_mask)
#             if max_iou < tmp_iou:
#                 index = j
#                 max_iou = tmp_iou
#         gt_class_id = gt_class_ids[index]
#         if max_iou < iou_threshold:
#             fps[i] = 1
#         else:
#             if int(gt_class_id) == int(pred_class_id):
#                 tps[i] = 1
#             else:
#                 fps[i] = 1
#     tps = tps[np.argsort(scores)]
#     fps = fps[np.argsort(scores)]
#     num_gts = len(gt_class_ids)
#     return tp, fp, num_gts







def plot_precision_recall(save_dir, image_name, precisions, recalls, mAP):
    plt.plot(recalls, precisions)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title("%s @mAP=%0.2f"%(image_name, mAP))
    plt.savefig(os.path.join(save_dir, image_name))
    plt.close()

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
    mask_dir = cfgs.mask_dir
    yaml_dir = cfgs.yaml_dir

    val_dataset = Dataset()
    val_dataset.load_data(rgb_val_dir, mask_dir, yaml_dir)
    val_dataset.prepare()

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model_path = os.path.join(cfgs.log_dir,'multibox_roi20180808T0911/mask_rcnn_multibox_roi_0016.h5')
    model.load_weights(model_path, by_name=True)
    save_dir = cfgs.val_result_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_PR_dir = cfgs.val_result_pr_dir
    if not os.path.exists(save_PR_dir):
        os.mkdir(save_PR_dir)
    # test
    image_ids = val_dataset.image_ids
    APs = []
    for image_id in image_ids:
        t0 = time.time()
        image_info = val_dataset.image_info[image_id]
        image_path = image_info["path"]
        print("---+"*30)
        print('inference {}...'.format(image_path))
        image_name = os.path.split(image_path)[-1]
        try:
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(val_dataset, inference_config, image_id, use_mini_mask=False)
        except Exception:
            print("ERROR!!!!!!!!!!")
            continue
        t1 = time.time()  # load gt
        results = model.detect([original_image], verbose=1)
        t2 = time.time()  # detect
        r = results[0]
        # Draw precision-recall curve
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
        plot_precision_recall(save_PR_dir, image_name, precisions, recalls, AP)
        t3 = time.time()  # plot PR
        # rois = r['rois']
        # if len(rois) == 0:
        #     print('no any objects.')
        #     continue
        # class_ids = r['class_ids']
        # masks = r['masks']
        # colors = random_colors(len(class_ids))
        # img_mask = original_image
        # for i in range(len(class_ids)):
        #     mask = masks[:, :, i]
        #     img_mask = apply_mask(img_mask, mask, colors[i])
        # cv2.imwrite(os.path.join(save_dir, image_name.replace(".png", "_mask.png")), img_mask)
        t4 = time.time()  # plot mask
        print('>>>> load gt time:%0.2f, detect time:%0.2f, plot PR time:%0.2f' % (t1 - t0,
                                                                                  t2 - t1,
                                                                                  t3 - t2))
        print('>>>> plot mask time:%0.2f, total time:%0.2f' % (t4-t3, time.time()-t0))
    print("mAP:".format(np.mean(APs)))




if __name__ == '__main__':
    inference()