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


def draw_mask(num_obj, mask, height, width, image):
    for index in range(num_obj):
        for i in range(width):
            for j in range(height):
                at_pixel = image.getpixel((i, j))
                if at_pixel == index + 1:
                    mask[j, i, index] = 1
    return mask

def load_mask(mask_path, yaml_path):
    img = Image.open(mask_path)
    width, height = img.size
    num_obj = np.max(img)
    mask = np.zeros([height, width, num_obj], dtype=np.uint8)
    mask = draw_mask(num_obj, mask, height, width, img)
    labels = yaml.load(open(yaml_path).read())['label_names']
    labels_form = []
    for i in range(len(labels)):
        for name in label_name_dict.values():
            if labels[i].find(name) != -1:
                labels_form.append(name)
    class_ids = np.array([list(label_name_dict.values()).index(s) + 1 for s in labels_form])
    return mask, class_ids.astype(np.int32)
def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def load_image(image_path):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):


    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    tp = np.count_nonzero(pred_match > -1)
    fp = len(gt_class_ids) - tp
    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps, tp, fp
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
    files = get_files(rgb_val_dir)
    APs = []
    num_gts = 0
    tps = 0
    fps = 0
    for file in files:
        image_path = os.path.join(rgb_val_dir, file)
        mask_path = os.path.join(mask_dir, file)
        yaml_path = os.path.join(yaml_dir, file.replace('png', 'yaml'))
        print("---+"*30)
        print('inference {}...'.format(image_path))
        gt_mask, gt_class_id = load_mask(mask_path, yaml_path)
        num_gts += len(gt_class_id)
        gt_bbox = extract_bboxes(gt_mask)
        original_image = load_image(image_path)
        t1 = time.time()  # load gt
        results = model.detect([original_image], verbose=1)
        t2 = time.time()  # detect
        r = results[0]
        if len(r['class_ids']) == 0:
            continue
        # Draw precision-recall curve
        AP, precisions, recalls, overlaps, tp, fp = compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        tps += tp
        fps += fp
        APs.append(AP)
        plot_precision_recall(save_PR_dir, file, precisions, recalls, AP)
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
        print('>>>> detect time:%0.2f, plot PR time:%0.2f' % (t2 - t1,
                                                              t3 - t2))
        print('>>>> plot mask time:%0.2f, total time:%0.2f' % (t4-t3, time.time()-t1))
    print("mAP:{}".format(np.mean(APs)))
    print("recall:{}, mAP:{}".format(tps/num_gts, tps/(tps+fps)))




if __name__ == '__main__':
    inference()