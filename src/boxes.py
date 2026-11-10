"""Box utilities: YOLO label loading, IoU, greedy matching and area coverage.

All boxes are pixel coordinates in xyxy format: (x1, y1, x2, y2).
"""
import os

import numpy as np


def load_yolo_labels(path, img_w, img_h):
    """Read a YOLO label file (class xc yc w h, normalised 0-1) and return pixel xyxy boxes."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, 4))
    arr = np.loadtxt(path, ndmin=2)
    xc, yc = arr[:, 1] * img_w, arr[:, 2] * img_h
    w, h = arr[:, 3] * img_w, arr[:, 4] * img_h
    return np.c_[xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def box_iou(a, b):
    """IoU matrix between boxes a (N x 4) and b (M x 4)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_boxes(pred, conf, gt, iou_thr=0.5):
    """Greedy matching, highest confidence first, as in standard detection metrics.

    Returns (tp, gt_hit): tp[i] is True if prediction i matched a label box;
    gt_hit[j] is True if label box j was found.
    """
    pred = np.asarray(pred, float).reshape(-1, 4)
    gt = np.asarray(gt, float).reshape(-1, 4)
    tp = np.zeros(len(pred), bool)
    gt_hit = np.zeros(len(gt), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp, gt_hit
    order = np.argsort(-np.asarray(conf, float))
    iou = box_iou(pred[order], gt)
    for k, i in enumerate(order):
        row = np.where(gt_hit, -1.0, iou[k])
        j = int(row.argmax())
        if row[j] >= iou_thr:
            gt_hit[j] = True
            tp[i] = True
    return tp, gt_hit


def occupancy_mask(boxes, img_w, img_h, scale=8):
    """Low-resolution mask of the image area covered by any box (1 cell = scale x scale px)."""
    mask = np.zeros((int(np.ceil(img_h / scale)), int(np.ceil(img_w / scale))), bool)
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
        mask[max(int(y1 // scale), 0):max(int(np.ceil(y2 / scale)), 0),
             max(int(x1 // scale), 0):max(int(np.ceil(x2 / scale)), 0)] = True
    return mask


def rect_coverage(mask, rect, scale=8):
    """Share (0-1) of a rectangle's area that the occupancy mask marks as covered."""
    x1, y1, x2, y2 = rect
    h, w = mask.shape
    ys, xs = min(max(int(y1 // scale), 0), h), min(max(int(x1 // scale), 0), w)
    ye = min(max(int(np.ceil(y2 / scale)), ys + 1), h)
    xe = min(max(int(np.ceil(x2 / scale)), xs + 1), w)
    sub = mask[ys:ye, xs:xe]
    return float(sub.mean()) if sub.size else 0.0
