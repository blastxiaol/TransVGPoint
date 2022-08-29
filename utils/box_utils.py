import torch
from torchvision.ops.boxes import box_area
import math
import numpy as np
from shapely.geometry import Polygon

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # print(box1, box1.shape)
    # print(box2, box2.shape)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy2xywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2.0, (y0 + y1) / 2.0,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert len(boxes1) == len(boxes2)
    ious, unions, areas = [], [], []
    for i in range(len(boxes1)):
        iou, union, corner1, corner2 = box_iou3d(boxes1[i], boxes2[i])
        ious.append(iou)
        unions.append(union)
        x1min, y1min, z1min = corner1.min(dim=0).values
        x1max, y1max, z1max = corner1.max(dim=0).values
        x2min, y2min, z2min = corner2.min(dim=0).values
        x2max, y2max, z2max = corner2.max(dim=0).values
        xmin = min(x1min, x2min)
        xmax = max(x1max, x2max)
        ymin = min(y1min, y2min)
        ymax = max(y1max, y2max)
        zmin = min(z1min, z2min)
        zmax = max(z1max, z2max)
        area = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
        areas.append(area)
    ious = torch.stack(ious)
    unions = torch.stack(unions)
    areas = torch.tensor(areas, device=boxes1.device)
    giou = ious - (areas - unions) / areas
    return giou


def cal_corner_after_rotation(corner, center, r):
        x1, y1 = corner
        x0, y0 = center
        x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
        y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
        return x2, y2

def eight_points(center, size, rotation=0):
    x, y, z = center
    w, l, h = size
    w = w/2
    l = l/2
    h = h/2

    x1, y1, z1 = x-w, y-l, z+h
    x2, y2, z2 = x+w, y-l, z+h
    x3, y3, z3 = x+w, y-l, z-h
    x4, y4, z4 = x-w, y-l, z-h
    x5, y5, z5 = x-w, y+l, z+h
    x6, y6, z6 = x+w, y+l, z+h
    x7, y7, z7 = x+w, y+l, z-h
    x8, y8, z8 = x-w, y+l, z-h

    if rotation != 0:
        x1, y1 = cal_corner_after_rotation(corner=(x1, y1), center=(x, y), r=rotation)
        x2, y2 = cal_corner_after_rotation(corner=(x2, y2), center=(x, y), r=rotation)
        x3, y3 = cal_corner_after_rotation(corner=(x3, y3), center=(x, y), r=rotation)
        x4, y4 = cal_corner_after_rotation(corner=(x4, y4), center=(x, y), r=rotation)
        x5, y5 = cal_corner_after_rotation(corner=(x5, y5), center=(x, y), r=rotation)
        x6, y6 = cal_corner_after_rotation(corner=(x6, y6), center=(x, y), r=rotation)
        x7, y7 = cal_corner_after_rotation(corner=(x7, y7), center=(x, y), r=rotation)
        x8, y8 = cal_corner_after_rotation(corner=(x8, y8), center=(x, y), r=rotation)

    conern1 = torch.tensor([x1, y1, z1])
    conern2 = torch.tensor([x2, y2, z2])
    conern3 = torch.tensor([x3, y3, z3])
    conern4 = torch.tensor([x4, y4, z4])
    conern5 = torch.tensor([x5, y5, z5])
    conern6 = torch.tensor([x6, y6, z6])
    conern7 = torch.tensor([x7, y7, z7])
    conern8 = torch.tensor([x8, y8, z8])
    
    eight_corners = torch.stack([conern1, conern2, conern6, conern5, conern4, conern3, conern7, conern8], axis=0)
    return eight_corners

def cal_inter_area(box1, box2):
    """
    box: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    a=np.array(box1).reshape(4, 2)   
    poly1 = Polygon(a).convex_hull  
    
    b=np.array(box2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    
    union_poly = np.concatenate((a,b))   
    if not poly1.intersects(poly2): 
        inter_area = 0
    else:
        inter_area = poly1.intersection(poly2).area
    return poly1.area, poly2.area, inter_area

def box_iou3d(box1, box2):
    """
    box: [x, y, z, w, h, l, r] center(x, y, z)
    """
    center1 = box1[:3]
    size1 = box1[3:6]
    rotation1 = box1[6] if len(box1) > 6 else 0
    eight_corners1 = eight_points(center1, size1, rotation1)
    
    center2 = box2[:3]
    size2 = box2[3:6]
    rotation2 = box2[6] if len(box2) > 6 else 0
    eight_corners2 = eight_points(center2, size2, rotation2)
    
    area1, area2, inter_area = cal_inter_area(eight_corners1[:4, :2].reshape(-1), eight_corners2[:4, :2].reshape(-1))
    
    h1 = box1[5]
    z1 = box1[2]
    h2 = box2[5]
    z2 = box2[2]
    volume1 = h1 * area1
    volume2 = h2 * area2
    
    bottom1, top1 = z1 - h1/2, z1 + h1/2
    bottom2, top2 = z2 - h2/2, z2 + h2/2
    
    inter_bottom = max(bottom1, bottom2)
    inter_top = min(top1, top2)
    inter_h = inter_top - inter_bottom if inter_top > inter_bottom else 0
    
    inter_volume = inter_area * inter_h
    union_volume = volume1 + volume2 - inter_volume
    
    iou = inter_volume / union_volume
    
    return iou, union_volume, eight_corners1, eight_corners2