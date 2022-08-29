import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy, box_iou3d


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    if pred_boxes.shape[1] == 4:
        pred_boxes = xywh2xyxy(pred_boxes)
        pred_boxes = torch.clamp(pred_boxes, 0, 1)
        gt_boxes = xywh2xyxy(gt_boxes)
        iou = bbox_iou(pred_boxes, gt_boxes)
        accu50 = torch.sum(iou >= 0.5) / float(batch_size)
        accu25 = torch.sum(iou >= 0.25) / float(batch_size)
    else:
        iou = torch.zeros((batch_size, ), dtype=torch.float32, device=pred_boxes.device)
        for i in range(batch_size):
            pred_box = pred_boxes[i]
            gt_box = gt_boxes[i]
            if pred_boxes.shape[1] == 7:
                pred_box[6] = pred_box[6] * (2*np.pi) - np.pi
                gt_box[6] = gt_box[6] * (2*np.pi) - np.pi
            iou[i] = box_iou3d(pred_box, gt_box)[0]
        accu50 = torch.sum(iou >= 0.5) / float(batch_size)
        accu25 = torch.sum(iou >= 0.25) / float(batch_size)

    return iou, accu50, accu25

def trans_vg_eval_test(pred_boxes, gt_boxes, parameters=None):
    pred_boxes = pred_boxes.cpu()
    gt_boxes = gt_boxes.cpu()
    
    if pred_boxes.shape[1] == 4:
        results = []
        pred_boxes = xywh2xyxy(pred_boxes)
        pred_boxes = torch.clamp(pred_boxes, 0, 1)
        gt_boxes = xywh2xyxy(gt_boxes)
        iou = bbox_iou(pred_boxes, gt_boxes)
        accu50 = torch.sum(iou >= 0.5)
        accu25 = torch.sum(iou >= 0.25)
    else:
        batch_size = pred_boxes.shape[0]
        iou = torch.zeros((batch_size, ), dtype=torch.float32, device=pred_boxes.device)
        results = []
        for i in range(batch_size):
            result = dict()
            para = parameters[i]
            idx = para.idx
            target = para.target
            sentence = para.sentence
            vmax = torch.tensor(para.vmax)
            vmin = torch.tensor(para.vmin)
            vsize = torch.tensor(para.vsize)
            point_cloud_path = para.point_cloud_path
            image_path = para.image_path
            calib_path = para.calib_path
  
            pred_box = pred_boxes[i]
            pred_box[:3] = pred_box[:3] * (vmax - vmin) + vmin
            pred_box[3:6] = pred_box[3:6] * vsize
            gt_box = torch.tensor(target, device=gt_boxes.device)
            if pred_boxes.shape[1] == 7:
                pred_box[6] = pred_box[6] * (2*np.pi) - np.pi
                gt_box[6] = gt_box[6] * (2*np.pi) - np.pi
            each_iou = box_iou3d(pred_box, gt_box)[0]
            iou[i] = each_iou
         
            result['idx'] = idx
            result['target'] = target
            result['sentence'] = sentence
            result['point_cloud_path'] = point_cloud_path
            result['iou'] = each_iou.item()
            result['image_path'] = image_path
            result['calib_path'] = calib_path
            result['pred_box'] = pred_box.cpu().numpy().tolist()
            results.append(result)
        accu50 = torch.sum(iou >= 0.5)
        accu25 = torch.sum(iou >= 0.25)
    return accu50, accu25, iou, results
