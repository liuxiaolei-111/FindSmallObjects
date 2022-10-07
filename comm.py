import torch
import torch.nn.functional as F
import torch.distributed as dist
import math

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_ious(pred, target, location):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    # 求出预测框左上角右下角
    b1_mins = torch.cat([location[:, 0].unsqueeze(dim=-1) - pred[:, 0].unsqueeze(dim=-1),
                         location[:, 1].unsqueeze(dim=-1) - pred[:, 1].unsqueeze(dim=-1)], dim=-1)
    b1_maxes = torch.cat([location[:, 0].unsqueeze(dim=-1) + pred[:, 2].unsqueeze(dim=-1),
                          location[:, 1].unsqueeze(dim=-1) + pred[:, 3].unsqueeze(dim=-1)], dim=-1)
    b1_xy =  (b1_maxes+b1_mins)/2
    b1_wh = b1_maxes - b1_mins
    # b1_wh_half = b1_wh / 2.


    # 求出真实框左上角右下角
    b2_mins = torch.cat([location[:, 0].unsqueeze(dim=-1) - target[:, 0].unsqueeze(dim=-1),
                         location[:, 1].unsqueeze(dim=-1) - target[:, 1].unsqueeze(dim=-1)], dim=-1)
    b2_maxes = torch.cat([location[:, 0].unsqueeze(dim=-1) + target[:, 2].unsqueeze(dim=-1),
                          location[:, 1].unsqueeze(dim=-1) + target[:, 3].unsqueeze(dim=-1)], dim=-1)
    b2_xy =  (b2_maxes+b2_mins)/2
    b2_wh = b2_maxes - b2_mins
    # b2_wh_half = b2_wh / 2.


    # 求真实框和预测框所有的iou
    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    # 计算中心的差距
    center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), dim=-1)

    # 找到包裹两个框的最小框的左上角和右下角
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))

    # 计算对角线距离
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)
    diou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
        b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
    with torch.no_grad():
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = diou - alpha * v

    return  ciou  #iou, diou,

