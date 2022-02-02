import torch
from iou import intersection_over_union
def nms(
    bboxes,
    iou_threshold,
    threshold,
    box_format = "corners"):
    # predictions  = [[1,0.9,x1,x2,y1,y2] ...]

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes,key = lambda x: x[1],reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        # this bboxes keeps other bboxes beside the chosen box and it's copy
        bboxes = [
            box 
            for box in bboxes 
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms