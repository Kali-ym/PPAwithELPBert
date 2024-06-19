import math
import torch


class BoxCoder(object):
    def __init__(self, bbox_xform_clip=math.log(1000. / 16)):
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):

        targets = self.encode_boxes(reference_boxes, proposals)
        return targets

    def encode_boxes(self, reference_boxes, proposals):
        proposals_x1 = proposals[:, 0].unsqueeze(1)
        proposals_x2 = proposals[:, 1].unsqueeze(1)

        reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
        reference_boxes_x2 = reference_boxes[:, 1].unsqueeze(1)

        ex_lengths = proposals_x2 - proposals_x1
        ex_ctr_x = proposals_x1 + 0.5 * ex_lengths

        gt_lengths = reference_boxes_x2 - reference_boxes_x1
        gt_ctr_x = reference_boxes_x1 + 0.5 * gt_lengths

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_lengths
        targets_dl = torch.log(gt_lengths / ex_lengths)

        targets = torch.cat((targets_dx, targets_dl), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        pred_boxes = self.decode_single(
            rel_codes, concat_boxes
        )

        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 2)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        boxes = boxes.to(rel_codes.dtype)

        lengths = boxes[:, 1] - boxes[:, 0]
        ctr_x = boxes[:, 0] + 0.5 * lengths

        dx = rel_codes[:, 0::2]
        dl = rel_codes[:, 1::2]

        dl = torch.clamp(dl, max=self.bbox_xform_clip)

        pred_ctr_x = dx * lengths[:, None] + ctr_x[:, None]
        pred_l = torch.exp(dl) * lengths[:, None]

        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_l.device) * pred_l
        pred_boxes2 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_l.device) * pred_l

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2), dim=2).flatten(1)
        return pred_boxes
