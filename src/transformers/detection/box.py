import torch
import torchvision


def clip_box_within_bound(box, seq_length):
    return box.clamp(min=0, max=seq_length-1)


def remove_small_box(box, min_size):
    length = box[:, 1] - box[:, 0]
    keep = torch.ge(length, min_size)
    keep = torch.where(keep)[0]
    return keep


def expand_dimension(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = torch.zeros_like(x1)
    y2 = torch.ones_like(x2)
    return torch.cat((x1, y1, x2, y2,), dim=1)


def nms(box, score, iou_threshold):
    if box.numel() == 0:
        return torch.empty((0,))
    box_for_nms = expand_dimension(box)
    return torchvision.ops.nms(box_for_nms, score, iou_threshold)


def kmeans_nms(box, score, iou_threshold):
    if box.numel() == 0:
        return torch.empty((0,))
    scores, keep = torch.sort(score, descending=True)
    boxes = box[keep]
    keeps = []
    while len(keep):
        b_iou = iou(boxes, boxes)
        mask = torch.ge(b_iou[0], iou_threshold)
        cluster = iou(boxes[mask], boxes[mask])
        filters = torch.nonzero(mask).squeeze().reshape(-1)
        means = torch.mean(cluster, dim=0)
        keeps.append(keep[filters[torch.argmax(means)]].item())
        boxes = boxes[~mask]
        keep = keep[~mask]
        pass
    return keeps


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def iou_1d(set_a, set_b):
    x1, x2 = set_a  # (left, right)
    y1, y2 = set_b  # (left, right)

    low = max(x1, y1)
    high = min(x2, y2)
    # intersection
    if high - low < 0:
        inter = 0
    else:
        inter = high - low
    # union
    union = (x2 - x1) + (y2 - y1) - inter
    # iou
    iou = inter / union
    return iou


def iou(A, B):
    x_a = expand_dimension(A)
    x_b = expand_dimension(B)
    iou = torchvision.ops.box_iou(x_a, x_b)
    return iou


class Matcher(object):
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )

        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )

        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class BalancedPositiveNegtiveSampler(object):
    def __init__(self, batchsize_per_image, positive_fraction):
        self.batchsize_per_image = batchsize_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idx):
        pos_idx = []
        neg_idx = []
        for matched_idx_per_image in matched_idx:
            positive = torch.where(torch.ge(matched_idx_per_image, 1))[0]
            negative = torch.where(torch.eq(matched_idx_per_image, 0))[0]
            num_pos = int(self.batchsize_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batchsize_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask = torch.zeros_like(
                matched_idx_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idx_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    n = torch.abs(input - target)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    # ctrx = n[:, 0]
    # length = n[:, 1]
    # cond_ctrx = torch.lt(ctrx, beta)
    # loss_ctrx = torch.where(cond_ctrx, 0.5 * ctrx ** 2 / beta, ctrx - 0.5 * beta)
    # cond_len = torch.lt(length, beta)
    # loss_len = torch.where(cond_len, 0.5 * length ** 2 / beta, length - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
