import torch
import torch.nn as nn
import torch.nn.functional as F

from .box import BalancedPositiveNegtiveSampler, Matcher, clip_box_within_bound, iou, nms, remove_small_box, \
    smooth_l1_loss, kmeans_nms
from .box_coder import BoxCoder
from .windows_generator import WindowsGenerator


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, 3, 1, 1)
        self.cls_logits = nn.Conv1d(in_channels, num_anchors, 1, 1)
        # self.cls_logits1 = nn.Conv1d(in_channels, in_channels // 2, 1, 1)
        # self.cls_logits2 = nn.Conv1d(in_channels // 2, in_channels // 4, 1, 1)
        # self.cls_logits3 = nn.Conv1d(in_channels // 4, num_anchors, 1, 1)
        self.bbox_pred = nn.Conv1d(in_channels, num_anchors * 2, 1, 1)
        # self.bbox_pred1 = nn.Conv1d(in_channels, in_channels // 2, 1, 1)
        # self.bbox_pred2 = nn.Conv1d(in_channels // 2, in_channels // 4, 1, 1)
        # self.bbox_pred3 = nn.Conv1d(in_channels // 4, num_anchors * 2, 1, 1)
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        t = F.relu(self.conv(x))
        # logits = self.cls_logits3(F.relu(self.cls_logits2(F.relu(self.cls_logits1(t)))))
        # bbox_reg = self.bbox_pred3(F.relu(self.bbox_pred2(F.relu(self.bbox_pred1(t)))))
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, L):
    layer = layer.view(N, -1, C, L)
    layer = layer.permute(0, 3, 1, 2)  # [N, L, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    N, AxC, L = box_cls.shape
    Ax2 = box_regression.shape[1]
    A = Ax2 // 2
    C = AxC // A

    box_cls = permute_and_flatten(box_cls, N, A, C, L)

    box_regression = permute_and_flatten(box_regression, N, A, 2, L)

    box_cls = box_cls.flatten(0, -2)  # start_dim, end_dim
    box_regression = box_regression.reshape(-1, 2)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, num_anchors, fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                 batchsize_per_image=256,
                 positive_fraction=0.5, score_thresh=0.9, nms_thresh=0.9):
        super().__init__()
        self.windows_generator = WindowsGenerator()
        self.head = RPNHead(in_channels, num_anchors)
        self.min_size = 1
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self._pre_nms_top_n = {'training': 1000, 'testing': 1000}
        self._post_nms_top_n = {'training': 1000, 'testing': 1000}
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.box_coder = BoxCoder()

        self.proposal_matcher = Matcher(
            self.fg_iou_thresh,
            self.bg_iou_thresh,
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = BalancedPositiveNegtiveSampler(
            batchsize_per_image, positive_fraction
        )

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def filter_proposals(self, proposal, objectness, seq_length, num_windows):
        batch_size = proposal.shape[0]

        objectness = objectness.detach()
        objectness = objectness.reshape(batch_size, -1)

        pre_nms_top_n = min(self.pre_nms_top_n(), num_windows)
        _, top_n_idx = objectness.topk(pre_nms_top_n, dim=1)
        batch_idx = torch.arange(batch_size)[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        proposal = proposal[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        count = 0
        for box, score in zip(proposal, objectness_prob):
            box = clip_box_within_bound(box, seq_length[count])
            count = count + 1
            keep = remove_small_box(box, self.min_size)
            box, score = box[keep], score[keep]

            keep = torch.where(torch.ge(score, self.score_thresh))[0]
            box, score = box[keep], score[keep]

            keep = nms(box, score, self.nms_thresh)

            keep = keep[:self.post_nms_top_n()]
            box, score = box[keep], score[keep]

            final_boxes.append(box)
            final_scores.append(score)
        return final_boxes, final_scores

    def assign_windows_to_targets(self, windows, targets):
        labels = []
        matched_gt_boxes = []
        for windows_per_image, targets_per_image in zip(windows, targets):
            if targets_per_image.numel() == 0:
                matched_gt_boxes_per_image = torch.zeros(windows_per_image.shape, dtype=torch.float32)
                labels_per_image = torch.zeros((windows_per_image.shape[0],), dtype=torch.float32)
            else:
                match_quality_matrix = iou(targets_per_image, windows_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                matched_gt_boxes_per_image = targets_per_image[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0
            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def loss_function(self, objectness, bbox_reg_pred, labels, regression_targets):
        sampled_pos_idxs, sampled_neg_idxs = self.fg_bg_sampler(labels)

        sampled_pos_idxs = torch.where(torch.cat(sampled_pos_idxs, dim=0))[0]
        self.sampled_pos_idx = sampled_pos_idxs
        sampled_neg_idxs = torch.where(torch.cat(sampled_neg_idxs, dim=0))[0]

        sampled_idxs = torch.cat([sampled_pos_idxs, sampled_neg_idxs], dim=0)
        objectness = objectness.flatten()

        bbox_reg_pred = bbox_reg_pred.view(-1, 2)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        regression_loss = smooth_l1_loss(
            bbox_reg_pred[sampled_pos_idxs],
            regression_targets[sampled_pos_idxs],
            beta=0.01,
            size_average=True,
        )

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_idxs], labels[sampled_idxs]
        )

        return objectness_loss, regression_loss

    def forward(self, features, targets, real_len):
        boxes, scores = None, None
        batch_size = features.shape[0]
        features = features.transpose(1, 2)
        objectness, bbox_pred_delta = self.head(features)
        windows = self.windows_generator(features, stride=1)
        s0, s1, s2 = objectness.shape
        num_windows = s1 * s2

        objectness, bbox_pred_delta = concat_box_prediction_layers(objectness, bbox_pred_delta)
        proposals = self.box_coder.decode(bbox_pred_delta.detach(), windows)
        proposals = proposals.view(batch_size, -1, 2)

        # if not self.training:
        boxes, scores = self.filter_proposals(proposals, objectness, real_len, num_windows)

        loss = {}
        # if self.training:
        #     assert targets is not None
        if targets is not None:
            labels, gt_matched_boxes = self.assign_windows_to_targets(windows, targets)
            # TODO：padding处该不该discard？
            for i in range(batch_size):
                labels[i][real_len[i] * s1:] = -1.0
            regression_targets = self.box_coder.encode(gt_matched_boxes, windows)
            loss_objectness, loss_regression = self.loss_function(
                objectness, bbox_pred_delta, labels, regression_targets
            )
            # loss = {
            #     "loss_objectness": loss_objectness,
            #     "loss_regression": loss_regression
            # }
            loss = loss_regression + loss_objectness
        return loss, boxes
