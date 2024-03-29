import torch
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import RoIHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss, \
    keypointrcnn_inference, keypointrcnn_loss, maskrcnn_inference, maskrcnn_loss
from typing import Dict, List, Optional, Tuple
from torch import nn


class TFRCNNBoxHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x, descriptors):
        x = x.flatten(start_dim=1)

        batch_size = descriptors.size()[0]
        num_anchors_boxes = x.size()[0] // batch_size

        # Append the descriptor text appending to the fully connected layer.
        b = []
        for i in range(descriptors.size()[0]):
            t = torch.unsqueeze(descriptors[i], 0)
            t = t.expand(num_anchors_boxes, descriptors.size()[1])
            b.append(t)
        d = torch.cat(b, dim=0)

        x = torch.cat([x, d], dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class TFRCNNRoIHeads(RoIHeads):
    def __init__(
            self, box_roi_pool, box_head, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights, score_thresh,
            nms_thresh, detections_per_img,
            mask_roi_pool=None, mask_head=None, mask_predictor=None,
            keypoint_roi_pool=None, keypoint_head=None,
            keypoint_predictor=None):
        super().__init__(
            box_roi_pool, box_head, box_predictor,
            fg_iou_thresh, bg_iou_thresh, batch_size_per_image,
            positive_fraction, bbox_reg_weights, score_thresh,
            nms_thresh, detections_per_img,
            mask_roi_pool=mask_roi_pool, mask_head=mask_head,
            mask_predictor=mask_predictor,
            keypoint_roi_pool=keypoint_roi_pool, keypoint_head=keypoint_head,
            keypoint_predictor=keypoint_predictor)

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]],
        descriptors=None,
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            descriptors
            box_features
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, \
                    "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, \
                    "target labels must of int64 type"
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, \
                        "target keypoints must of float type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        # We have to go through the ROI Pool again because box_features
        # are different when training vs. evaluation.
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features, descriptors)
        class_logits, box_regression = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier,
                      "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels,
                    pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints,
                    pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores,
                                                 result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    sampled_neg_inds_subset = torch.where(labels <= 0)[0]
    labels_neg = labels[sampled_neg_inds_subset]
    N, num_classes = class_logits.shape
   
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    #max_vals, max_inds = torch.max(class_logits, dim=1)
    #max_inds.to('cuda')
    #labels.to('cuda')
    #inversed = torch.logical_not(labels, out=torch.empty(labels.size()[0]).to('cuda'))
    #classification_loss = F.triplet_margin_loss(max_inds, labels, inversed)

    #OLD ONE
    # box_loss = F.smooth_l1_loss(
    #     box_regression[sampled_pos_inds_subset, labels_pos],
    #     regression_targets[sampled_pos_inds_subset],
    #     beta=1 / 9,
    #     reduction="sum",
    # )
    anchor = box_regression[sampled_pos_inds_subset, labels_pos]
    pos = regression_targets[sampled_pos_inds_subset]
    neg = regression_targets[sampled_neg_inds_subset]
    size = anchor.size()[0]
    neg = neg[:size]

    box_loss = F.triplet_margin_loss(
        anchor,
        pos,
        neg
    )

    #classification_loss = classification_loss / labels.numel()
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss
