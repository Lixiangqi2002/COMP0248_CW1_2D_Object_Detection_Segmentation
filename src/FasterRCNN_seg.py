import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
import torch.nn as nn

class FasterRCNNHead(nn.Module):
    def __init__(self, backbone, num_classes=6):
        super(FasterRCNNHead, self).__init__()
        backbone.out_channels = 256  # FPN output channels

        # **RPN generator**
        anchor_generator = AnchorGenerator(
            sizes=((4,), (8,), (16,), (32,), (64,), (128,), (256,)),
            aspect_ratios=((0.01, 0.25, 0.5, 1.0, 2.0),) * 5
        )

        # **ROI Align**
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
        )

        # **Construct Faster R-CNN**
        self.faster_rcnn = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, targets=None, fg_mask=None, training=True, epoch=0, batch_idx=0):
        """
        Object detection forward pass
        """
        if training:
            enhanced_images = images * (fg_mask * 0.5 + 0.5)  # Make background still visible
            # # visualize the enhanced images & original images
            # for i in range(len(images)):
            #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            #     ax[0].imshow(images[i].permute(1, 2, 0).cpu().numpy())
            #     ax[0].set_title("Original Image")
            #     ax[1].imshow(enhanced_images[i].permute(1, 2, 0).cpu().numpy())
            #     ax[1].set_title("Enhanced Image")
            #     plt.show()
            loss_dict = self.faster_rcnn(enhanced_images, targets)
            gt_masks = []
            for t in targets:
                masks = t['masks'].long()

                mask_cur = torch.argmax(masks, dim=0)
                gt_masks.append(mask_cur)
            gt_masks = torch.stack(gt_masks, dim=0).unsqueeze(1)
            binary_gt = (gt_masks > 0).float()

            intersection = (fg_mask * binary_gt).sum()
            union = fg_mask.sum() + binary_gt.sum() - intersection
            similarity = intersection / union
            # print("Similarity: ", similarity)
            
            # if predicted segmentation is not similar to the gt segmentation, increase the weight of the fg_mask
            fg_weight = torch.clamp(torch.tensor((1-similarity)*10, device=images.device), min=0.0, max=10.0)  
            # print("Foreground Weight: ", fg_weight)
            # increase the loss of the rpn box regression and objectness based on the similarity of the predicted
            # segmentation and the gt segmentation
            loss_dict["loss_rpn_box_reg"] *= fg_weight
            loss_dict["loss_objectness"] *= fg_weight
            class_weights = torch.tensor([0.05, 2.58, 2.14, 9.80, 343.25, 23.27], dtype=torch.float32, device=images.device)
            loss_dict["loss_classifier"] *= class_weights[targets[0]["labels"]].mean()
            loss_dict["loss_box_reg"] *= class_weights[targets[0]["labels"]].mean()

            return loss_dict, sum(loss for loss in loss_dict.values())

        else:
            enhanced_images = images * (fg_mask * 0.5 + 0.5)  # Make background still visible
            return self.faster_rcnn(enhanced_images)
