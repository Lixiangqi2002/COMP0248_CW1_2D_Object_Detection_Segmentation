from typing import OrderedDict
import cv2
import numpy as np
import torch
import torch.nn as nn
# from FasterRCNN_custom import FasterRCNN
from DeepLabV3Plus_custom import DeepLabHeadV3Plus
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from FasterRCNN_seg import FasterRCNNHead
import torch.nn as nn
from visualize import visualize_batch


class MultiHeadModel(nn.Module):
    def __init__(self, num_classes_detection, num_classes_segmentation):
        super(MultiHeadModel, self).__init__()

        # Backbone
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        self.backbone.out_channels = 256
       
        self.detection_head = FasterRCNNHead(self.backbone, num_classes=num_classes_detection)

        self.segmentation_head = DeepLabHeadV3Plus(
                in_channels=256, 
                low_level_channels=256, 
                num_classes=6  # 5 classes + 1 background
            )
   
        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False


    def dilate_mask(self, segmentation_output, kernel_size=5, iterations=1):
        """
        Perform dilation on the segmentation mask to reduce target loss
        
        :param segmentation_output: (N, 6, H, W) semantic segmentation result
        :param kernel_size: size of the dilation kernel
        :param iterations: number of dilation iterations
        :return: dilated mask (N, 1, H, W)
        """
        batch_size, num_classes, H, W = segmentation_output.shape

        # Calculate foreground mask
        foreground_mask = (segmentation_output[:, 1:, :, :].sum(dim=1) > 0).float().cpu().numpy()  # (N, H, W)

        # Morphological dilation
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 5x5 dilation kernel
        dilated_masks = []
        for i in range(batch_size):
            dilated_mask = cv2.dilate(foreground_mask[i], kernel, iterations=iterations)  # Dilation
            dilated_masks.append(dilated_mask)

        # Convert back to tensor
        dilated_masks = torch.tensor(dilated_masks, dtype=torch.float32, device=segmentation_output.device).unsqueeze(1)  # (N, 1, H, W)

        return dilated_masks

    # segmentation -> detection
    def forward(self, x, targets=None, img_name=None, epoch=-1, training=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Step 1: Compute FPN features
        features = self.backbone(x)
        out = features['2']
        low_level_feature = features['0']
        segmentation_feature = OrderedDict([("out", out), ("low_level", low_level_feature)])
        # Step 2: Compute segmentation result
        segmentation_output, seg_loss = self.segmentation_head(segmentation_feature, targets)  # (N, 6, H, W)
        
        # print(segmentation_output)
        # Step 3: Compute foreground Mask
        import torch.nn.functional as F
        foreground_mask = self.dilate_mask(segmentation_output, kernel_size=1, iterations=2)
        # foreground_mask = (segmentation_output[:, 1:6, :, :].sum(dim=1) > 0).float().unsqueeze(1)  # (N, 1, H, W)
        foreground_mask = F.interpolate(foreground_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
        foreground_mask = (foreground_mask > 0.5).float()

        if training:
       
            detection_loss_dict, detection_loss = self.detection_head(x,  targets, foreground_mask,training=True)
            return detection_loss_dict, detection_loss, segmentation_output, seg_loss
        else:

            detection_output = self.detection_head(x, targets, foreground_mask, training=False)
            return detection_output, segmentation_output, seg_loss

    
if __name__ == "__main__":
    model = MultiHeadModel(num_classes_detection=6, num_classes_segmentation=6)
    print(model)
