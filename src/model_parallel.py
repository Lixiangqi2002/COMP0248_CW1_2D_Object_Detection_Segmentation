from typing import OrderedDict
import torch
import torch.nn as nn
from DeepLabV3Plus_custom import DeepLabHeadV3Plus
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from FasterRCNN import FasterRCNNHead
import torch.nn as nn


class MultiHeadModel(nn.Module):
    def __init__(self, num_classes_detection, num_classes_segmentation):
        super(MultiHeadModel, self).__init__()

        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        self.backbone.out_channels = 256
    
        self.detection_head = FasterRCNNHead(self.backbone, num_classes=num_classes_detection)

        self.segmentation_head = DeepLabHeadV3Plus(
                in_channels=256, 
                low_level_channels=256, 
                num_classes=6  # 5 classes + 1 background
            )
 
        for param in self.backbone.parameters():
            param.requires_grad = False

    # segmentation // detection
    def forward(self, x, targets=None, training=True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Object detection branch
        if training:
            # det_output, det_loss, features = self.detection_head(x, targets, device=device)
            # rpn_output, frcnn_output, features, high_score_iou_mean, high_score_mean = self.detection_head(x, targets, device=device, training = training)
            # Faster R-CNN training
            detection_loss_dict, detection_loss, features = self.detection_head(x, targets, training=True)
  
            out = features['2']
            low_level_feature = features['0']
            segmentation_feature = OrderedDict([("out", out), ("low_level", low_level_feature)])
            
            segmentation_output, seg_loss = self.segmentation_head(segmentation_feature, targets)  # Input original image for segmentation
       
            return detection_loss_dict, detection_loss, segmentation_output, seg_loss 
        
        else:
            # rpn_output, frcnn_output , features = self.detection_head(x, device='cpu')
            detection_output, features = self.detection_head(x, training=False)
           
            out = features['2']
            low_level_feature = features['0']
            segmentation_feature = OrderedDict([("out", out), ("low_level", low_level_feature)])
            
            segmentation_output, seg_loss = self.segmentation_head(segmentation_feature, targets)  # Input original image for segmentation
        
            return detection_output, segmentation_output, seg_loss
        
    
if __name__ == "__main__":
    model = MultiHeadModel(num_classes_detection=6, num_classes_segmentation=6)
    print(model)
