import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class FasterRCNNHead(nn.Module):
    def __init__(self, backbone, num_classes=6):
        """
        Object Detection Head (Faster R-CNN)
        
        :param backbone: Pre-trained backbone network (ResNet50 + FPN)
        :param num_classes: Number of target classes (5 classes + 1 background)
        """
        super(FasterRCNNHead, self).__init__()

        # Ensure the backbone outputs the correct number of channels
        backbone.out_channels = 256  # Number of channels output by FPN

        # Define the Anchor Generator for RPN
        # anchor_generator = AnchorGenerator(
        #     sizes=((2,), (4,), (16,), (32,), (64,), (128,), (256,)),  # Anchor sizes on each feature map
        #     aspect_ratios=((0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0),) * 5  # Anchor aspect ratios on each feature map
        # )
        anchor_generator = AnchorGenerator(
            sizes=((4,), (8,), (16,), (32,), (64,), (128,), (256,)),  # Anchor sizes on each feature map
            aspect_ratios=((0.01, 0.25, 0.5, 1.0, 2.0),) * 5  # Anchor aspect ratios on each feature map
        )
        # Define ROI Align (Multi-scale feature pooling)
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],  # Select which layers of FPN output to use
            output_size=7,  # 7x7 pooling
            sampling_ratio=2
        )

        # Construct Faster R-CNN
        self.faster_rcnn = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, images, mask_fg=None, targets=None, training=True):
        """
        Forward pass

        :param images: Input images
        :param targets: Target annotations during training (dictionary: {boxes, labels})
        :param training: Whether in training mode
        :return: If training, return loss; otherwise return detection results
        """

        feat = self.faster_rcnn.backbone(images)  # Feature maps output by the backbone
        
        # for k,v in feat.items():
        #     print(k, v.shape)
        if training:
            loss_dict = self.faster_rcnn(images, targets)
            # {3: 9.807142857142857, 1: 2.585687382297552, 2: 2.1486697965571206, 5: 23.271186440677965, 4: 343.25}
            class_weights = torch.tensor([0.05, 2.58, 2.14, 9.80, 343.25, 23.27], dtype=torch.float32, device=images.device)
            loss_dict["loss_classifier"] *= class_weights[targets[0]["labels"]].mean()
            loss_dict["loss_box_reg"] *= class_weights[targets[0]["labels"]].mean()
           
            total_loss = sum(loss for loss in loss_dict.values())  # Calculate total loss
            return loss_dict, total_loss, feat
        else:
            print(images.shape)
            images = [images]
            return self.faster_rcnn(images), feat  # Inference mode returns detection results
