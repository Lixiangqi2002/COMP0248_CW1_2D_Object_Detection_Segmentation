import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
import cv2
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_image_with_bboxes(image, bboxes, labels):
    """
    Visualize image with bounding boxes
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    for bbox, label in zip(bboxes, labels):
        if label == 0:
            continue  # Skip padding
        x1, y1, x2, y2 = bbox
        if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
            continue
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, str(label.item()), color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.show()

def visualize_anchors(image, anchors, bboxes):
    """
    Visualize Anchor boxes and ground truth bboxes
    """
    image = np.array(image.permute(1, 2, 0).cpu())  # Restore image
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    # Draw GT boxes
    for box in bboxes.cpu().numpy():
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)

    # Draw Anchors
    for i in range(0, len(anchors), len(anchors) // 100):  # Only draw 100 anchors
        x1, y1, x2, y2 = anchors[i].cpu().numpy()
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=1, linestyle="dashed")
        ax.add_patch(rect)

    plt.show()

def visualize_boxes(image, boxes, title="Visualization", color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the given image.
    :param image: Input PyTorch tensor image (C, H, W).
    :param boxes: Tensor of shape (N, 4) representing bounding boxes [x1, y1, x2, y2].
    :param title: Title of the visualization.
    :param color: Color of the boxes (BGR).
    :param thickness: Thickness of the box lines.
    """
    # Convert image format
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)  # De-normalize
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box in boxes.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Display image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_boxes(img, boxes, labels=None, scores=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the image
    :param img: (Tensor) Input image (C, H, W)
    :param boxes: (Tensor) Predicted boxes, shape (N, 4)
    :param labels: (Tensor) Classes, shape (N,)
    :param scores: (Tensor) Confidence scores, shape (N,)
    :param color: Color of the boxes
    :param thickness: Thickness of the box lines
    :return: Image with drawn boxes
    """
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    num_boxes = boxes.shape[0]
    for i in range(num_boxes):
        x1, y1, x2, y2 = map(int, boxes[i])
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if scores is not None:
            score = scores[i]
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if labels is not None:
            label = labels[i]
            cv2.putText(img, f"{label}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    return img


def visualize_proposals(image, proposals, title):
    """
    Visualize image with bounding boxes
    """
    img = image[0].cpu().numpy().transpose(1, 2, 0).copy()
    for box in proposals.cpu().detach().numpy():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue box
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.title(title)
    plt.show()

def visualize_batch(images, masks, masked_images, segmentation_output, batch_idx=0, name=""):
    """
    Visualize results of 4 key steps:
    1. Original input image
    2. Predicted foreground Mask
    3. Masked processed input
    4. Semantic segmentation output (each class)

    Parameters:
    - images: Original input (N, C, H, W)
    - masks: Foreground Mask (N, 1, H, W)
    - masked_images: Processed Masked input (N, C, H, W)
    - segmentation_output: Semantic segmentation results (N, 6, H, W)
    - batch_idx: Select which image in the batch to visualize
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))  # Changed to 2 rows 3 columns
    axes = axes.flatten()  # Flatten to 1D to avoid index out of range

    # Select one image from the batch
    image = images[batch_idx].cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
    mask = masks[batch_idx, 0].cpu().numpy()  # (H, W)
    masked_image = masked_images[batch_idx].cpu().permute(1, 2, 0).numpy()
    seg_output = segmentation_output[batch_idx].detach().cpu().numpy()  # (6, H, W)

    # Display original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    # Display foreground Mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Foreground Mask")

    # Display Masked processed image
    axes[2].imshow(masked_image)
    axes[2].set_title("Masked Image")

    # plt.savefig(f"results/seg_results_for_detection/{name}")
    plt.close(fig)
