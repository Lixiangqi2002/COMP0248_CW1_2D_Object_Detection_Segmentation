import torch
from torchvision.ops import box_iou
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.ops import nms
from PIL import Image
import glob
import os
import xml.etree.ElementTree as ET


def compute_ap(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {cls: 0.0 for cls in gt_labels.unique().tolist()}  # If no targets, set AP to 0

    ap_per_class = {}

    for cls in gt_labels.unique().tolist():
        gt_cls_boxes = gt_boxes[gt_labels == cls]
        pred_cls_boxes = pred_boxes[pred_labels == cls]
        pred_cls_scores = pred_scores[pred_labels == cls]

        if len(pred_cls_boxes) == 0 or len(gt_cls_boxes) == 0:
            ap_per_class[cls] = 0.0
            continue

        # Calculate IoU
        iou_matrix = box_iou(pred_cls_boxes, gt_cls_boxes)  # (M, N)

        # Calculate TP, FP
        tp = torch.zeros(len(pred_cls_boxes))
        fp = torch.zeros(len(pred_cls_boxes))
        matched_gt = torch.zeros(len(gt_cls_boxes), dtype=torch.bool)

        sorted_indices = torch.argsort(pred_cls_scores, descending=True)

        for i, pred_idx in enumerate(sorted_indices):
            iou_values = iou_matrix[pred_idx]
            max_iou, max_gt_idx = iou_values.max(0)

            if max_iou >= iou_threshold and not matched_gt[max_gt_idx]:
                tp[i] = 1
                matched_gt[max_gt_idx] = True
            else:
                fp[i] = 1

        # Avoid all-zero data
        if tp.sum() == 0 and fp.sum() == 0:
            ap_per_class[cls] = 0.0
            continue

        # Calculate Precision-Recall
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / len(gt_cls_boxes)

        # Avoid empty `y_true`
        if len(recall) == 0 or len(precision) == 0:
            ap_per_class[cls] = 0.0
        else:
            ap_per_class[cls] = average_precision_score(recall.numpy(), precision.numpy())

    return ap_per_class


def compute_iou(mask_pred, mask_gt):
    """
    Calculate IoU
    """
    intersection = torch.logical_and(mask_pred, mask_gt).sum().item()
    union = torch.logical_or(mask_pred, mask_gt).sum().item()
    return intersection / union if union > 0 else 0

def compute_class_iou(pred, gt, num_classes):
    """Calculate IoU for each class and return per-class IoU and overall IoU"""
    iou_per_class = {}
    iou_weights = {}
    overall_iou = []

    for cls in range(1, num_classes):  # Ignore background class (0)
        pred_cls = (pred == cls).numpy()
        gt_cls = (gt == cls).numpy()

        if gt_cls.sum() == 0:  # No such class in GT
            continue
        else:
            intersection = np.logical_and(pred_cls, gt_cls).sum()
            union = np.logical_or(pred_cls, gt_cls).sum()
            iou = intersection / union if union > 0 else 0
            iou_per_class[cls] = iou
            iou_weights[cls] = 1.0  # Normal class weight

    mean_overall_iou = sum(iou_per_class[cls] * iou_weights[cls] for cls in iou_per_class) / sum(iou_weights.values()) if iou_per_class else 0
    return iou_per_class, mean_overall_iou


def compute_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes, iou_thresholds=[0.5, 0.75]):
    """
    Calculate mAP (mean Average Precision) for object detection using 1:1 matching strategy.
    - Each GT matches only one prediction box with the highest IoU.
    - Unmatched prediction boxes are counted as FP, unmatched GTs are counted as FN.

    :param pred_boxes: Predicted box coordinates (List of N tensors, each shape [num_preds, 4])
    :param pred_scores: Predicted box confidence scores (List of N tensors, each shape [num_preds])
    :param pred_labels: Predicted box classes (List of N tensors, each shape [num_preds])
    :param gt_boxes: Ground truth box coordinates (List of N tensors, each shape [num_gts, 4])
    :param gt_labels: Ground truth box classes (List of N tensors, each shape [num_gts])
    :param num_classes: Total number of classes (including background)
    :param iou_thresholds: List of IoU thresholds (usually set to [0.5, 0.75])

    :return: dict - mAP value for each IoU threshold
    """
    ap_per_iou = {iou_thresh: [] for iou_thresh in iou_thresholds}

    for cls in range(1, num_classes):  # Skip background class
        cls_precisions = []

        for iou_thresh in iou_thresholds:
            tp, fp, fn = 0, 0, 0

            for img_idx in range(len(pred_boxes)):  # Iterate over each image
                pred_cls_mask = (pred_labels[img_idx] == cls)
                gt_cls_mask = (gt_labels[img_idx] == cls)

                pred_cls_boxes = pred_boxes[img_idx][pred_cls_mask]
                gt_cls_boxes = gt_boxes[img_idx][gt_cls_mask]

                if len(gt_cls_boxes) == 0 and len(pred_cls_boxes) == 0:
                    continue  # This class does not exist in this image, skip

                if len(pred_cls_boxes) > 0 and len(gt_cls_boxes) == 0:
                    fp += len(pred_cls_boxes)  # Predictions exist, GT does not, all are FP

                elif len(gt_cls_boxes) > 0 and len(pred_cls_boxes) == 0:
                    fn += len(gt_cls_boxes)  # GT exists, predictions do not, all are FN

                else:
                    # Calculate IoU matrix
                    iou_matrix = box_iou(pred_cls_boxes, gt_cls_boxes)
                    ious = iou_matrix.numpy()

                    matched_gt = set()  # Record matched GTs
                    matched_pred = set()  # Record matched prediction boxes

                    for gt_idx in range(len(gt_cls_boxes)):  # Iterate over GT boxes
                        best_pred_idx = np.argmax(ious[:, gt_idx])  # Find prediction box with highest IoU
                        best_iou = ious[best_pred_idx, gt_idx]

                        if best_iou >= iou_thresh and best_pred_idx not in matched_pred:
                            tp += 1
                            matched_gt.add(gt_idx)
                            matched_pred.add(best_pred_idx)

                    # Calculate FP (unmatched prediction boxes)
                    fp += len(pred_cls_boxes) - len(matched_pred)

                    # Calculate FN (unmatched GT boxes)
                    fn += len(gt_cls_boxes) - len(matched_gt)

            # Calculate Precision and Recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap = precision * recall  # Calculate AP (simplified version, can be extended to PR curve calculation)

            cls_precisions.append(ap)

        for iou_thresh, ap in zip(iou_thresholds, cls_precisions):
            ap_per_iou[iou_thresh].append(ap)

    # Calculate mAP
    map_results = {iou_thresh: np.mean(ap_per_iou[iou_thresh]) if len(ap_per_iou[iou_thresh]) > 0 else 0 for iou_thresh in iou_thresholds}
    return map_results


def histogram_equalization_color(image_path, save_path):
    """
    Perform histogram equalization on a color image (only equalize the Y channel)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Unable to load image: {image_path}")
        return

    # Convert to YCrCb color space
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Only equalize the Y channel
    image_ycrcb[:, :, 0] = cv2.equalizeHist(image_ycrcb[:, :, 0])

    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Save the equalized color image
    output_path = os.path.join(save_path,f"{os.path.basename(image_path)[:-4]}_L.png")
    cv2.imwrite(output_path, equalized_image)
    print(f"Saved: {output_path}")

    # Calculate original & equalized histograms
    colors = ('b', 'g', 'r')  # BGR channels
    plt.figure(figsize=(10, 5))
    
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_eq = cv2.calcHist([equalized_image], [i], None, [256], [0, 256])

        plt.subplot(2, 3, i + 1)
        plt.plot(hist_orig, color=color)
        plt.title(f"Original {color.upper()} Histogram")
        
        plt.subplot(2, 3, i + 4)
        plt.plot(hist_eq, color=color)
        plt.title(f"Equalized {color.upper()} Histogram")
    
    # hist_output_path = os.path.join(save_path, f"{os.path.basename(image_path)[:-4]}_L.png")
    # plt.savefig(hist_output_path)
    # plt.close()


def process_folder_color(folder_path, save_path):
    """
    Traverse the folder, process all color images and perform histogram equalization
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("Not found any image files in the folder.")
        return

    for img_file in image_files:
        image_path = os.path.join(folder_path, img_file)
        histogram_equalization_color(image_path, save_path)

# Update to include only the classes you need and the background class (Void)
RGBLabel2LabelName = {
    (0, 128, 192): "Bicyclist",       # Bicyclist
    (64, 0, 128): "Car",              # Car
    (192, 0, 192): "MotorcycleScooter",# MotorcycleScooter
    (64, 64, 0): "Pedestrian",        # Pedestrian
    (192, 128, 192): "Truck_Bus",     # Truck_Bus
}

RGBLabel = [  [0, 128, 192],    # Bicyclist
    [64, 0, 128],     # Car
    [192, 0, 192],    # MotorcycleScooter
    [64, 64, 0],      # Pedestrian
    [192, 128, 192],  # Truck_Bus
]

# Classification label ID, only includes 5 classes and Void class
Class2LabelId = {
    "Bicyclist": 0,
    "Car": 1,
    "MotorcycleScooter": 2,
    "Pedestrian": 3,
    "Truck_Bus": 4,
    "Void": 255  # Void class is 255
}

# Corresponding RGB color palette
palette = [
    0, 128, 192,    # Bicyclist
    64, 0, 128,     # Car
    192, 0, 192,    # MotorcycleScooter
    64, 64, 0,      # Pedestrian
    192, 128, 192,  # Truck_Bus
    0, 0, 0         # Void (background class) needs to be filled
]

# Pad the palette
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def convert32to5(label_dir, save_dir):
    
    img_paths = glob.glob(os.path.join(label_dir, '*.png'))
    img_paths.sort()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, img_path in enumerate(img_paths):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        np_img = np.array(img)
        np_img_ret = 255 * np.ones(np_img.shape[:2], dtype=np.uint8)  # Default Void class
        w, h = np_img.shape[:2]
        for x in range(w):
            for y in range(h):
                rgb = tuple(np_img[x, y, :])
                if rgb in RGBLabel2LabelName:
                    label = RGBLabel2LabelName[rgb]
                else:
                    label = 'Void'  
                np_img_ret[x, y] = Class2LabelId[label]
        img = Image.fromarray(np_img_ret)

        img.save(os.path.join(save_dir, img_name))
        print(idx, img_name)
    print('Done')


def convertGraytoRGB(gray_dir, save_dir):

    img_paths = glob.glob(os.path.join(gray_dir, '*.png'))
    img_paths.sort()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, img_path in enumerate(img_paths):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('P')
        img.putpalette(palette)
        img.save(os.path.join(save_dir, img_name))
        print(idx, img_name)
    print('Done')


def convertRGBtoAnno(rgb_dir, target_colors, class_names, iou_threshold=0.8, save_dir=False, save_xml_dir='CamVidColor5/annotations/train_annotations'):

    img_paths = glob.glob(os.path.join(rgb_dir, '*.png'))
    img_paths.sort()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, img_path in enumerate(img_paths):
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path).convert('RGB')
        np_img = np.array(img)
        # Step 1: Extract the target color region and set others to white
        np_img = np.array(img)
        all_bboxes = []
        class_names_all = []
        for target_color, class_name in zip(target_colors, class_names):
            # Create a mask for the target color
            target_color = np.array(target_color)
            mask = np.all(np_img == target_color, axis=-1)

            # Create output image with white background
            output_image = np.ones_like(np_img) * 255  # White background
            output_image[mask] = np_img[mask]  # Retain target color
     
            # Step 2: Extract bounding boxes using contours
            mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bboxes = []
            for contour in contours:
                # Get bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)
                # no small area bbox
                if w * h < 100:
                    continue
                bboxes.append([x, y, x + w, y + h])

            if len(bboxes) > 0:
                bboxes = torch.tensor(bboxes).float()
                scores = torch.ones(len(bboxes))  # Assuming equal confidence for all boxes
                keep = nms(bboxes, scores, iou_threshold)
            
                # Return the bounding boxes after NMS
                nms_bboxes = bboxes[keep]
                all_bboxes.extend(nms_bboxes)
                class_names_all.extend(class_name for _ in range(len(nms_bboxes)))
                # # visualize
                img_bbx = np_img.copy() 
                # for bbox in nms_bboxes:
                #     x1, y1, x2, y2 = bbox.int().tolist()
                #     img_bbx = cv2.rectangle(img_bbx, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Optionally save the processed image
            else:
                # print(f"No bounding boxes found for {class_name} in image {img_name}")
                nms_bboxes = torch.tensor([])  # Return an empty tensor if no bounding boxes are found
                img_bbx = np_img.copy()

        # Plot all bounding boxes on the img_bbx
        for bbox, class_name in zip(all_bboxes, class_names_all):
            
            x1, y1, x2, y2 = bbox.int().tolist()
            img_bbx = cv2.rectangle(img_bbx, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img_bbx = cv2.putText(img_bbx, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # img.save(os.path.join(save_dir, img_name))
            img_bbx = Image.fromarray(img_bbx)
            img_bbx.save(os.path.join(save_dir, img_name.replace('.png', '_bbx.png')))

        #  Save XML annotation for the image
        xml_path = os.path.join(save_xml_dir, img_name.replace('.png', '.xml'))
        save_to_xml(img_path, all_bboxes, class_names_all, xml_path, image_size=img.size)


    return nms_bboxes


def save_to_xml(image_path, bboxes, class_names, xml_path, image_size):

    if not os.path.exists(os.path.dirname(xml_path)):
        os.makedirs(os.path.dirname(xml_path))
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = os.path.dirname(image_path)
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    
    # Image size
    size_elem = ET.SubElement(root, "size")
    ET.SubElement(size_elem, "width").text = str(image_size[0])
    ET.SubElement(size_elem, "height").text = str(image_size[1])
    ET.SubElement(size_elem, "depth").text = "3"

    # Add each bounding box
    for i, bbox in enumerate(bboxes):
        obj = ET.SubElement(root, "object")
        name = class_names[i] if class_names else "target"
        ET.SubElement(obj, "name").text = name  # Class name (e.g., "Car", "Pedestrian")
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(bbox[0].item()))
        ET.SubElement(bndbox, "ymin").text = str(int(bbox[1].item()))
        ET.SubElement(bndbox, "xmax").text = str(int(bbox[2].item()))
        ET.SubElement(bndbox, "ymax").text = str(int(bbox[3].item()))

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(xml_path)