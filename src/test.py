import time
import torch
import torchvision
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from coco_eval import CocoEvaluator
from dataloader import CamVidDataset

import torch.nn.functional as F
from torchvision.ops import box_iou
from collections import defaultdict
from train import convert_to_coco, format_outputs
from utils import compute_class_iou, compute_map
import utils_coco


if __name__ == "__main__":
    mode = "flow"  # Choose between "flow" or "parallel"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_classes = {
        "Background": 0, 
        "Car": 1,
        "Pedestrian": 2,
        "Bicyclist": 3,
        "MotorcycleScooter": 4,
        "Truck_Bus": 5
    }
    idx_to_class = {v: k for k, v in selected_classes.items()}
    color = {
        "Car": (255, 255, 255),
        "Pedestrian": (0, 255, 0),
        "Bicyclist": (0, 0, 255),
        "MotorcycleScooter": (255, 255, 0),
        "Truck_Bus": (255, 0, 255)
    }

    # Global storage for evaluation results
    results_storage = {
        "mAP_50_95": [],
        "mAP_50": [],
        "mAP_75": [],
        "mAP_small": [],
        "mAP_medium": [],
        "mAP_large": [],
        "mAR_1": [],
        "mAR_10": [],
        "mAR_100": [],
        "mAR_small": [],
        "mAR_medium": [],
        "mAR_large": [],
        "segmentation_iou": defaultdict(list),
        "overall_iou": []
    }

    transform_test = A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1))

    test_dataset = CamVidDataset(
        img_dir="CamVidColor5/test_he_color",
        mask_dir="CamVidColor5/test_labels",
        anno_dir="CamVidColor5/test_annotations",
        selected_classes=selected_classes,
        transforms=transform_test,
        plot=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    if mode == "flow":
        from model_flow import MultiHeadModel
        model = MultiHeadModel(num_classes_detection=6, num_classes_segmentation=6)
        model.load_state_dict(torch.load("weights/multihead_best_model_seg+det_flow.pth", map_location=device))
    elif mode == "parallel":
        from model_parallel import MultiHeadModel
        model = MultiHeadModel(num_classes_detection=6, num_classes_segmentation=6)
        model.load_state_dict(torch.load("weights/multihead_best_model_parallel.pth", map_location=device))

    model.to(device)
    model.eval()

    n_threads = torch.get_num_threads()
    cpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    metric_logger = utils_coco.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = convert_to_coco(test_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    iou_scores = defaultdict(list)

    total_samples = 0
    for images, targets, idx in metric_logger.log_every(test_loader, 100, header):   
        if mode == "flow":     
            targets = [targets] # for testing, generating formated targets
    
            # print("Target: ", targets)
            images = images.to(device)
            print("Image Shape: ", images.shape)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            epoch = 0
            with torch.no_grad():
                detection_output, segmentation_output, seg_loss = model(images, training=False)
            import copy
            detection_output_clone = copy.deepcopy(detection_output)
            segmentation_output_clone = copy.deepcopy(segmentation_output)
            segmentation_output = segmentation_output.squeeze(0)  # (1, 6, 64, 64) -> (6, 64, 64)
            pred_masks = torch.argmax(segmentation_output, dim=0).cpu()
            
            for batch_idx, target in enumerate(targets):
                print("Inside Target: ", target)

                gt_masks = target["masks"].cpu()  # (num_classes, H, W)
                gt_masks = torch.argmax(gt_masks, dim=0)

                import torch.nn.functional as F
                gt_masks = F.interpolate(gt_masks.unsqueeze(0).unsqueeze(0).float(),
                                        size=pred_masks.shape,
                                        mode="nearest").squeeze(0).squeeze(0).long()

                iou_per_class, mean_iou = compute_class_iou(pred_masks, gt_masks, 6)
                
                for cls, iou in iou_per_class.items():
                    iou_scores[cls].append(iou)

                iou_scores["overall"].append(mean_iou)
                total_samples += 1

            outputs = []
            image_ids = [int(target["image_id"]) for target in targets]
            outputs = format_outputs(detection_output, segmentation_output, image_ids)
            outputs = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            coco_evaluator.synchronize_between_processes()

            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            torch.set_num_threads(n_threads)

            print("\nSegmentation IoU Results:")
            mean_iou_per_class = {}
            for cls in range(1, 6):  # Only count foreground classes
                if cls in iou_scores:
                    mean_iou = np.mean(iou_scores[cls]) if len(iou_scores[cls]) > 0 else 0
                    mean_iou_per_class[f"IoU Class {cls}"] = mean_iou
                    print(f"  Class {cls}: IoU = {mean_iou:.4f}")

            overall_iou = np.mean(iou_scores["overall"]) if len(iou_scores["overall"]) > 0 else 0
            print(f"\nOverall IoU = {overall_iou:.4f}")

            # Store COCO evaluation metrics
            coco_eval_stats = coco_evaluator.coco_eval['bbox'].stats
            results_storage["mAP_50_95"].append(coco_eval_stats[0])
            results_storage["mAP_50"].append(coco_eval_stats[1])
            results_storage["mAP_75"].append(coco_eval_stats[2])
            results_storage["mAP_small"].append(coco_eval_stats[3])
            results_storage["mAP_medium"].append(coco_eval_stats[4])
            results_storage["mAP_large"].append(coco_eval_stats[5])

            results_storage["mAR_1"].append(coco_eval_stats[6])
            results_storage["mAR_10"].append(coco_eval_stats[7])
            results_storage["mAR_100"].append(coco_eval_stats[8])
            results_storage["mAR_small"].append(coco_eval_stats[9])
            results_storage["mAR_medium"].append(coco_eval_stats[10])
            results_storage["mAR_large"].append(coco_eval_stats[11])

            # Store segmentation IoU per class
            for cls, iou in mean_iou_per_class.items():
                results_storage["segmentation_iou"][cls].append(iou)

            # Store overall segmentation IoU
            results_storage["overall_iou"].append(overall_iou)

            # test(model, test_loader, device, coco_evaluator, overall_iou, mean_iou_per_class)
            # img = img.unsqueeze(0).to(device)

            # with torch.no_grad():
            #     detection_output, segmentation_output, _ = model(img, training=False)

            pred_boxes = detection_output_clone[0]["boxes"].cpu().numpy()
            pred_scores = detection_output_clone[0]["scores"].cpu().numpy()
            pred_labels = detection_output_clone[0]["labels"].cpu().numpy()
            pred_masks = torch.argmax(segmentation_output_clone[0], dim=0).cpu().numpy()

            img_np = images.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Ground truth
            true_boxes = target["boxes"].cpu().numpy()
            true_labels = target["labels"].cpu().numpy()
            mask_gt = target["masks"].squeeze(0).cpu().numpy()
            mask_gt = np.argmax(mask_gt, axis=0)

            # Create color segmentation image
            pred_masks_color = np.zeros((pred_masks.shape[0], pred_masks.shape[1], 3), dtype=np.uint8)
            mask_gt_color = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)

            for class_name, class_idx in selected_classes.items():
                if class_name in color:
                    pred_masks_color[pred_masks == class_idx] = color[class_name]
                    mask_gt_color[mask_gt == class_idx] = color[class_name]

            # Overlay segmentation results on the original image
            pred_masks_color_resized = cv2.resize(pred_masks_color, (img_np.shape[1], img_np.shape[0]))
            mask_gt_color_resized = cv2.resize(mask_gt_color, (img_np.shape[1], img_np.shape[0]))
            img_with_pred_mask = cv2.addWeighted(img_np, 1, pred_masks_color_resized, 0.5, 0)
            img_with_gt_mask = cv2.addWeighted(img_np, 1, mask_gt_color_resized, 0.5, 0)

            # Draw ground truth bounding boxes
            for b in true_boxes:
                cv2.rectangle(img_np, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)  # Blue for GT

            # Draw predicted bounding boxes with confidence > 0.5
            for j in range(len(pred_boxes)):
                box = pred_boxes[j]
                label = idx_to_class[pred_labels[j]]
                score = pred_scores[j]

                if score > 0.5:
                    color_box = color[label] if label in color else (0, 255, 0)  # Default to green if color is missing
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_box, 2)
                    cv2.putText(img_np, f"{label}: {score:.2f}", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

            # Plot all visualizations
            plt.figure(figsize=(12, 6))

            # Image with object detection results
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title(f"Detection Results\nmAP@0.5: {coco_evaluator.coco_eval['bbox'].stats[0]:.3f} | mAP@0.75: {coco_evaluator.coco_eval['bbox'].stats[1]:.3f}")

            # Image with predicted segmentation mask
            plt.subplot(1, 3, 2)
            plt.imshow(img_with_pred_mask, cmap="jet", alpha=0.6)
            plt.title(f"Segmentation Results - IoU: {overall_iou:.3f}")

            # Image with ground truth segmentation mask
            plt.subplot(1, 3, 3)
            plt.imshow(img_with_gt_mask, cmap="jet", alpha=0.6)
            plt.title("Ground Truth")

            plt.tight_layout()
            plt.savefig(f"results/{idx[:-4]}_result.png")
            

            for cls, iou in mean_iou_per_class.items():
                print(f"  {cls}: IoU = {iou:.4f}")

            # n_threads = torch.get_num_threads() # Save number of threads    
            metric_logger = utils_coco.MetricLogger(delimiter="  ")
            header = "Test:"

            # coco = convert_to_coco(test_loader.dataset)
            # iou_types = ["bbox"]
            # coco_evaluator = CocoEvaluator(coco, iou_types)
            coco_evaluator.clear()
            iou_scores = defaultdict(list)

            total_samples = 0
        else:
            # print("Target: ", targets)
            images = images.to(device)
            print("Image Shape: ", images.shape)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            epoch = 0
            with torch.no_grad():
                detection_output, segmentation_output, seg_loss = model(images, training=False)
            # print("Detection Output: ", detection_output)
            # print("Segmentation Output: ", segmentation_output)
            import copy
            detection_output_clone = copy.deepcopy(detection_output)
            segmentation_output_clone = copy.deepcopy(segmentation_output)
            segmentation_output = segmentation_output.squeeze(0)  # (1, 6, 64, 64) -> (6, 64, 64)
            pred_masks = torch.argmax(segmentation_output, dim=0).cpu()
            
            # for batch_idx, target in enumerate(targets):
            # print("Inside Target: ", targets)

            gt_masks = targets["masks"].cpu()  # (num_classes, H, W)
            gt_masks = torch.argmax(gt_masks, dim=0)

            import torch.nn.functional as F
            gt_masks = F.interpolate(gt_masks.unsqueeze(0).unsqueeze(0).float(),
                                    size=pred_masks.shape,
                                    mode="nearest").squeeze(0).squeeze(0).long()

            iou_per_class, mean_iou = compute_class_iou(pred_masks, gt_masks, 6)
            
            for cls, iou in iou_per_class.items():
                iou_scores[cls].append(iou)

            iou_scores["overall"].append(mean_iou)
            total_samples += 1

            outputs = []
            image_ids = [int(targets["image_id"])]

            outputs = format_outputs(detection_output, segmentation_output, image_ids)
            # print("Output: ", outputs)
        
            outputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs[0].items()}
            model_time = time.time() - model_time

            res = {targets["image_id"]: outputs}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            coco_evaluator.synchronize_between_processes()

            coco_evaluator.accumulate()
            coco_evaluator.summarize()
            torch.set_num_threads(n_threads)

            print("\nSegmentation IoU Results:")
            mean_iou_per_class = {}
            for cls in range(1, 6):  # Only count foreground classes
                if cls in iou_scores:
                    mean_iou = np.mean(iou_scores[cls]) if len(iou_scores[cls]) > 0 else 0
                    mean_iou_per_class[f"IoU Class {cls}"] = mean_iou
                    print(f"  Class {cls}: IoU = {mean_iou:.4f}")

            overall_iou = np.mean(iou_scores["overall"]) if len(iou_scores["overall"]) > 0 else 0
            print(f"\nOverall IoU = {overall_iou:.4f}")

            # Store COCO evaluation metrics
            coco_eval_stats = coco_evaluator.coco_eval['bbox'].stats
            results_storage["mAP_50_95"].append(coco_eval_stats[0])
            results_storage["mAP_50"].append(coco_eval_stats[1])
            results_storage["mAP_75"].append(coco_eval_stats[2])
            results_storage["mAP_small"].append(coco_eval_stats[3])
            results_storage["mAP_medium"].append(coco_eval_stats[4])
            results_storage["mAP_large"].append(coco_eval_stats[5])

            results_storage["mAR_1"].append(coco_eval_stats[6])
            results_storage["mAR_10"].append(coco_eval_stats[7])
            results_storage["mAR_100"].append(coco_eval_stats[8])
            results_storage["mAR_small"].append(coco_eval_stats[9])
            results_storage["mAR_medium"].append(coco_eval_stats[10])
            results_storage["mAR_large"].append(coco_eval_stats[11])

            # Store segmentation IoU per class
            for cls, iou in mean_iou_per_class.items():
                results_storage["segmentation_iou"][cls].append(iou)

            # Store overall segmentation IoU
            results_storage["overall_iou"].append(overall_iou)

            # test(model, test_loader, device, coco_evaluator, overall_iou, mean_iou_per_class)
            # img = img.unsqueeze(0).to(device)

            # with torch.no_grad():
            #     detection_output, segmentation_output, _ = model(img, training=False)

            pred_boxes = detection_output_clone[0]["boxes"].cpu().numpy()
            pred_scores = detection_output_clone[0]["scores"].cpu().numpy()
            pred_labels = detection_output_clone[0]["labels"].cpu().numpy()
            pred_masks = torch.argmax(segmentation_output_clone[0], dim=0).cpu().numpy()

            img_np = images.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)

            # Ground truth
            true_boxes = targets["boxes"].cpu().numpy()
            true_labels = targets["labels"].cpu().numpy()
            mask_gt = targets["masks"].squeeze(0).cpu().numpy()
            mask_gt = np.argmax(mask_gt, axis=0)

            # Create color segmentation image
            pred_masks_color = np.zeros((pred_masks.shape[0], pred_masks.shape[1], 3), dtype=np.uint8)
            mask_gt_color = np.zeros((mask_gt.shape[0], mask_gt.shape[1], 3), dtype=np.uint8)

            for class_name, class_idx in selected_classes.items():
                if class_name in color:
                    pred_masks_color[pred_masks == class_idx] = color[class_name]
                    mask_gt_color[mask_gt == class_idx] = color[class_name]

            # Overlay segmentation results on the original image
            pred_masks_color_resized = cv2.resize(pred_masks_color, (img_np.shape[1], img_np.shape[0]))
            mask_gt_color_resized = cv2.resize(mask_gt_color, (img_np.shape[1], img_np.shape[0]))
            img_with_pred_mask = cv2.addWeighted(img_np, 1, pred_masks_color_resized, 0.5, 0)
            img_with_gt_mask = cv2.addWeighted(img_np, 1, mask_gt_color_resized, 0.5, 0)

            # Draw ground truth bounding boxes
            for b in true_boxes:
                cv2.rectangle(img_np, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)  # Blue for GT

            # Draw predicted bounding boxes with confidence > 0.5
            for j in range(len(pred_boxes)):
                box = pred_boxes[j]
                label = idx_to_class[pred_labels[j]]
                score = pred_scores[j]

                if score > 0.5:
                    color_box = color[label] if label in color else (0, 255, 0)  # Default to green if color is missing
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_box, 2)
                    cv2.putText(img_np, f"{label}: {score:.2f}", (int(box[0]), int(box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 2)

            # Plot all visualizations
            plt.figure(figsize=(12, 6))

            # Image with object detection results
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title(f"Detection Results\nmAP@0.5: {coco_evaluator.coco_eval['bbox'].stats[0]:.3f} | mAP@0.75: {coco_evaluator.coco_eval['bbox'].stats[1]:.3f}")

            # Image with predicted segmentation mask
            plt.subplot(1, 3, 2)
            plt.imshow(img_with_pred_mask, cmap="jet", alpha=0.6)
            plt.title(f"Segmentation Results - IoU: {overall_iou:.3f}")

            # Image with ground truth segmentation mask
            plt.subplot(1, 3, 3)
            plt.imshow(img_with_gt_mask, cmap="jet", alpha=0.6)
            plt.title("Ground Truth")

            plt.tight_layout()
            # plt.savefig(f"results/{idx[:-4]}_result.png")
            

            for cls, iou in mean_iou_per_class.items():
                print(f"  {cls}: IoU = {iou:.4f}")

            # n_threads = torch.get_num_threads() # Save number of threads    
            metric_logger = utils_coco.MetricLogger(delimiter="  ")
            header = "Test:"

            # coco = convert_to_coco(test_loader.dataset)
            # iou_types = ["bbox"]
            # coco_evaluator = CocoEvaluator(coco, iou_types)
            coco_evaluator.clear()
            iou_scores = defaultdict(list)

            total_samples = 0

    # Compute final average statistics
    final_results = {
        "Mean mAP 50-95": np.mean(results_storage["mAP_50_95"]),
        "Mean mAP 50": np.mean(results_storage["mAP_50"]),
        "Mean mAP 75": np.mean(results_storage["mAP_75"]),
        "Mean mAP Small": np.mean(results_storage["mAP_small"]),
        "Mean mAP Medium": np.mean(results_storage["mAP_medium"]),
        "Mean mAP Large": np.mean(results_storage["mAP_large"]),
        "Mean mAR 1": np.mean(results_storage["mAR_1"]),
        "Mean mAR 10": np.mean(results_storage["mAR_10"]),
        "Mean mAR 100": np.mean(results_storage["mAR_100"]),
        "Mean mAR Small": np.mean(results_storage["mAR_small"]),
        "Mean mAR Medium": np.mean(results_storage["mAR_medium"]),
        "Mean mAR Large": np.mean(results_storage["mAR_large"]),
        "Mean Overall IoU": np.mean(results_storage["overall_iou"])
    }

    # Compute per-class IoU averages
    for cls in results_storage["segmentation_iou"]:
        final_results[f"Mean IoU Class {cls}"] = np.mean(results_storage["segmentation_iou"][cls])

    # Print final results
    print("\n============================")
    print(" Final Evaluation Results")
    print("============================")
    for metric, value in final_results.items():
        print(f"  {metric}: {value:.4f}")
    print("============================\n")

   

    #####################################################################
    # Parallel Results
    #####################################################################
    # ============================
    # Final Evaluation Results
    # ============================
    # Mean mAP 50-95: 0.1007
    # Mean mAP 50: 0.3115
    # Mean mAP 75: 0.0462
    # Mean mAP Small: -0.1154
    # Mean mAP Medium: -0.0170
    # Mean mAP Large: 0.0325
    # Mean mAR 1: 0.0705
    # Mean mAR 10: 0.1291
    # Mean mAR 100: 0.1398
    # Mean mAR Small: -0.0959
    # Mean mAR Medium: 0.0111
    # Mean mAR Large: 0.0696
    # Mean Overall IoU: 0.2921
    # Mean IoU Class IoU Class 1: 0.4886
    # Mean IoU Class IoU Class 2: 0.1612
    # Mean IoU Class IoU Class 5: 0.1745
    # Mean IoU Class IoU Class 3: 0.2246
    # Mean IoU Class IoU Class 4: 0.0025
    # ============================



    #####################################################################
    # Flow Results
    #####################################################################
    # ============================
    # Final Evaluation Results
    # ============================
    # Mean mAP 50-95: 0.2440
    # Mean mAP 50: 0.5110
    # Mean mAP 75: 0.2061
    # Mean mAP Small: -0.0762
    # Mean mAP Medium: 0.1150
    # Mean mAP Large: 0.2189
    # Mean mAR 1: 0.1715
    # Mean mAR 10: 0.2757
    # Mean mAR 100: 0.2827
    # Mean mAR Small: -0.0495
    # Mean mAR Medium: 0.1442
    # Mean mAR Large: 0.2387
    # Mean Overall IoU: 0.3971
    # Mean IoU Class IoU Class 1: 0.6198
    # Mean IoU Class IoU Class 2: 0.2759
    # Mean IoU Class IoU Class 5: 0.2380
    # Mean IoU Class IoU Class 3: 0.2968
    # Mean IoU Class IoU Class 4: 0.0022
    # ============================