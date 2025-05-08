from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataloader import CamVidDataset
from model_flow import MultiHeadModel
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from utils import compute_ap, compute_iou, compute_class_iou
import math
import sys
import time
import torch
import utils_coco
from coco_eval import CocoEvaluator
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import torch.nn.functional as F

def convert_to_coco(dataset):
    """
    Convert a dataset (CamVidDataset) into COCO API format.

    Args:
        dataset (Dataset): Your custom dataset (CamVidDataset).

    Returns:
        coco_ds (COCO): COCO-formatted dataset for evaluation.
    """
    coco_ds = COCO()
    ann_id = 1  # COCO annotations ID must start from 1
    dataset_dict = {"images": [], "categories": [], "annotations": []}
    categories = set()

    for img_idx in range(len(dataset)):
        img, targets, img_name = dataset[img_idx]  # Get image, target info, and image name
        image_id = targets["image_id"]

        # Add image info
        img_dict = {
            "id": image_id,
            "file_name": img_name,  # Store file name
            "height": img.shape[-2],  # H
            "width": img.shape[-1],   # W
        }
        dataset_dict["images"].append(img_dict)

        # Parse target info
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]  # Convert to (x, y, width, height)
        bboxes = bboxes.tolist()

        labels = targets["labels"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        # Handle masks (if any)
        masks = targets.get("masks", None)
        old_mask = masks.clone()
        if masks is not None:
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)  # Ensure Fortran contiguous
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": labels[i],
                "bbox": bboxes[i],  # COCO requires (x, y, width, height)
                "iscrowd": iscrowd[i],
                "area": bboxes[i][2] * bboxes[i][3],  # Calculate area
            }
            categories.add(labels[i])  # Add category info

            mask_cls_idx = labels[i]  # Get bbox corresponding class index
            if 1 <= mask_cls_idx <= 5:  # Ensure valid class
                mask = masks[mask_cls_idx]  # Get mask of the class (whole image)
                x_min, y_min, w, h = map(int, bboxes[i])  # Get bbox of the object
                mask = mask[y_min:y_min+h, x_min:x_min+w]  # Get mask within bbox
                mask = (mask > 0).to(torch.uint8)  # Binarization
                ann["segmentation"] = coco_mask.encode(np.asfortranarray(mask.cpu().numpy()))  # Convert to NumPy and encode

            dataset_dict["annotations"].append(ann)
            ann_id += 1

    # COCO categories format
    dataset_dict["categories"] = [{"id": i, "name": f"class_{i}"} for i in sorted(categories)]

    # Create COCO API
    coco_ds.dataset = dataset_dict
    coco_ds.createIndex()
    return coco_ds

def format_outputs(detection_outputs, segmentation_outputs, image_ids):
    """
    Combine batch dimension detection and segmentation results, generate COCO evaluation format
    """
    formatted_outputs = []

    batch_size = len(image_ids)  # Get batch size
    for i in range(batch_size):
        detection_output = detection_outputs[i]  # Get detection result of the i-th image in the batch
        segmentation_output = segmentation_outputs[i]  # Get segmentation result of the i-th image in the batch
        image_id = int(image_ids[i])  # COCO requires int type image_id

        output = {
            "image_id": image_id,
            "boxes": detection_output["boxes"],  # (num_objs, 4)
            "labels": detection_output["labels"],  # (num_objs,)
            "scores": detection_output["scores"],  # (num_objs,)
            "masks": segmentation_output,  # (num_classes, H, W)
        }
        formatted_outputs.append(output)

    return formatted_outputs  # Return formatted outputs list




@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    cpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    metric_logger = utils_coco.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = convert_to_coco(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    iou_scores = defaultdict(list)

    total_samples = 0
    for images, targets, idx in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        detection_output, segmentation_output, seg_loss = model(images, img_name=idx, epoch=epoch, training=False)
        
        segmentation_output = segmentation_output.squeeze(0)  # (1, 6, 64, 64) -> (6, 64, 64)
        pred_masks = torch.argmax(segmentation_output, dim=0).cpu()
        
        for batch_idx, target in enumerate(targets):
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
        metric_logger.update_train(model_time=model_time, evaluator_time=evaluator_time)

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
            mean_iou_per_class[f"Val/IoU Class {cls}"] = mean_iou
            print(f"  Class {cls}: IoU = {mean_iou:.4f}")

    overall_iou = np.mean(iou_scores["overall"]) if len(iou_scores["overall"]) > 0 else 0
    print(f"\nOverall IoU = {overall_iou:.4f}")

    return coco_evaluator, overall_iou

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils_coco.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils_coco.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    warmup_epochs = 5
    lr_scheduler = None
    if epoch < warmup_epochs:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    total_loss = 0.0
    total_det_loss = 0.0
    total_seg_loss = 0.0
    num_batches = len(data_loader)

    for images, targets, idx in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            detection_output_dict, det_loss, segmentation_output, seg_loss = model(images, targets, idx, epoch, training=True)
            loss_dict = detection_output_dict
            loss_dict["segmentation_loss"] = seg_loss 
            losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = utils_coco.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update_train(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update_train(lr=optimizer.param_groups[0]["lr"])
        total_loss += loss_value
        total_det_loss += det_loss.item()
        total_seg_loss += seg_loss.item()

    avg_total_loss = total_loss / num_batches
    avg_det_loss = total_det_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches

    return avg_total_loss, avg_det_loss, avg_seg_loss



def collate_fn(batch):
    """
    Collate function to handle the different number of bounding boxes and masks
    in each image. Pads the bounding boxes and masks to the maximum number in the batch.
    """
    max_num_objs = max(len(item[1]["boxes"]) for item in batch)

    batch_new = []
    for img, target, idx in batch:
        batch_new.append((img, target, idx))

    imgs = torch.stack([item[0] for item in batch_new], dim=0)

    return imgs, [item[1] for item in batch_new], [item[2] for item in batch_new]




if __name__ == "__main__":
    # wandb.init(project="project-LiXiangqi", name="frcnn-map + deeplabv3plus-iou") 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Rotate(limit=10, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1))

    transform_val = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.Rotate(limit=10, p=0.5),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1))

    selected_classes = {
        "Background": 0, 
        "Car": 1,
        "Pedestrian": 2,
        "Bicyclist": 3,
        "MotorcycleScooter": 4,
        "Truck_Bus": 5
    }

    train_dataset = CamVidDataset(
        img_dir="CamVidColor5/train_he_color", 
        mask_dir="CamVidColor5/train_labels", 
        anno_dir="CamVidColor5/train_annotations", 
        selected_classes=selected_classes,
        transforms=transform,
        plot=False
    )

    val_dataset = CamVidDataset(
        img_dir="CamVidColor5/val_he_color",
        mask_dir="CamVidColor5/val_labels",
        anno_dir="CamVidColor5/val_annotations",
        selected_classes=selected_classes,
        transforms=transform_val,
        plot=False
    )
    train_loader = DataLoader(train_dataset, batch_size= 4, num_workers = 8, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size= 1, shuffle=False, collate_fn=collate_fn, drop_last=True)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")
    for img, target, name in train_loader:
        print(f"Image shape: {img.shape}")
        print(f"Name: {name}")
        break
    for img, target, name in val_loader:
        print(f"Image shape: {img.shape}")
        print(f"Name: {name}")
        break

    model = MultiHeadModel(num_classes_detection=6, num_classes_segmentation=6)
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([
            {'params': model.detection_head.parameters(), 'lr': 1e-3},
            {'params': model.segmentation_head.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)
    num_epochs = 300
    early_stopping = 50
    early_stopping_counter = 0
    best_val_loss = float("inf")
    best_val_det_loss = float("inf")
    best_val_seg_loss = float("inf")
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs // 2)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}   LR for Detection Head: {optimizer.param_groups[0]['lr']}   LR for Segmentation Head: {optimizer.param_groups[1]['lr']}")
        start_time = time.time()
        train_total_loss, train_det_loss, train_seg_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=20
        )

        val_evaluator, val_seg_loss = evaluate(model, val_loader, device)
        val_loss = (1 - val_evaluator.coco_eval['bbox'].stats[0] + 1 - val_seg_loss) / 2

        lr_scheduler.step()
        if val_loss < best_val_loss:  
            best_val_loss = val_loss
            torch.save(model.state_dict(), "weights/multihead_best_model.pth")
            print("Saved new best model!")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        end_time = time.time()
        print(f"Epoch [{epoch+1}/{num_epochs}], Time: {end_time - start_time:.2f}s, Val Loss: {val_loss:.4f}, Val Seg IOU: {val_seg_loss:.4f}, Val Det MaP: {val_evaluator.coco_eval['bbox'].stats[0]:.4f}")
        # wandb.log({
        #     "Train/Total Loss": train_total_loss,
        #     "Train/Detection Loss": train_det_loss,
        #     "Train/Segmentation Loss": train_seg_loss,
        #     "Val/Total Loss": val_loss,
        #     "Val/Detection MaP": val_evaluator.coco_eval['bbox'].stats[0],
        #     "Val/Segmentation IOU": val_seg_loss,
        #     "Epoch": epoch
        # })
        
        if early_stopping_counter > early_stopping:
            print("Early stopping!")
            break
        
    # wandb.finish()
