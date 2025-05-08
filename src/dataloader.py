import os
from typing import Counter
from matplotlib import patches, pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET

from utils import process_folder_color

class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, anno_dir, selected_classes, transforms=None, plot=False):
        """
        CamVid Dataset (Mask + BBox from VOC XML)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.anno_dir = anno_dir
        self.transforms = transforms  # Preprocessing
        self.selected_classes = selected_classes  # Target classes

        # Get all image & Mask & XML files
        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        self.xml_files = sorted(os.listdir(self.anno_dir))
        self.plot = plot

    def get_name(self, idx):
        return self.img_files[idx]

    def get_bboxes_from_xml(self, xml_path):
        """
        Parse bounding box information from XML file
        """
        bboxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for obj in root.iter("object"):
            class_name = obj.find("name").text
            if class_name in self.selected_classes:
                bboxes.append([
                    float(obj.find(".//bndbox/xmin").text),
                    float(obj.find(".//bndbox/ymin").text),
                    float(obj.find(".//bndbox/xmax").text),
                    float(obj.find(".//bndbox/ymax").text)
                ])
                labels.append(self.selected_classes[class_name])  # Convert class name to ID
        
        return bboxes, labels

    def get_masks_from_rgb(self, rgb_mask):
        """
        Generate corresponding object detection masks from RGB image, where each pixel value corresponds to a class ID.
        Here it is assumed that each class corresponds to a specific RGB color value.
        """
        masks_list = []
        labels = []
        palette = [
            0, 0, 0,        # Void (Background class) needs to be filled
            64, 0, 128,     # Car
            64, 64, 0,      # Pedestrian
            0, 128, 192,    # Bicyclist
            192, 0, 192,    # MotorcycleScooter
            192, 128, 192,  # Truck_Bus            
        ]
        # Traverse each color defined in the palette to generate the corresponding binary mask
        for idx in range(0, len(palette), 3):
            # Get the RGB color value of the class
            class_rgb = np.array(palette[idx:idx+3], dtype=np.uint8)
            # print(f"Class {idx // 3}: {class_rgb}")
            # Create a binary mask: if the pixel color matches the class color, set to 1, otherwise set to 0
            binary_mask = np.all(rgb_mask == class_rgb, axis=-1).astype(np.uint8)

            # if np.sum(binary_mask) == 0:
            #     continue  # Skip if the class is not in the mask
            # print(f"Class {idx // 3}: {np.sum(binary_mask)} pixels")
            # Generate the mask for the class
            class_mask = np.zeros_like(rgb_mask[:, :, 0], dtype=np.uint8)
            class_mask[binary_mask == 1] = idx//3   # Fill the mask area with id
            # print(class_mask)
            masks_list.append(class_mask)  # Store the mask of the class
            labels.append(idx // 3)  # Class label (idx is the position of RGB, divided by 3 to get the class ID)
            # visualize the mask
            # plt.imshow(class_mask, cmap="gray")
            # plt.show()
            # plt.imshow(rgb_mask)
            # plt.show()
            # labels.append(idx // 3)  # Class label (idx is the position of RGB, divided by 3 to get the class ID)

        return masks_list, labels


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        # print("Image Name: ", img_name)
        # mask_name = img_name[:-4] +"_L.png"
        xml_name = img_name[:-4] + ".xml"
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        xml_path = os.path.join(self.anno_dir, xml_name)

        # image = read_image(img_path)  # Read RGB image.
        # Convert to FloatTensor when getting the image
        image = read_image(img_path).float() / 255.0  # Normalize and convert to FloatTensor

        mask = np.array(Image.open(mask_path).convert("RGB"))  # Read mask and convert to NumPy
        # print(self.get_name(idx))
        # Get bounding boxes and labels from XML
        bboxes, labels = self.get_bboxes_from_xml(xml_path)
        # print("size of bboxes: ", len(bboxes))
        # print("size of labels: ", len(labels))
        if len(bboxes) == 0 :
            # print(f"Skipping empty sample {idx}")
            return self.__getitem__((idx + 1))  # Get the next sample again

        # Get masks from RGB image
        masks_list, labels_m = self.get_masks_from_rgb(mask)
        # print("Labels: ", labels)
        # print("Labels_m: ", labels_m)
        # Merge XML and Mask information
        labels = labels
        bboxes = bboxes
        masks = masks_list
        # masks = self.match_masks_to_labels(bboxes, labels, masks_list)

        # print("Masks: ", masks)
        # print(masks_list)
         # Convert to `tv_tensors`
        img = tv_tensors.Image(image)
        # print(f"idx type: {type(idx)}")  # Check the type of idx

        target = {
            "boxes": tv_tensors.BoundingBoxes(bboxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            # "area": area,
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64)
        }
        if self.transforms:
            if len(target["masks"]) == 0:
                # Ensure `bboxes` is a NumPy array
                if isinstance(target["boxes"], list):
                    bboxes = np.array(target["boxes"], dtype=np.float32)
                    masks = np.array(target["masks"], dtype=np.uint8)
                else:
                    bboxes = target["boxes"].cpu().numpy()  # If it is a PyTorch Tensor, convert to NumPy
                    # labels = target["labels"].cpu().numpy()  # If it is a PyTorch Tensor, convert to NumPy
                    masks = target["masks"].cpu().numpy()  # If it is a PyTorch Tensor, convert to NumPy
                labels = np.array(target["labels"], dtype=np.int64)
                # Apply transformations
                img_ori = img.clone()
                # print(img.shape)
                img = np.transpose(img, (1, 2, 0)) 
                num_classes, h, w = target["masks"].shape  # (N, H, W)
                # Ensure `img` is a NumPy array
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                transformed = self.transforms(image=img, bboxes=bboxes.tolist(), labels=labels.tolist())
                mask_vis = np.zeros((256,256, 3), dtype=np.uint8)
                target["boxes"] = torch.tensor(torch.zeros((0, 4), dtype=torch.float32))
                target["labels"] = torch.zeros((0,), dtype=torch.int64)
                img = tv_tensors.Image(transformed["image"])
                target["masks"] = [np.zeros((256,256), dtype=np.uint8)]
                if self.plot :
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    ax[0].imshow(img_ori.permute(1, 2, 0).numpy())
                    ax[0].set_title("Transformed Image")
                    
                    ax[1].imshow(mask_vis)
                    # save the mask image
                    plt.show()
                # mask_save_path = f"CamVid/annotations/train_masks/{self.get_name(idx)}"
                # plt.imsave(mask_save_path, mask_vis)
                # print(img.shape)
                # print("target type:", type(target["masks"]))
                return img, target, self.get_name(idx)
             
            else:
                img = np.transpose(img, (1, 2, 0)) 

                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                
                if isinstance(target["boxes"], list):
                    bboxes = np.array(target["boxes"], dtype=np.float32)
                    masks = np.array(target["masks"], dtype=np.uint8)
                else:
                    bboxes = target["boxes"].cpu().numpy() 
                    masks = target["masks"].cpu().numpy()  
                    # print("mask before transform:", masks)
                labels = np.array(target["labels"], dtype=np.int64)
                # Apply transformations
                transformed = self.transforms(image=img, bboxes=bboxes.tolist(), labels=labels.tolist(), masks=masks)
                # print(transformed)
                img = tv_tensors.Image(transformed["image"])
                target["masks"] = tv_tensors.Mask(transformed["masks"])
                # print("mask after transform:", target["masks"])

                target["boxes"] =  tv_tensors.BoundingBoxes(transformed["bboxes"], format="XYXY", canvas_size=F.get_size(img))
                target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)
                target["iscrowd"] = torch.zeros((len(transformed["bboxes"]),), dtype=torch.int64)
                if self.plot :
                    # print(target["masks"])
                    # Plot the transformed image and mask
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    # plt.show()

                    ax[0].imshow(img.permute(1, 2, 0))
                    ax[0].set_title("Transformed Image")
                    num_classes, h, w = target["masks"].shape  # (N, H, W)
                    num_classes = len(labels)
                    # print("num_classes: ", num_classes)
                    cmap = {
                        0: "black",   # Background
                        1: "cyan",    # Car
                        2: "red",     # Pedestrian
                        3: "yellow",  # Bicyclist
                        4: "green",   # MotorcycleScooter
                        5: "purple"   # Truck_Bus
                    }
                    class_names = {
                        "Background": 0,
                        "Car": 1,
                        "Pedestrian": 2,
                        "Bicyclist": 3,
                        "MotorcycleScooter": 4,
                        "Truck_Bus": 5
                    }
                
                    mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
                    # fig, ax = plt.subplots(figsize=(8, 6))
                    ax[1].set_title("Visualized Masks")
                  
                    # print("labels: ", labels)
                    # print(num_classes)
                    for i in range(num_classes):
                        # if i>=len(labels):
                        #     break
                        label = labels[i].item()
                       
                        if label in cmap:
                            # print(f"Class {label}")
                            color = np.array(mcolors.to_rgb(cmap[label])) * 255
                            # print(f"Class {label}: {class_names.get(label, 'Unknown')} - {color}")
                            mask_vis[target["masks"][label]!=0] = color
                            # print(color)
                            # mask_vis[target["masks"][i] > 0] = color

                        ax[1].imshow(mask_vis)
                    
                    
                    for i in range(len(target["boxes"][0])):
                        # print(target["boxes"][0][i])
                        x_min, y_min, x_max, y_max = target["boxes"][0][i].tolist()
                        label = target["labels"][i].item()
                        class_name = class_names.get(label, f"Class {label}")
                        
                        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_max,
                                                linewidth=2, edgecolor="white", facecolor="none")
                        ax[1].add_patch(rect)

                        ax[1].text(x_min, y_min - 5, class_name, fontsize=8, color="white",
                                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none"))

                    plt.axis("off")
                    plt.show()
                # print("Unique values in masks:", torch.unique(target["masks"]).tolist())
                # print("Unique values in labels:", torch.unique(target["labels"]).tolist())
            
                return img, target, self.get_name(idx)

if __name__ == "__main__":
    # Set input & output folder paths
    input_folder = "CamVidColor5/train"    # Replace with your input folder path
    output_folder = "CamVidColor5/train_he_color"  # Replace with your output folder path
    process_folder_color(input_folder, output_folder)

    input_folder = "CamVidColor5/test"    # Replace with your input folder path
    output_folder = "CamVidColor5/test_he_color"  # Replace with your output folder path
    process_folder_color(input_folder, output_folder)

    input_folder = "CamVidColor5/val"    # Replace with your input folder path
    output_folder = "CamVidColor5/val_he_color"  # Replace with your output folder path
    process_folder_color(input_folder, output_folder)

    selected_classes = {
        "Background": 0, 
        "Car": 1,
        "Pedestrian": 2,
        "Bicyclist": 3,
        "MotorcycleScooter": 4,
        "Truck_Bus": 5,
    }

    transform = A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),  # ±10 degree rotation
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.1))

    train_dataset = CamVidDataset(
        img_dir="CamVidColor5/train_he_color",
        mask_dir="CamVidColor5/train_labels",
        anno_dir="CamVidColor5/train_annotations",
        selected_classes=selected_classes,
        transforms=transform,
        # plot=False
    )

    print(f"Total samples: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Count the total number of pixels for each class (label occurrence count)
    class_pixel_counts = Counter()
    total_pixels = 0  # Calculate the total number of pixels

    class_instance_count =  Counter()
    total_instance = 0

    for _, target, idx in train_dataset:
        # print(f"Image ID: {idx}")
        masks = target["masks"]  # Shape: (num_classes, H, W), each channel represents a class mask
        for class_idx in range(1, masks.shape[0]):  # Traverse all classes
            # print(f"Class {class_idx}: {torch.sum(masks[class_idx])} pixels")
            pixel_count = torch.sum(masks[class_idx]).item()  # Calculate the number of pixels for the current class
            class_pixel_counts[class_idx] += pixel_count  # Accumulate class pixel count
            total_pixels += pixel_count  # Calculate the total number of pixels in the entire dataset
        
        labels = target["labels"]
        for lab in labels:
            class_instance_count[lab.item()] += 1
            total_instance += 1

    image_area = masks.shape[1] * masks.shape[2] * len(train_dataset)  # Image area = H * W * num_samples
    background_pixels = image_area - total_pixels
    class_pixel_counts[0] = background_pixels
    print("For Segmentation:")

    # Print statistics
    print("Class pixel statistics:", class_pixel_counts)
    print("Total number of pixels:", total_pixels)

    # Calculate the pixel proportion (frequency) of each class
    class_frequencies = {cls: count / total_pixels for cls, count in class_pixel_counts.items()}
    print("Class pixel proportion:", class_frequencies)

    # Calculate class weights (using `1 / frequency` or `log` normalization)
    class_weights = {cls: 1.0 / freq if freq > 0 else 0.0 for cls, freq in class_frequencies.items()}
    print("Class weights:", class_weights) # [0.07, 2.15, 5.42, 4.77, 304.47, 7.26]

    print("For Detection:")
    # Calculate the instance proportion of each class (class frequency)
    class_frequencies = {cls: count / total_instance for cls, count in class_instance_count.items()}

    # Calculate class weights (inverse frequency 1/freq, prevent division by zero)
    class_weights = {cls: 1.0 / freq if freq > 0 else 0.0 for cls, freq in class_frequencies.items()}

    # Print statistics
    print("Class instance statistics:", class_instance_count)
    print("Total number of instances:", total_instance)
    print("Class instance proportion:", class_frequencies)
    print("Class weights:", class_weights)

    # For Segmentation:
    # Class pixel statistics: Counter({0: 360903045, 1: 12095875, 3: 5456841, 2: 4797114, 5: 3586225, 4: 85444})
    # Total number of pixels: 26021499
    # Class pixel proportion: {1: 0.46484159117812546, 2: 0.1843519468267374, 3: 0.2097050980806294, 4: 0.0032835925401530483, 5: 0.13781777137435472, 0: 13.86941793783671}
    # Class weights: {1: 2.151270495106803, 2: 5.424407049738655, 3: 4.7686012841495655, 4: 304.54448527690647, 5: 7.255958284825966, 0: 0.07210107911392102}
    # For Detection:
    # Class instance statistics: Counter({2: 1278, 1: 1062, 3: 280, 5: 118, 4: 8})
    # Total number of instances: 2746
    # Class instance proportion: {3: 0.10196649672250546, 1: 0.38674435542607427, 2: 0.4654042243262928, 5: 0.04297159504734159, 4: 0.0029133284777858705}
    # Class weights: {3: 9.807142857142857, 1: 2.585687382297552, 2: 2.1486697965571206, 5: 23.271186440677965, 4: 343.25}
