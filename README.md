# **Multi-Head Detection and Segmentation Framework**
This project implements a multi-head detection and segmentation framework using **DeepLabV3+** for segmentation and **Faster R-CNN** for object detection. Two modes are supported:

- **Parallel Mode**: Independent segmentation and detection branches.
- **Flow Mode**: Segmentation-guided detection, where segmentation influences detection results.

---

## **Dataset Preparation**
The dataset is based on **CamVid**, and preprocessing steps generate segmentation masks and object detection annotations.

### **camvid_get_data.py**
- Converts original CamVid images into **CamVid_Gray** and **CamVidColor5** folders, extracting **5 semantic classes**.
- Generates **bounding boxes based on color segmentation**.
- Produces **COCO-style XML annotations** for detection.

### **dataloader.py**
- Performs **histogram equalization** on images.
- Computes **class statistics at pixel and object levels**.

#### **Class Statistics**
##### **Segmentation (Pixel Level)**
    Class pixel statistics: {0: 360903045, 1: 12095875, 3: 5456841, 2: 4797114, 5: 3586225, 4: 85444} 
    Class pixel proportions: {Car: 46.48%, Pedestrian: 18.44%, Bicyclist: 20.97%, Motorcycle/Scooter: 0.33%, Truck/Bus: 13.78%} 
    Computed class weights: {Car: 2.15, Pedestrian: 5.42, Bicyclist: 4.76, Motorcycle/Scooter: 304.54, Truck/Bus: 7.25}

##### **Detection (Instance Level)**
    Class instance statistics: {Car: 1062, Pedestrian: 1278, Bicyclist: 280, Motorcycle/Scooter: 8, Truck/Bus: 118} 
    Instance proportions: {Car: 38.67%, Pedestrian: 46.54%, Bicyclist: 10.20%, Motorcycle/Scooter: 0.29%, Truck/Bus: 4.30%} 
    Computed class weights: {Car: 2.58, Pedestrian: 2.14, Bicyclist: 9.80, Motorcycle/Scooter: 343.25, Truck/Bus: 23.27}


---

## **Model Implementations**
### **deeplabv3plus_custom.py**
- Custom **DeepLabV3+** segmentation model.

### **faster_rcnn.py**
- **Parallel Mode** Faster R-CNN (from torchvision).

### **faster_rcnn_seg.py**
- **Flow Mode** Faster R-CNN (from torchvision).

### **model_flow.py**
- **Segmentation-guided detection**, where segmentation influences the detection process.

### **model_parallel.py**
- **Independent segmentation and detection**, where both tasks are completely separate.

---

## **Training and Testing**
### **train.py**
- Used to train models. By default, trains in **Flow Mode**.

### **test.py**
- Evaluates both models by selecting the mode (`flow` or `parallel`).
- Computes **mAP, mAR, and IoU** metrics.

### **Evaluation Results**
#### **Parallel Mode**
    Mean mAP (50-95): 0.1007
    Mean mAP (50): 0.3115
    Mean mAP (75): 0.0462
    Mean mAP (Medium): -0.0170
    Mean mAP (Large): 0.0325
    Mean mAR (1): 0.0705
    Mean mAR (10): 0.1291
    Mean mAR (100): 0.1398
    Mean mAR (Medium): 0.0111
    Mean mAR (Large): 0.0696
    Mean Overall IoU: 0.2921
    IoU per Class: Car (0.4886), Pedestrian (0.1612), Bicyclist (0.2246), Motorcycle/Scooter (0.0025), Truck/Bus (0.1745)
#### **Flow Mode**


    Mean mAP (50-95): 0.2440
    Mean mAP (50): 0.5110
    Mean mAP (75): 0.2061
    Mean mAP (Medium): 0.1150
    Mean mAP (Large): 0.2189
    Mean mAR (1): 0.1715
    Mean mAR (10): 0.2757
    Mean mAR (100): 0.2827
    Mean mAR (Medium): 0.1442
    Mean mAR (Large): 0.2387
    Mean Overall IoU: 0.3971
    IoU per Class: Car (0.6198), Pedestrian (0.2759), Bicyclist (0.2968), Motorcycle/Scooter (0.0022), Truck/Bus (0.2380)

---

## **Model Weights**
Stored in the `weights/` directory:
- `multihead_best_model_parallel.pth`: Parallel mode weights.
- `multihead_best_model_seg+det_flow.pth`: Flow mode weights.

---

## **Results**
Stored in the `results/` directory:
- `results_flow/`: **Flow Mode** test outputs.
- `results_parallel/`: **Parallel Mode** test outputs.

---

## **Dependencies**
Ensure all required libraries are installed:
```bash
pip install -r requirements.txt
