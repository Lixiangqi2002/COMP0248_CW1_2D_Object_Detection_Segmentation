from utils import convert32to5, convertGraytoRGB, convertRGBtoAnno, RGBLabel

def main():
    """
    Main function to convert the CamVid dataset to the format required by the project
    """
    # 1. Generate Gray labels
    convert32to5(label_dir='CamVid/train_labels',
                  save_dir='CamVid_Gray/train_labels')
    # 2. From Gray labels, generate 5 classes Color labels
    convertGraytoRGB(gray_dir='CamVid_Gray/train_labels',
                     save_dir='CamVidColor5/train_labels')
    # 3. Generate xml annotations
    convertRGBtoAnno(rgb_dir='CamVidColor5/train_labels',
                      target_colors=RGBLabel,
                      class_names=['Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus'],
                     save_dir='CamVidColor5/train_bbx',
                     save_xml_dir='CamVidColor5/train_annotations') 
    
    # 1. Generate Gray labels
    convert32to5(label_dir='CamVid/test_labels',
                  save_dir='CamVid_Gray/test_labels')
    # 2. From Gray labels, generate 5 classes Color labels
    convertGraytoRGB(gray_dir='CamVid_Gray/test_labels',
                     save_dir='CamVidColor5/test_labels')
    # 3. Generate xml annotations
    convertRGBtoAnno(rgb_dir='CamVidColor5/test_labels',
                      target_colors=RGBLabel,
                      class_names=['Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus'],
                     save_dir='CamVidColor5/test_bbx',
                     save_xml_dir='CamVidColor5/test_annotations') 
    
    # 1. Generate Gray labels
    convert32to5(label_dir='CamVid/val_labels',
                  save_dir='CamVid_Gray/val_labels')
    # 2. From Gray labels, generate 5 classes Color labels
    convertGraytoRGB(gray_dir='CamVid_Gray/val_labels',
                     save_dir='CamVidColor5/val_labels')
    # 3. Generate xml annotations
    convertRGBtoAnno(rgb_dir='CamVidColor5/val_labels',
                      target_colors=RGBLabel,
                      class_names=['Bicyclist', 'Car', 'MotorcycleScooter', 'Pedestrian', 'Truck_Bus'],
                     save_dir='CamVidColor5/val_bbx',
                     save_xml_dir='CamVidColor5/val_annotations') # 


if __name__ == '__main__':
    main()
