import os
import random
import cv2
import numpy as np
from src.preprocess import preprocess_image

class WeaponDetectionDataset:
    def __init__(self, image_folder, label_folder, batch_size=8, target_size=(416, 416), shuffle=True):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        
        self.image_list = os.listdir(self.image_folder)
        if self.shuffle:
            random.shuffle(self.image_list)
    
    def __len__(self):
        return len(self.image_list) // self.batch_size

    def __getitem__(self, idx):
        batch_images = []
        batch_boxes = []

        batch_files = self.image_list[idx*self.batch_size : (idx+1)*self.batch_size]
        
        for filename in batch_files:
            img_path = os.path.join(self.image_folder, filename)
            label_path = os.path.join(self.label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pre_img, original_size = preprocess_image(image, self.target_size)

            boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        cls, x_center, y_center, width, height = map(float, line.strip().split())

                        if width == 0 or height == 0:
                            continue
                        
                        # Recalculate bbox according to new size
                        orig_w, orig_h = original_size
                        x_center *= orig_w
                        y_center *= orig_h
                        width *= orig_w
                        height *= orig_h

                        # Scale to new target size
                        scale_x = self.target_size[0] / orig_w
                        scale_y = self.target_size[1] / orig_h

                        x_center *= scale_x
                        y_center *= scale_y
                        width *= scale_x
                        height *= scale_y

                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)

                        boxes.append([x1, y1, x2, y2, int(cls)])
            
            batch_images.append(pre_img)
            batch_boxes.append(boxes)

        return np.array(batch_images), batch_boxes
