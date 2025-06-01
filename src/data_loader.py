import cv2
import matplotlib.pyplot as plt
import os

def load_image_and_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_center, y_center, width, height = map(float, parts)
                    
                    # Check if values are valid
                    if width == 0 or height == 0:
                        continue
                    
                    # Convert to pixel values
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    boxes.append([x1, y1, x2, y2, int(cls)])
    return image, boxes


def plot_multiple_images(image_list, boxes_list, num_images=6):
    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(2, 3, i+1)
        image = image_list[i]
        boxes = boxes_list[i]
        
        plt.imshow(image)
        for box in boxes:
            x1, y1, x2, y2, cls = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              edgecolor='red', facecolor='none', linewidth=2))
            plt.text(x1, y1 - 5, str(cls), color='red', fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
