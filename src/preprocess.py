import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preprocess_image(image, target_size=(416, 416)):
    original_h, original_w = image.shape[:2]
    target_w, target_h = target_size

    # Resize
    image = cv2.resize(image, (target_w, target_h))

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Optional: Light Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # Normalize
    image = image / 255.0

    return image, (original_w, original_h)


def adjust_boxes(boxes, original_size, target_size=(416, 416)):
    original_w, original_h = original_size
    target_w, target_h = target_size

    new_boxes = []
    for box in boxes:
        x1, y1, x2, y2, cls = box
        x1 = int(x1 * target_w / original_w)
        y1 = int(y1 * target_h / original_h)
        x2 = int(x2 * target_w / original_w)
        y2 = int(y2 * target_h / original_h)
        new_boxes.append([x1, y1, x2, y2, cls])
    
    return new_boxes


def plot_original_and_preprocessed_images(image_list, boxes_list, preprocessed_images, preprocessed_boxes_list, num_images=4):
    plt.figure(figsize=(15, 8))  # Adjust the figure size to fit your images better
    for i in range(num_images):
        # Original image with boxes
        plt.subplot(2, 4, i+1)
        image = image_list[i]
        boxes = boxes_list[i]
        
        plt.imshow(image)
        for box in boxes:
            x1, y1, x2, y2, cls = box
            plt.gca().add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                  edgecolor='red', facecolor='none', linewidth=2))
            plt.text(x1, y1 - 5, str(cls), color='red', fontsize=8)
        plt.title("Original Image")
        plt.axis('off')
        
        # Preprocessed image with boxes
        plt.subplot(2, 4, i+5)
        pre_img = preprocessed_images[i]
        pre_boxes = preprocessed_boxes_list[i]
        
        plt.imshow(pre_img.squeeze(), cmap='gray')  # Display grayscale image
        for box in pre_boxes:
            x1, y1, x2, y2, cls = box
            plt.gca().add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                  edgecolor='red', facecolor='none', linewidth=2))
            plt.text(x1, y1 - 5, str(cls), color='red', fontsize=8)
        plt.title("Preprocessed Image")
        plt.axis('off')

    # Adjust the layout to minimize gaps
    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)  # Adjust padding values
    plt.show()
