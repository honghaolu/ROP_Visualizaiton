# Step 1:Image enhancement.(norm、clahe、gamma、gray).

import os
import cv2
import numpy as np
from tqdm import tqdm

def my_PreProc(data):
    # Use the original images
    assert len(data.shape) == 4
    assert data.shape[3] == 3  # 3 channels (RGB)

    # My preprocessing:
    data = dataset_normalized(data)
    data = clahe_equalized(data)
    data = adjust_gamma(data, 1.2)
    data = rgb2gray(data)   # 1 channel(gray)

    return data

# Normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs[i] - np.min(imgs[i])) / (np.max(imgs[i]) - np.min(imgs[i]))) * 255
    return imgs_normalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization) for each channel
def clahe_equalized(imgs):
    assert len(imgs.shape) == 4  # 4D arrays
    assert imgs.shape[3] == 3  #  3 channels (RGB)

    # Create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)

    for i in range(imgs.shape[0]):
        for c in range(3):
            # Convert to 8-bit for CLAHE (assuming the input images are 8-bit)
            img_8bit = imgs[i][:, :, c].astype(np.uint8)
            imgs_equalized[i][:, :, c] = clahe.apply(img_8bit)

    return imgs_equalized

# Adjust gamma for each channel
def adjust_gamma(imgs, gamma=1.0):
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        for c in range(3):
            new_imgs[i][:, :, c] = cv2.LUT(np.array(imgs[i][:, :, c], dtype=np.uint8), create_gamma_lut(gamma))
    return new_imgs

# Create a lookup table mapping pixel values [0, 255] to adjusted gamma values
def create_gamma_lut(gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return table

#convert RGB image to gray
def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[3]==3)
    gray_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114 #灰度化加权平均值法
    gray_imgs = np.reshape(gray_imgs,(rgb.shape[0],rgb.shape[1],rgb.shape[2],1)) #灰度化后channel改为1
    return gray_imgs

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # Input directory containing subfolders with images
    DATA = "dataset\img_mask\images"
    images_preprocess = "dataset\img_mask\images_preprocess"

    os.makedirs(images_preprocess, exist_ok=True)

    # Collect image files in the subfolder
    image_files = [f for f in os.listdir(DATA) if f.endswith('.png')]

    for i in tqdm(range(len(image_files))):
        image = image_files[i]
        output_image_path = os.path.join(images_preprocess, image)
        output_image_path = output_image_path.split('.')[0] + 'ours.png'
        if os.path.exists(output_image_path):
            continue

        image_path = os.path.join(DATA, image)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array([img])
        print(f"\n -------The NO.{i + 1} img---------")
        print(f" original image, min value: {img.min()},  max value: {img.max()}, shap: {img.shape}")
        preproc_image = my_PreProc(img)[0]
        print(
            f" preprocess image, min value: {preproc_image.min()},  max value: {preproc_image.max()}, shap: {preproc_image.shape}")

        # Save preprocessed images in the output subfolder
        cv2.imwrite(output_image_path, preproc_image)