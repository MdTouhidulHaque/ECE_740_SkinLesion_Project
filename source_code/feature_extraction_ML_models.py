#extract the three features for baselines ML models

import cv2
import numpy as np
import os
import pandas as pd 
from skimage.feature import greycomatrix, greycoprops



def color_histogram(image, bins=32):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten()


def hu_moments(image):

    if image is None:
        raise ValueError("Input image is invalid or not found")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's Thresholding
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary_image)
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    return hu_moments


def haralick_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    haralick_features = [greycoprops(glcm, prop).mean() for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')]
    return np.array(haralick_features)


def extract_features(image_path):
    #extract and return all 3 features
    image = cv2.imread(image_path)
    color_hist = color_histogram(image)
    hu_mom = hu_moments(image)
    haralick = haralick_texture(image)
    return np.concatenate([color_hist, hu_mom, haralick])

def process_images(data_dir, output_fileName):
    features = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                file_path = os.path.join(class_dir, filename)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract features and append them with the label
                    feature_vector = extract_features(file_path)
                    features.append(feature_vector)
                    labels.append(label)
  

    #Save as .npz
    features_array = np.array(features, dtype=object)   
    labels_array = np.array(labels)
    fname= output_fileName + '.npz'
    np.savez(fname, features=features_array, labels=labels_array)

# Uncomment to extract features from train data
data_dir = '../data/train_balanced_160'  
output_fileName= '../extracted_features/train_extracted_features/train_extracted_features'
process_images(data_dir, output_fileName)

# Uncomment to extract features for test images from balanced dataset
data_dir = '../data/test_balanced_40'  
output_fileName= '../extracted_features/balanced_test_extracted_features/balanced_test_extracted_features'
process_images(data_dir, output_fileName)

# Uncomment to extract features for independent test dataset  
data_dir = '../data/test_independent'  
output_fileName= '../extracted_features/independent_test_extracted_features/independent_test_extracted_features'
process_images(data_dir, output_fileName)


