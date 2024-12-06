#Image pre processing for cnn is done here. Exact same preprocessing steps as the original paper
import os
import cv2 
import pandas as pd
import shutil
import random 
import numpy as np 

# Three tasks. Three methods are written for (1) image ordering, (2) creating balanced training set, 
# (3) horizontal flip augmentation. Just uncomment the three lines to call the methods.
np.random.seed(740)
################### 1. Order the dataset in different folders and each of size 220x220 
def order_images(tval):
    if (tval=='train'): 
        images_folder = '../data/original/HAM10000_images'   # Need to modify if directory is different
        csv_file_path = '../data/original/HAM10000_metadata.csv' # Need to modify if directory is different
    else: 
        images_folder= '../data/original/ISIC2018_Task3_Test_Images' # Need to modify if directory is different
        csv_file_path = '../data/original/ISIC2018_Task3_Test_GroundTruth.csv' # Need to modify if directory is different
    output_folder = '../data/' + tval # modify according to the directory

    data = pd.read_csv(csv_file_path)

    # Loop through each row in the CSV
    for index, row in data.iterrows():
        image_id = row['image_id']
        dx = row['dx']
        source_image_path = os.path.join(images_folder, f"{image_id}.jpg")  

        target_folder = os.path.join(output_folder, str(dx))
        os.makedirs(target_folder, exist_ok=True)  # Create folder for dx if it doesn't exist
        target_image_path = os.path.join(target_folder, f"{image_id}.jpg")

        if os.path.exists(source_image_path):
            img = cv2.imread(source_image_path)
        
            if img is not None:
                resized_img = cv2.resize(img, (220, 220))
                cv2.imwrite(target_image_path, resized_img)
            else:
                print(f"Failed to load image {image_id}.")
        else:
            print(f"Image {image_id} not found in source folder.")

    print("Image ordering done.")

#uncomment the line below to order dataset
#order_images('train')           #argument: train or test--> which dataset to order?
#order_images('test_independent') 

##################  2. Create a balanced train set. Rule: 
# For each class:
# if #samples in that class >=2000: move 2000 random samples into a different directory
#else: augment data in that class until it's >=2000 and then move 2000 samples


def create_balanced_train_set(num=2000):
    source_folder = '../data/train'
    target_folder = '../data/balanced_upsampled_2000'

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        
        if os.path.isdir(subfolder_path):  # Check if it's a directory
            images = [img for img in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, img))]
            curNum = len(images)
            print(f"Current number of images in {subfolder}: {curNum}")
            
            while curNum < num:
                for image_name in os.listdir(subfolder_path):
                    image_path = os.path.join(subfolder_path, image_name)
                    img = cv2.imread(image_path)
                    
                    if img is not None:
                        h, w = img.shape[:2]  # Get image dimensions
                        
                        # Horizontal flip augmentation
                        flipped_img = cv2.flip(img, 1)
                        flipped_image_name = f"{os.path.splitext(image_name)[0]}_flipped{os.path.splitext(image_name)[1]}"
                        flipped_image_path = os.path.join(subfolder_path, flipped_image_name)
                        cv2.imwrite(flipped_image_path, flipped_img)

                        # Rotation
                        angle = random.randint(-30, 30)
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h))
                        rotated_image_name = f"{os.path.splitext(image_name)[0]}_rotated{os.path.splitext(image_name)[1]}"
                        rotated_image_path = os.path.join(subfolder_path, rotated_image_name)
                        cv2.imwrite(rotated_image_path, rotated_img)

                        # Shearing
                        shear_factor = random.uniform(-0.2, 0.2)
                        shear_matrix = np.array([[1, shear_factor, 0],
                                                [shear_factor, 1, 0]], dtype=np.float32)
                        sheared_img = cv2.warpAffine(img, shear_matrix, (w, h))
                        sheared_image_name = f"{os.path.splitext(image_name)[0]}_sheared{os.path.splitext(image_name)[1]}"
                        sheared_image_path = os.path.join(subfolder_path, sheared_image_name)
                        cv2.imwrite(sheared_image_path, sheared_img)

                        # Contrast adjustment
                        contrast_factor = random.uniform(0.8, 1.2)
                        contrast_img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)
                        contrast_image_name = f"{os.path.splitext(image_name)[0]}_contrast{os.path.splitext(image_name)[1]}"
                        contrast_image_path = os.path.join(subfolder_path, contrast_image_name)
                        cv2.imwrite(contrast_image_path, contrast_img)

                        # Saturation and Hue Adjustment
                        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        hsv_img = np.array(hsv_img, dtype=np.float64)
                        hsv_img[:, :, 1] *= random.uniform(0.8, 1.2)  # Adjust saturation
                        hsv_img[:, :, 0] += random.uniform(-10, 10)  # Adjust hue
                        hsv_img[:, :, 1:][hsv_img[:, :, 1:] > 255] = 255
                        hsv_img[hsv_img < 0] = 0
                        hsv_img = np.array(hsv_img, dtype=np.uint8)
                        adjusted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
                        sat_hue_image_name = f"{os.path.splitext(image_name)[0]}_sathue{os.path.splitext(image_name)[1]}"
                        sat_hue_image_path = os.path.join(subfolder_path, sat_hue_image_name)
                        cv2.imwrite(sat_hue_image_path, adjusted_img)

                        # Gaussian noise
                        mean = 0
                        stddev = random.uniform(5, 25)
                        gaussian_noise = np.random.normal(mean, stddev, img.shape).astype(np.uint8)
                        noisy_img = cv2.add(img, gaussian_noise)
                        noisy_image_name = f"{os.path.splitext(image_name)[0]}_noisy{os.path.splitext(image_name)[1]}"
                        noisy_image_path = os.path.join(subfolder_path, noisy_image_name)
                        cv2.imwrite(noisy_image_path, noisy_img)

                    else:
                        print(f"Failed to load image {image_name} in {subfolder}")

                images = [img for img in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, img))]
                curNum = len(images)
                print(f"Number of images in {subfolder} after augmentation: {curNum}")

            # Move a balanced set of images to the target folder
            selected_images = random.sample(images, num)
            os.makedirs(os.path.join(target_folder, subfolder), exist_ok=True)
            for image in selected_images:
                source_image_path = os.path.join(subfolder_path, image)
                target_image_path = os.path.join(target_folder, subfolder, image)
                shutil.move(source_image_path, target_image_path)

# Split into training and testing sets
def split_dataset(source_folder):
    train_folder = '../Data/train_balanced_upsampled_1600'
    test_folder = '../Data/test_balanced_upsampled_400'

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        if os.path.isdir(subfolder_path):
            images = [img for img in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, img))]
            random.shuffle(images)

            split_point = int(len(images) * 0.8)
            train_images = images[:split_point]
            test_images = images[split_point:]

            train_subfolder = os.path.join(train_folder, subfolder)
            test_subfolder = os.path.join(test_folder, subfolder)

            os.makedirs(train_subfolder, exist_ok=True)
            os.makedirs(test_subfolder, exist_ok=True)

            for image in train_images:
                source_image_path = os.path.join(subfolder_path, image)
                target_image_path = os.path.join(train_subfolder, image)
                shutil.copy2(source_image_path, target_image_path)

            for image in test_images:
                source_image_path = os.path.join(subfolder_path, image)
                target_image_path = os.path.join(test_subfolder, image)
                shutil.copy2(source_image_path, target_image_path)

    print("Dataset split into training and testing sets.")

#uncomment the line below to upsample and move balanced train data to another folder 
#create_balanced_train_set()
#split_dataset('../Data/balanced_upsampled_2000')