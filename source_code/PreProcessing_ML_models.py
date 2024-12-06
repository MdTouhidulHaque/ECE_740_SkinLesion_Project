import os
import cv2
import pandas as pd
import shutil
import random
import numpy as np

np.random.seed(740)
# 1. Order the dataset
# Do not do it if ordering is conducted in preprocessing of CNN
def order_images():
    images_folder = '../data/original/HAM10000_images'
    csv_file_path = '../data/original/HAM10000_metadata.csv'
    output_folder = '../data/train'

    data = pd.read_csv(csv_file_path)

    # Loop through each row in the CSV
    for _, row in data.iterrows():
        image_id = row['image_id']
        dx = row['dx']

        source_image_path = os.path.join(images_folder, f"{image_id}.jpg")
        target_folder = os.path.join(output_folder, str(dx))
        os.makedirs(target_folder, exist_ok=True)
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

# 2. Create a balanced dataset
def create_balanced_dataset():
    source_folder = '../data/train'
    balanced_folder = '../data/balanced_200'

    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            images = [img for img in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, img))]
            selected_images = random.sample(images, 100)
            target_subfolder = os.path.join(balanced_folder, subfolder)
            os.makedirs(target_subfolder, exist_ok=True)

            for image in selected_images:
                source_image_path = os.path.join(subfolder_path, image)
                target_image_path = os.path.join(target_subfolder, image)
                shutil.copy2(source_image_path, target_image_path)

    print("Balanced dataset created.")

# 3. Augment the dataset
def augment_data(source_folder):
    for subfolder in os.listdir(source_folder):
        subfolder_path = os.path.join(source_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for image_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_name)
                img = cv2.imread(image_path)
                if img is not None:
                    flipped_img = cv2.flip(img, 1)
                    flipped_image_name = f"{os.path.splitext(image_name)[0]}_flipped{os.path.splitext(image_name)[1]}"
                    flipped_image_path = os.path.join(subfolder_path, flipped_image_name)
                    cv2.imwrite(flipped_image_path, flipped_img)
                else:
                    print(f"Failed to load image {image_name} in {subfolder}")

    print("Horizontal flip augmentation done.")

# 4. Split into training and testing sets
def split_dataset(source_folder):
    train_folder = '../data/train_balanced_160'
    test_folder = '..//data/test_balanced_40'

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

# Execute the steps in sequence
#order_images()
create_balanced_dataset()
augment_data('../data/balanced_200')
split_dataset('../data/balanced_200')
