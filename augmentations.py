import albumentations as A
import cv2
import os

# Set the input directory (where your original images are) and the output directory
input_directory = "/input_dir"
output_directory = "/output_dir"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List all image files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Define the number of augmentations (N)
N = 20

# Define augmentation pipeline
augmentations = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.7, brightness_limit=(-0.15, 0.15)),
        A.ShiftScaleRotate(shift_limit_x=(-0.01, 0.01), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.1, 0.1), rotate_limit=(-5, 5), p=0.9),
        A.GaussNoise(var_limit=(0, 25), p=0.8)
    ]
)

# Loop through each image and generate N augmentations
for image_file in image_files:
    image_path = os.path.join(input_directory, image_file)
    original_image = cv2.imread(image_path)
    
    # Generate N augmentations
    for i in range(N):
        augmented = augmentations(image=original_image)
        augmented_image = augmented["image"]
        
        # Define the output file path and save the augmented image
        output_file = os.path.join(output_directory, f"{os.path.splitext(image_file)[0]}_{i}.jpg")
        cv2.imwrite(output_file, augmented_image)
        print(f"Saved {output_file}")

print("Augmentations completed.")
