import os
import shutil

# Define dataset labels
DATASET_LABELS = {
    '0': "Bread",
    '1': "Dairy product",
    '2': "Dessert",
    '3': "Egg",
    '4': "Fried food",
    '5': "Meat",
    '6': "Noodles/Pasta",
    '7': "Rice",
    '8': "Seafood",
    '9': "Soup",
    '10': "Vegetable/Fruit"
}

# Define root folder paths
root_folder = "food11 dataset"
subfolders = ["evaluation", "training", "validation"]

# Create class folders in each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(root_folder, subfolder)
    for label_key, label_name in DATASET_LABELS.items():
        class_folder_path = os.path.join(subfolder_path, label_name)
        os.makedirs(class_folder_path, exist_ok=True)

    # Move images to respective class folders
    for filename in os.listdir(subfolder_path):
        if os.path.isfile(os.path.join(subfolder_path, filename)) and "_" in filename:
            label_key = filename.split("_")[0]
            if label_key in DATASET_LABELS:
                label_name = DATASET_LABELS[label_key]
                src_path = os.path.join(subfolder_path, filename)
                dest_path = os.path.join(subfolder_path, label_name, filename)
                shutil.move(src_path, dest_path)
                print(f"Moved {filename} to {label_name} folder in {subfolder}")

print("Images organized into class folders successfully.")