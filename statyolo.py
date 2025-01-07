import os
from collections import defaultdict

# Specify the path to the directory containing YOLO annotations
annotation_dir = r'D:\datasets\Collected\20241211\anno_source\20241222-1_yolo'  # Replace with your path

# Construct the path to the classes file within the annotation directory
classes_file = os.path.join(annotation_dir, 'classes.txt')

# Dictionary to count the number of bounding boxes for each category
category_bbox_count = defaultdict(int)
# Dictionary to count the number of images with specific numbers of bounding boxes
bbox_count_distribution = defaultdict(int)
# Count of total images
image_count = 0
# Count of total bounding boxes
total_boxes = 0

# Ensure the classes file exists
if not os.path.exists(classes_file):
    print(f"The file {classes_file} does not exist. Please check the path.")
    exit(1)

# Load the classes from the classes file
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Ensure the annotation directory exists
if not os.path.exists(annotation_dir):
    print(f"The directory {annotation_dir} does not exist. Please check the path.")
    exit(1)

# Read YOLO annotations from .txt files
for filename in os.listdir(annotation_dir):
    if filename.endswith('.txt') and filename != 'classes.txt':
        annotation_file = os.path.join(annotation_dir, filename)
        
        with open(annotation_file, 'r') as f:
            lines = f.readlines()

        bbox_count = len(lines)
        bbox_count_distribution[bbox_count] += 1

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            class_name = classes[class_id] if class_id < len(classes) else 'Unknown'
            
            category_bbox_count[class_name] += 1

        total_boxes += bbox_count
        image_count += 1

# Calculate the average number of bounding boxes per image
avg_boxes_per_image = total_boxes / image_count if image_count > 0 else 0

# Print the summary of results
print(f"Total number of images: {image_count}")
print(f"Total number of bounding boxes: {total_boxes}")
print(f"Average number of bounding boxes per image: {avg_boxes_per_image:.2f}")

print("\nNumber of bounding boxes for each category:")
for category, count in category_bbox_count.items():
    print(f"{category}: {count} bounding boxes")

print("\nDistribution of images by number of bounding boxes:")
for bbox_count, img_count in sorted(bbox_count_distribution.items()):
    print(f"Images with {bbox_count} bounding boxes: {img_count} images")