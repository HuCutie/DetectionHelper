import os
import json
from collections import defaultdict

# Specify the path to the COCO annotation JSON file
annotation_file = r"C:\Users\husma\Downloads\annotations\instances_val2017.json"  # Replace with your path

# Dictionaries to count the number of bounding boxes for each category and image
category_bbox_count = defaultdict(int)
image_box_count = defaultdict(int)
# Count of total images
image_count = 0
# Count of total bounding boxes
total_boxes = 0

# Ensure the annotation file exists
if not os.path.exists(annotation_file):
    print(f"The file {annotation_file} does not exist. Please check the path.")
    exit(1)

# Load the COCO annotation JSON file
with open(annotation_file, 'r') as f:
    data = json.load(f)

# Extract information about images, annotations, and categories
images = data.get('images', [])
annotations = data.get('annotations', [])
categories = data.get('categories', [])

# Create a mapping of category IDs to category names
category_id_to_name = {category['id']: category['name'] for category in categories}

# Count the number of bounding boxes and images
for annotation in annotations:
    category_id = annotation['category_id']
    image_id = annotation['image_id']
    category_name = category_id_to_name.get(category_id, 'Unknown')
    
    category_bbox_count[category_name] += 1
    image_box_count[image_id] += 1

# Update total counts
image_count = len(images)
total_boxes = len(annotations)

# Calculate the average number of bounding boxes per image
avg_boxes_per_image = total_boxes / image_count if image_count > 0 else 0

# Calculate how many images have a specific number of bounding boxes
image_bbox_distribution = defaultdict(int)
for bbox_count in image_box_count.values():
    image_bbox_distribution[bbox_count] += 1

# Print the summary of results
print(f"Total number of images: {image_count}")
print(f"Total number of bounding boxes: {total_boxes}")
print(f"Average number of bounding boxes per image: {avg_boxes_per_image:.2f}")

print("\nNumber of bounding boxes for each category:")
for category, count in category_bbox_count.items():
    print(f"{category}: {count} bounding boxes")

print("\nDistribution of bounding boxes per image:")
# Sort by the number of bounding boxes (box_num)
for box_num, img_count in sorted(image_bbox_distribution.items()):
    print(f"Images with {box_num} bounding boxes: {img_count}")