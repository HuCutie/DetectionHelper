import os
import xml.etree.ElementTree as ET
from collections import defaultdict

# Specify the root directory containing annotation files
root_dir = r'D:\datasets\Collected\20241211\anno'  # Replace with your path

# Dictionary to count the number of bounding boxes for each label
label_bbox_count = defaultdict(int)
# Dictionary to count the number of bounding boxes in each image
image_box_count = defaultdict(int)
# Count of total images
image_count = 0
# Count of total bounding boxes
total_boxes = 0

# Ensure the path exists
if not os.path.exists(root_dir):
    print(f"The path {root_dir} does not exist. Please check the path.")
    exit(1)

# Traverse all directories and files
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.xml'):
            # Get the XML file path
            xml_file = os.path.join(dirpath, filename)
            
            try:
                # Parse the XML file
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except ET.ParseError as e:
                print(f"Error parsing file {xml_file}: {e}")
                continue

            # Count the number of bounding boxes in each image
            num_boxes_in_image = 0

            for obj in root.findall('object'):
                # Each <object> tag may have a <bndbox> tag
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    num_boxes_in_image += 1  # Increment for each found bounding box
                    
                    # Get the label name and update its bounding box count
                    label = obj.find('name').text
                    label_bbox_count[label] += 1

            # Update the count of bounding boxes per image
            image_box_count[num_boxes_in_image] += 1
            total_boxes += num_boxes_in_image
            image_count += 1

# Calculate the average number of bounding boxes per image
avg_boxes_per_image = total_boxes / image_count if image_count > 0 else 0

# Print the summary of results
print(f"Total number of images: {image_count}")
print(f"Total number of bounding boxes: {total_boxes}")
print(f"Average number of bounding boxes per image: {avg_boxes_per_image:.2f}")

print("\nNumber of bounding boxes for each category:")
for label, count in label_bbox_count.items():
    print(f"{label}: {count} bounding boxes")

print("\nDistribution of bounding boxes per image:")
for box_num, img_count in sorted(image_box_count.items()):
    print(f"Images with {box_num} bounding boxes: {img_count}")