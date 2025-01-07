import argparse
import os
import cv2
from lxml import etree, objectify
from tqdm import tqdm

# Global variables
images_nums = 0
category_nums = 0
bbox_nums = 0

def save_anno_to_xml(filename, size, objs, save_path):
    """Save the annotation information to XML format"""
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder("DATA"),
        E.filename(filename),
        E.source(
            E.database("The VOC Database"),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size[1]),
            E.height(size[0]),
            E.depth(size[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        anno_tree.append(
            E.object(
                E.name(obj[0]),
                E.pose("Unspecified"),
                E.truncated(0),
                E.difficult(0),
                E.bndbox(
                    E.xmin(obj[1][0]),
                    E.ymin(obj[1][1]),
                    E.xmax(obj[1][2]),
                    E.ymax(obj[1][3])
                )
            )
        )
    anno_path = os.path.join(save_path, filename[:-3] + "xml")
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)

def xywhn2xyxy(bbox, size):
    """Convert YOLO bbox (xywhn) to (xyxy) format"""
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    xmax = (bbox[0] + bbox[2] / 2.) * size[1]
    ymax = (bbox[1] + bbox[3] / 2.) * size[0]
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def parse(anno_path, save_path, image_path):
    """Parse YOLO annotation files and save to VOC XML format"""
    global images_nums, category_nums, bbox_nums
    
    # Check if the provided paths exist
    assert os.path.exists(image_path), f"ERROR: {image_path} does not exist"
    assert os.path.exists(anno_path), f"ERROR: {anno_path} does not exist"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read category names
    category_set = []
    with open(os.path.join(anno_path, 'classes.txt'), 'r') as f:
        category_set = [line.strip() for line in f.readlines()]
    category_nums = len(category_set)
    category_id = {k: v for k, v in enumerate(category_set)}

    # Prepare image and annotation file lists
    images = [os.path.join(image_path, img) for img in os.listdir(image_path)]
    image_index = {os.path.splitext(os.path.basename(img))[0]: img for img in images}
    files = [os.path.join(anno_path, f) for f in os.listdir(anno_path) if f.endswith('.txt')]

    images_nums = len(images)

    # Iterate through each annotation file with a progress bar
    for file in tqdm(files, desc="Processing annotations", ncols=100):
        filename = os.path.splitext(os.path.basename(file))[0]
        
        # Skip if file is not an annotation file or is a class file
        if 'classes' in filename or not file.endswith('.txt'):
            continue

        # Find corresponding image
        if filename in image_index:
            img_path = image_index[filename]
            img = cv2.imread(img_path)
            shape = img.shape  # Get image shape (height, width, channels)
        
        else:
            continue
        
        objects = []
        with open(file, 'r') as fid:
            for line in fid.readlines():
                # Read and process each line (object information)
                parts = line.strip().split()
                category = int(parts[0])
                category_name = category_id[category]
                bbox = xywhn2xyxy(parts[1:], shape)
                objects.append([category_name, bbox])

        # Update bbox count and save annotations
        bbox_nums += len(objects)
        save_anno_to_xml(filename, shape, objects, save_path)

    # Print final statistics
    print(f'class nums: {category_nums}')
    print(f'image nums: {images_nums}')
    print(f'bbox nums: {bbox_nums}')

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, required=True, help='Path to YOLO .txt annotations')
    parser.add_argument('-sp', '--save-path', type=str, required=True, help='Path to save VOC .xml annotations')
    parser.add_argument('-ip', '--img-path', type=str, required=True, help='Path to YOLO images')
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path, opt.img_path)