import argparse
import json
import os
from datetime import datetime
import cv2
from tqdm import tqdm

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

image_id = 0
annotation_id = 0

def addCatItem(category_dict):
    """Add category items to the coco dictionary"""
    for k, v in category_dict.items():
        category_item = {
            'supercategory': 'none',
            'id': int(k),
            'name': v
        }
        coco['categories'].append(category_item)

def addImgItem(file_name, size):
    """Add image item to the coco dictionary"""
    global image_id
    image_id += 1
    image_item = {
        'id': image_id,
        'file_name': file_name,
        'width': size[1],
        'height': size[0],
        'license': None,
        'flickr_url': None,
        'coco_url': None,
        'date_captured': str(datetime.today())
    }
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    """Add annotation item to the coco dictionary"""
    global annotation_id
    annotation_item = {
        'segmentation': [[
            bbox[0], bbox[1], # left_top
            bbox[0], bbox[1] + bbox[3], # left_bottom
            bbox[0] + bbox[2], bbox[1] + bbox[3], # right_bottom
            bbox[0] + bbox[2], bbox[1] # right_top
        ]],
        'area': bbox[2] * bbox[3],
        'iscrowd': 0,
        'ignore': 0,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id
    }
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def xywhn2xywh(bbox, size):
    """Convert normalized coordinates to absolute coordinates"""
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    return list(map(int, (xmin, ymin, w, h)))

def parse(anno_path, save_path, image_path):
    """Parse YOLO annotations and convert to COCO format"""
    assert os.path.exists(image_path), f"ERROR: {image_path} does not exist"
    assert os.path.exists(anno_path), f"ERROR: {anno_path} does not exist"

    # Read category names
    with open(os.path.join(anno_path, 'classes.txt'), 'r') as f:
        category_set = {k: v.strip() for k, v in enumerate(f.readlines())}
    addCatItem(category_set)

    # Get all image and annotation files
    images = {os.path.splitext(i)[0]: os.path.join(image_path, i) for i in os.listdir(image_path)}
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path) if i.endswith('.txt')]

    # Use tqdm for progress bar when processing annotation files
    for file in tqdm(files, desc="Processing annotation files", ncols=100):
        filename = os.path.splitext(os.path.basename(file))[0]
        if filename in images:
            img = cv2.imread(images[filename])
            shape = img.shape
            current_image_id = addImgItem(os.path.basename(images[filename]), shape)
        else:
            continue
        
        with open(file, 'r') as fid:
            for line in fid.readlines():
                category, x_center, y_center, w, h = map(float, line.strip().split())
                category_id = int(category)
                category_name = category_set[category_id]
                bbox = xywhn2xywh((x_center, y_center, w, h), shape)
                addAnnoItem(category_name, current_image_id, category_id, bbox)

    # Save COCO format data
    with open(save_path, 'w') as json_file:
        json.dump(coco, json_file)
    print(f"class nums: {len(coco['categories'])}")
    print(f"image nums: {len(coco['images'])}")
    print(f"bbox nums: {len(coco['annotations'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, required=True, help='Path to YOLO annotation folder')
    parser.add_argument('-sp', '--save-path', type=str, required=True, help='Path to save the generated COCO JSON file')
    parser.add_argument('-ip', '--img-path', type=str, required=True, help='Path to YOLO images folder')
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path, opt.img_path)