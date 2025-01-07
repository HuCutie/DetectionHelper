from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import argparse

# Global statistics variables
images_nums = 0
category_nums = 0
bbox_nums = 0

def catid2name(coco):
    """Convert category IDs to category names"""
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes

def xyxy2xywhn(object, width, height):
    """Convert bounding box from (x1, y1, x2, y2) to normalized (x, y, w, h)"""
    cat_id = object[0]
    xn = object[1] / width
    yn = object[2] / height
    wn = object[3] / width
    hn = object[4] / height
    out = "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(cat_id, xn, yn, wn, hn)
    return out

def save_anno_to_txt(images_info, save_path):
    """Save annotations in YOLO format (txt)"""
    filename = images_info['filename']
    txt_name = filename[:-3] + "txt"
    with open(os.path.join(save_path, txt_name), "w") as f:
        for obj in images_info['objects']:
            line = xyxy2xywhn(obj, images_info['width'], images_info['height'])
            f.write("{}\n".format(line))

def load_coco(anno_file, txt_save_path):
    """Load COCO annotations and save them in YOLO format"""
    global images_nums, category_nums, bbox_nums
    
    if os.path.exists(txt_save_path):
        shutil.rmtree(txt_save_path)
    os.makedirs(txt_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    category_nums = len(classes)  # Number of categories
    
    # Write classes to a file
    with open(os.path.join(txt_save_path, "classes.txt"), 'w') as f:
        for id in classes:
            f.write("{}\n".format(classes[id]))

    # Iterate over all images
    for imgId in tqdm(imgIds, desc="Processing images", ncols=100):
        info = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        width = img['width']
        height = img['height']
        info['filename'] = filename
        info['width'] = width
        info['height'] = height
        
        # Retrieve annotations for this image
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            bbox = list(map(float, ann['bbox']))
            xc = bbox[0] + bbox[2] / 2.
            yc = bbox[1] + bbox[3] / 2.
            w = bbox[2]
            h = bbox[3]
            obj = [ann['category_id'], xc, yc, w, h]
            objs.append(obj)
        
        # Update statistics
        bbox_nums += len(objs)
        images_nums += 1
        
        # Save the annotations in YOLO format
        info['objects'] = objs
        save_anno_to_txt(info, txt_save_path)

def parse(json_path, txt_save_path):
    """Parse COCO annotations and convert them to YOLO format"""
    assert os.path.exists(json_path), f"ERROR: {json_path} does not exist"
    
    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)

    assert json_path.endswith('json'), f"ERROR: {json_path} is not a JSON file!"

    load_coco(json_path, txt_save_path)

    # Print statistics at the end
    print(f'class nums: {category_nums}')
    print(f'image nums: {images_nums}')
    print(f'bbox nums: {bbox_nums}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, required=True, help='Path to COCO annotations')
    parser.add_argument('-sp', '--save-path', type=str, required=True, help='Path to save YOLO .txt annotations')
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path)