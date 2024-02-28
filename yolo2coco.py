import argparse
import json
import os
from datetime import datetime

import cv2

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

image_id = 000000
annotation_id = 0

def addCatItem(category_dict):
    for k, v in category_dict.items():
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(k)
        category_item['name'] = v
        coco['categories'].append(category_item)

def addImgItem(file_name, size):
    global image_id
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size[1]
    image_item['height'] = size[0]
    image_item['license'] = None
    image_item['flickr_url'] = None
    image_item['coco_url'] = None
    image_item['date_captured'] = str(datetime.today())
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def xywhn2xywh(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    box = (xmin, ymin, w, h)
    return list(map(int, box))

def parse(anno_path, save_path, image_path, json_name='train.json'):
    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    category_set = []
    with open(anno_path + '/classes.txt', 'r') as f:
        for i in f.readlines():
            category_set.append(i.strip())
    category_id = dict((k, v) for k, v in enumerate(category_set))
    addCatItem(category_id)

    images = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path)]
    images_index = dict((v.split(os.sep)[-1][:-4], k) for k, v in enumerate(images))
    for file in files:
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in images_index:
            index = images_index[file.split(os.sep)[-1][:-4]]
            img = cv2.imread(images[index])
            shape = img.shape
            filename = images[index].split(os.sep)[-1]
            current_image_id = addImgItem(filename, shape)
        else:
            continue
        with open(file, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split()
                category = int(i[0])
                category_name = category_id[category]
                bbox = xywhn2xywh((i[1], i[2], i[3], i[4]), shape)
                addAnnoItem(category_name, current_image_id, category, bbox)

    json.dump(coco, open(save_path, 'w'))
    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("bbox nums:{}".format(len(coco['annotations'])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='/workspace/yolo/labels', help='yolo .txt path')
    parser.add_argument('-sp', '--save-path', type=str, default='/workspace/coco/annotations/instances_train2017.json', help='coco .json save path')
    parser.add_argument('-ip', '--img-path', type=str, default='/workspace/yolo/images', help='yolo images path')
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path, opt.img_path)