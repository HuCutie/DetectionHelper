import argparse
import os
from collections import defaultdict
from xml import etree
from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

category_set = dict()
image_set = set()
every_class_num = defaultdict(int)

category_item_id = -1

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    category_set[name] = category_item_id
    return category_item_id

def draw_box(img, objects):
    for object in objects:
        category_name = object[0]
        every_class_num[category_name] += 1
        if category_name not in category_set:
            category_id = addCatItem(category_name)
        else:
            category_id = category_set[category_name]
        xmin = int(object[1])
        ymin = int(object[2])
        xmax = int(object[3])
        ymax = int(object[4])

        def hex2rgb(h):  # rgb order (PIL)
            return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

        hex = ('FF0000', '00FF00', '0000FF', 'FFA500', 'FF00FF', '00FFFF', 'FFD700', '800080', '008000', '800000',
               '008080', 'FF4500', '9400D3', '008B8B', 'FF1493', '32CD32', '1E90FF', 'FF69B4', 'FF6347', '20B2AA')


        palette = [hex2rgb('#' + c) for c in hex]
        n = len(palette)
        c = palette[int(category_id) % n]
        color = (c[2], c[1], c[0])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color)
        cv2.putText(img, category_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img

def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes

def show_image(image_path, anno_path, save_path, plot_image=False):
    assert os.path.exists(image_path), "image path:{} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "annotation path:{} does not exists".format(anno_path)
    if not anno_path.endswith(".json"):
        raise RuntimeError("ERROR {} dose not a json file".format(anno_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    coco = COCO(anno_path)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    for imgId in tqdm(imgIds):
        size = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        image_set.add(filename)
        width = img['width']
        height = img['height']
        size['width'] = width
        size['height'] = height
        size['depth'] = 3
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            # bbox:[x,y,w,h]
            bbox = list(map(int, ann['bbox']))
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            obj = [object_name, xmin, ymin, xmax, ymax]
            objs.append(obj)

        file_path = os.path.join(image_path, filename)
        img = cv2.imread(file_path)
        if img is None:
            continue
        img = draw_box(img, objs)
        res_path = os.path.join(save_path, filename)
        cv2.imwrite(res_path, img)
        
    if plot_image:
        plt.bar(range(len(every_class_num)), every_class_num.values(), align='center')
        plt.xticks(range(len(every_class_num)), every_class_num.keys(), rotation=0)
        for index, (i, v) in enumerate(every_class_num.items()):
            plt.text(x=index, y=v, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('class distribution')

        res_path = os.path.join(save_path, '00000_class_distribution.png')
        plt.savefig(res_path)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', type=str, required=True, help='Path to the directory containing the images')
    parser.add_argument('-ap', '--anno-path', type=str, required=True, help='Path to the COCO .json annotation file')
    parser.add_argument('-sp', '--save-path', type=str, required=True, help='Path to save the labeled images')
    parser.add_argument('-p', '--plot-image', action='store_true', help='Whether to save the statistical result as an image')
    opt = parser.parse_args()

    print(opt)
    show_image(opt.image_path, opt.anno_path, opt.save_path, opt.plot_image)
    print(every_class_num)
    print("category nums: {}".format(len(category_set)))
    print("image nums: {}".format(len(image_set)))
    print("bbox nums: {}".format(sum(every_class_num.values())))