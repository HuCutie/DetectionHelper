import argparse
import os
import sys
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

category_set = dict()
image_set = set()
every_class_num = defaultdict(int)

category_item_id = -1


def xywhn2xyxy(box, size):
    box = list(map(float, box))
    size = list(map(float, size))
    xmin = (box[0] - box[2] / 2.) * size[0]
    ymin = (box[1] - box[3] / 2.) * size[1]
    xmax = (box[0] + box[2] / 2.) * size[0]
    ymax = (box[1] + box[3] / 2.) * size[1]
    return (xmin, ymin, xmax, ymax)


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    category_set[name] = category_item_id
    return category_item_id


def draw_box(img, objects, draw=True):
    for object in objects:
        category_name = object[0]
        every_class_num[category_name] += 1
        if category_name not in category_set:
            category_id = addCatItem(category_name)
        else:
            category_id = category_set[category_name]
        xmin = int(object[1][0])
        ymin = int(object[1][1])
        xmax = int(object[1][2])
        ymax = int(object[1][3])
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


def show_image(image_path, anno_path, save_path, plot_image=False):
    assert os.path.exists(image_path), "image path:{} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "annotation path:{} does not exists".format(anno_path)
    anno_file_list = [os.path.join(anno_path, file) for file in os.listdir(anno_path) if file.endswith(".txt")]
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(anno_path + "/classes.txt", 'r') as f:
        classes = f.readlines()

    category_id = dict((k, v.strip()) for k, v in enumerate(classes))

    for txt_file in tqdm(anno_file_list):
        if not txt_file.endswith('.txt') or 'classes' in txt_file:
            continue
        filename = txt_file.split(os.sep)[-1][:-3] + "jpg"
        image_set.add(filename)
        file_path = os.path.join(image_path, filename)
        if not os.path.exists(file_path):
            continue

        img = cv2.imread(file_path)
        if img is None:
            continue
        width = img.shape[1]
        height = img.shape[0]

        objects = []
        with open(txt_file, 'r') as fid:
            for line in fid.readlines():
                line = line.strip().split()
                category_name = category_id[int(line[0])]
                bbox = xywhn2xyxy((line[1], line[2], line[3], line[4]), (width, height))
                obj = [category_name, bbox]
                objects.append(obj)

        img = draw_box(img, objects)
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
    parser.add_argument('-ip', '--image-path', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('-ap', '--anno-path', type=str, required=True, help='Path to the YOLO .txt annotations folder(with classes.txt)')
    parser.add_argument('-sp', '--save-path', type=str, required=True, help='Path to save the labeled images')
    parser.add_argument('-p', '--plot-image', action='store_true', help='Whether to save the statistical result as an image')
    opt = parser.parse_args()

    print(opt)
    show_image(opt.image_path, opt.anno_path, opt.save_path, opt.plot_image)
    print(every_class_num)
    print("category nums: {}".format(len(category_set)))
    print("image nums: {}".format(len(image_set)))
    print("bbox nums: {}".format(sum(every_class_num.values())))