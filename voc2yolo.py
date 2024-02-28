import os
import argparse
from lxml import etree
from tqdm import tqdm

image_set = set()
bbox_nums = 0

def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def xyxy2xywhn(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xc = (bbox[0] + (bbox[2] - bbox[0]) / 2.) / size[0]
    yc = (bbox[1] + (bbox[3] - bbox[1]) / 2.) / size[1]
    wn = (bbox[2] - bbox[0]) / size[0]
    hn = (bbox[3] - bbox[1]) / size[1]
    return (xc, yc, wn, hn)

def parser_info(info: dict, class_indices):
    filename = info['annotation']['filename']
    image_set.add(filename)
    objects = []
    width = int(info['annotation']['size']['width'])
    height = int(info['annotation']['size']['height'])
    for obj in info['annotation']['object']:
        obj_name = obj['name']
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        bbox = xyxy2xywhn((xmin, ymin, xmax, ymax), (width, height))
        if class_indices is not None:
            obj_category = class_indices[obj_name]
            object = [obj_category, bbox]
            objects.append(object)

    return filename, objects

def parse(voc_dir, save_dir, categories):
    assert os.path.exists(voc_dir), "ERROR {} does not exists".format(voc_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    xml_files = [os.path.join(voc_dir, i) for i in os.listdir(voc_dir) if os.path.splitext(i)[-1] == '.xml']

    with open(save_dir + "/classes.txt", 'w') as classes_file:
        for cat in categories:
            classes_file.write("{}\n".format(cat))

    class_indices = dict((v, k) for k, v in enumerate(categories))

    xml_files = tqdm(xml_files)
    for xml_file in xml_files:
        with open(xml_file) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        info_dict = parse_xml_to_dict(xml)
        filename, objects = parser_info(info_dict, class_indices=class_indices)
        if len(objects) != 0:
            global bbox_nums
            bbox_nums += len(objects)
            with open(save_dir + "/" + filename.split(".")[0] + ".txt", 'w') as f:
                for obj in objects:
                    f.write(
                        "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(obj[0], obj[1][0], obj[1][1], obj[1][2], obj[1][3]))

if __name__ == '__main__':
    # Define your categories
    categories = ['Armored vehicle', 'Transport vehicle', 'tent', 'car']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='/workspace/0test/labels/voc', help='voc .xml path')
    parser.add_argument('-sp', '--save-path', type=str, default='/workspace/0test/labelsnew/yolo/voc2yolo', help='yolo .txt save path')
    parser.add_argument('-c', '--categories', nargs='+', default=categories, help='categories in your dataset')
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path, opt.categories)