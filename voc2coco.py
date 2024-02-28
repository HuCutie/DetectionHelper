import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime
import argparse

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 000000
annotation_id = 0

def addCatItems(categories):
    global category_item_id
    category_ids = []
    for category in categories:
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item_id += 1
        category_item['id'] = category_item_id
        category_item['name'] = category
        coco['categories'].append(category_item)
        category_set[category] = category_item_id
        category_ids.append(category_item_id)

    return category_ids

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
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

def read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file, 'r') as f:
        for line in f.readlines():
            ids.append(line.strip())
    return ids

def parse(anno_path, save_path, categories, split='train'):
    assert os.path.exists(anno_path), "anno path:{} does not exist".format(anno_path)
    labelfile = split + ".txt"
    image_sets_file = os.path.join(anno_path, "ImageSets", "Main", labelfile)
    xml_files_list = []
    if os.path.isfile(image_sets_file):
        ids = read_image_ids(image_sets_file)
        xml_files_list = [os.path.join(anno_path, "Annotations", f"{i}.xml") for i in ids]
    elif os.path.isdir(anno_path):
        xml_dir = anno_path
        xml_list = os.listdir(xml_dir)
        xml_files_list = [os.path.join(xml_dir, i) for i in xml_list]

    # mapping categories to indices
    addCatItems(categories)
    
    for xml_file in xml_files_list:
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = dict()
        size['width'] = None
        size['height'] = None

        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        file_name = root.findtext('filename')
        assert file_name is not None, "filename is not in the file"

        size_info = root.findall('size')
        assert size_info is not None, "size is not in the file"
        for subelem in size_info[0]:
            size[subelem.tag] = int(subelem.text)

        if file_name is not None and size['width'] is not None and file_name not in image_set:
            current_image_id = addImgItem(file_name, size)
            print('add image with name: {}\tand\tsize: {}'.format(file_name, size))
        elif file_name in image_set:
            raise Exception('file_name duplicated')
        else:
            raise Exception("file name:{}\t size:{}".format(file_name, size))

        object_info = root.findall('object')
        if len(object_info) == 0:
            continue

        for object in object_info:
            object_name = object.findtext('name')
            current_category_id = category_set[object_name]

            bndbox = dict()
            bndbox['xmin'] = None
            bndbox['xmax'] = None
            bndbox['ymin'] = None
            bndbox['ymax'] = None
            # box:[xmin,ymin,xmax,ymax]
            bndbox_info = object.findall('bndbox')
            for box in bndbox_info[0]:
                bndbox[box.tag] = int(box.text)

            if bndbox['xmin'] is not None:
                if object_name is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_image_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_category_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                bbox = []
                # x
                bbox.append(bndbox['xmin'])
                # y
                bbox.append(bndbox['ymin'])
                # w
                bbox.append(bndbox['xmax'] - bndbox['xmin'])
                # h
                bbox.append(bndbox['ymax'] - bndbox['ymin'])
                print('add annotation with object_name:{}\timage_id:{}\tcat_id:{}\tbbox:{}'.format(object_name,
                                                                                                   current_image_id,
                                                                                                   current_category_id,
                                                                                                   bbox))
                addAnnoItem(object_name, current_image_id, current_category_id, bbox)

    json_parent_dir = os.path.dirname(save_path)
    if not os.path.exists(json_parent_dir):
        os.makedirs(json_parent_dir)
    json.dump(coco, open(save_path, 'w'))
    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("bbox nums:{}".format(len(coco['annotations'])))

if __name__ == '__main__':
    # Define your categories
    categories = ['Armored vehicle', 'Transport vehicle', 'tent', 'car']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='/workspace/0test/labels/voc', help='voc .xml path')
    parser.add_argument('-sp', '--save-path', type=str, default='/workspace/0test/labelsnew/coco/voc2coco.json', help='coco .json save path')
    parser.add_argument('-c', '--categories', nargs='+', default=categories, help='categories in your dataset')
    parser.add_argument('-t', '--type', type=str, default='train', help='only use in voc2012/2007')    
    opt = parser.parse_args()

    print(opt)
    parse(opt.anno_path, opt.save_path, opt.categories, opt.type)