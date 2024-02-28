from pycocotools.coco import COCO
import os
from lxml import etree, objectify
import shutil
from tqdm import tqdm
import argparse

def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes

def save_anno_to_xml(filename, size, objs, save_path):
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
            E.width(size['width']),
            E.height(size['height']),
            E.depth(size['depth'])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose("Unspecified"),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[1]),
                E.ymin(obj[2]),
                E.xmax(obj[3]),
                E.ymax(obj[4])
            )
        )
        anno_tree.append(anno_tree2)
    anno_path = os.path.join(save_path, filename[:-3] + "xml")
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)

def load_coco(anno_file, xml_save_path):
    if os.path.exists(xml_save_path):
        shutil.rmtree(xml_save_path)
    os.makedirs(xml_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()
    for imgId in tqdm(imgIds):
        size = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
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
        save_anno_to_xml(filename, size, objs, xml_save_path)


def parse(anno_path, xmls_save_path):
    assert os.path.exists(anno_path), "data dir:{} does not exits".format(anno_path)

    if os.path.isdir(anno_path):
        data_types = ['train2017', 'val2017']
        for data_type in data_types:
            ann_file = 'instances_{}.json'.format(data_type)
            anno_path = os.path.join(anno_path, ann_file)
            xmls_save_path = os.path.join(xmls_save_path, data_type)
            load_coco(anno_path, xmls_save_path)
    elif os.path.isfile(anno_path):
        anno_file = anno_path
        load_coco(anno_file, xmls_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ap', '--anno-path', type=str, default='/workspace/0test/labelsnew/coco/voc2coco.json', help='coco .json path or coco annotations dir path')
    parser.add_argument('-sp', '--save-path', type=str, default='/workspace/0test/labelsnew/voc/coco2voc', help='voc .xml save path')
    opt = parser.parse_args()
    
    print(opt)
    parse(opt.anno_path, opt.save_path)