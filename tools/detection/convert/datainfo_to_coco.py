import os
import os.path as osp
import json
import argparse

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if 0 < length != len(vars):
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def load_annotations_saved(ann_file):
    """Load data_infos from saved json."""
    with open(ann_file) as f:
        data_infos = json.load(f)
    return data_infos


def _convert(data_infos, json_file):

    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    classes = data_infos[0]['CLASSES']

    for data_info in data_infos:
        if 'CLASSES' in data_info.keys():
            continue
        image_id = int(data_info['id'])
        width = data_info['width']
        height = data_info['height']
        file_name = data_info['filename']
        image = {'file_name': file_name, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)
        boxes = data_info['ann']['bboxes']
        labels = data_info['ann']['labels']
        for i, bbox in enumerate(boxes):
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            category = classes[labels[i]]
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert datainfo annotations to coco format')
    parser.add_argument('datainfo_path', help='self data_info path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('-o', '--out-dir', help='output path')  # annotations 保存文件夹
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.datainfo_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    data_infos = load_annotations_saved(devkit_path)
    out_file = osp.join(args.out_dir, args.dataset + '_ft_datainfo_to_coco.json')
    _convert(data_infos, out_file)

    print('Done!')


if __name__ == '__main__':
    main()
