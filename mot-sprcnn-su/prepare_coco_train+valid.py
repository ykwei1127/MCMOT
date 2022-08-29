import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import os
from pycocotools.coco import COCO
import json
import numpy as np

def load_json(json_path):
    img_dir = json_path.parent / 'data'
    img_anns = []
    box_anns = []
    coco=COCO(json_path)
    category_ids = coco.getCatIds(["car","truck","bus"])
    with open(json_path) as f:
        labels = json.load(f)

    for label in labels["images"]:
        file_name = label["file_name"]
        img_id = label["id"]
        img_path = str(img_dir / file_name)
        img = Image.open(img_path)
        if len(np.array(img).shape) < 3:
            continue   
        imgW, imgH = img.size

        annotation_ids = coco.getAnnIds(imgIds=label["id"], catIds=category_ids)
        anns = coco.loadAnns(annotation_ids)
        bboxes = []
        for ann in anns:
            # filter out bounding box of multiple objects in one box 
            if ann['iscrowd'] == 1:
                continue
            bx, by, bw, bh = ann['bbox']
            if ann["area"] > 0:
                bboxes.append((bx, by, bw, bh))
        if len(bboxes) > 1:
            img_anns.append(
                {
                    'id': img_id,
                    'file_name': img_path,
                    'width': imgW,
                    'height': imgH,
                    'video': 'coco-2017',
                }
            )
            for box in bboxes:
                box_anns.append(
                    {
                        'id': f'{json_path.parent.stem}-{len(box_anns)}',
                        # 'id': len(box_anns),
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': box,
                    }
                )
        
    return {
        'images': img_anns,
        'annotations': box_anns,
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }


def df2coco(df, video_dir):

    images = []
    annotations = []
    imgW, imgH = Image.open(video_dir / 'imgs' / '0001.jpg').size

    for (fid, group) in df.groupby('fid'):
        image_id = f'{video_dir.stem}-{fid}'
        images.append(
            {
                'id': image_id,
                'file_name': str(video_dir / 'imgs' / f'{fid:04d}.jpg'),
                'width': imgW,
                'height': imgH,
                'video': video_dir.stem,
            }
        )

        data = group[['x', 'y', 'w', 'h', 'tag']].values
        for (x, y, w, h, tag) in data.tolist():
            annotations.append(
                {
                    'id': f'{video_dir.stem}-{len(annotations)}',
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': (x, y, w, h),
                    'vis': 1.0,
                    'area': w * h,
                    'iscrowd': 0,
                }
            )
    return images, annotations


def normalize_ids(data):
    data = copy.deepcopy(data)

    image_ids = [ann['id'] for ann in data['images']]
    mapping = {x: i for i, x in enumerate(image_ids)}
    for img_ann in data['images']:
        img_ann['id'] = mapping[img_ann['id']]
    for box_ann in data['annotations']:
        box_ann['image_id'] = mapping[box_ann['image_id']]

    box_ids = [ann['id'] for ann in data['annotations']]
    mapping = {x: i for i, x in enumerate(box_ids)}
    for box_ann in data['annotations']:
        box_ann['id'] = mapping[box_ann['id']]

    assert len(set([ann['id'] for ann in data['images']])) == len(data['images'])
    assert len(set([ann['id'] for ann in data['annotations']])) == len(data['annotations'])
    assert len(set([ann['image_id'] for ann in data['annotations']])) == len(data['images'])

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_train', type=Path, default='../coco-2017/train/')
    parser.add_argument('--aic21_train', type=Path, default='../mcmt/dataset/train')
    parser.add_argument('--coco_validation', type=Path, default='../coco-2017/validation/')
    parser.add_argument('--aic21_validation', type=Path, default='../mcmt/dataset/validation')
    parser.add_argument('--jsons', type=Path, default='./jsons')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()

    assert args.coco_train.exists()
    assert args.aic21_train.exists()
    assert args.coco_validation.exists()
    assert args.aic21_validation.exists()

    # Prepare jsons
    if not os.path.exists(args.jsons):
        args.jsons.mkdir(parents=True)

    coco_train_data = load_json(args.coco_train / 'labels.json')
    # with open(args.jsons / 'coco_train.json', 'w') as f:
    #     json.dump(normalize_ids(coco_train_data), f)
    print('coco_train: {} {}'.format(len(coco_train_data['images']), len(coco_train_data['annotations'])))

    coco_validation_data = load_json(args.coco_validation / 'labels.json')
    # with open(args.jsons / 'coco_validation.json', 'w') as f:
    #     json.dump(normalize_ids(coco_validation_data), f)
    print('coco_validation: {} {}'.format(len(coco_validation_data['images']), len(coco_validation_data['annotations'])))

    # coco_all = {
    #     'images': [*coco_train_data['images'], *coco_validation_data['images']],
    #     'annotations': [*coco_train_data['annotations'], *coco_validation_data['annotations']],
    #     'categories': [{'id': 1, 'name': 'vehicle'}],
    # }
    # with open(args.jsons / 'coco_all.json', 'w') as f:
    #     json.dump(normalize_ids(coco_all), f)
    # print('coco_all: {} {}'.format(len(coco_all['images']), len(coco_all['annotations'])))

    aic21_train_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.aic21_train.glob('**/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        aic21_train_data['images'].extend(imgs)
        aic21_train_data['annotations'].extend(anns)
    # with open(args.jsons / 'aic21_train.json', 'w') as f:
    #     json.dump(normalize_ids(aic21_train_data), f)
    print('aic21_train: {} {}'.format(len(aic21_train_data['images']), len(aic21_train_data['annotations'])))

    data_all = {
        'images': [*coco_train_data['images'], *coco_validation_data['images'], *aic21_train_data['images']],
        'annotations': [*coco_train_data['annotations'], *coco_validation_data['annotations'],
                        *aic21_train_data['annotations']],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    with open(args.jsons / 'coco+aic_train.json', 'w') as f:
        json.dump(normalize_ids(data_all), f)
    print('data_all: {} {}'.format(len(data_all['images']), len(data_all['annotations'])))

    aic21_validation_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.aic21_validation.glob('S02/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        aic21_validation_data['images'].extend(imgs)
        aic21_validation_data['annotations'].extend(anns)
    with open(args.jsons / 'aic21_validation_s02.json', 'w') as f:
        json.dump(normalize_ids(aic21_validation_data), f)
    print('aic21_validation_s02: {} {}'.format(len(aic21_validation_data['images']),
                                           len(aic21_validation_data['annotations'])))

